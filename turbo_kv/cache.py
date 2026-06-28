"""``TurboKVCache`` — int4 rotated KV cache (PLAN §M3, correctness/memory MVP).

Layout per layer:

* a **recent BF16 window** of the last ``residual_length`` tokens, kept exact
  (recent tokens dominate attention and are the most quantization-sensitive);
* **older tokens compressed**: each ``head_dim`` vector is rotated (energy-spread),
  per-token affine-quantized to 4 bits, and bit-packed to 0.5 bytes/value.

On :meth:`update` the cache returns the *full* reconstructed BF16 K/V (older
tokens are unpacked → dequantized → inverse-rotated, then concatenated with the
window), so attention runs unchanged — this is the M3 MVP. The memory win is in
what is *stored*, measured by :meth:`memory_bytes` and the allocator.

``torch`` / ``transformers`` are imported lazily so the module imports on CPU.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from turbo_kv import packing as P
from turbo_kv import rotations as R


def _try_cache_base():
    """Return ``transformers.Cache`` if available, else ``object`` (CPU import)."""
    try:
        from transformers.cache_utils import Cache

        return Cache
    except Exception:  # pragma: no cover - allows import without transformers
        return object


class TurboKVCache(_try_cache_base()):
    """KV cache with a BF16 recent window + int4 rotated compressed history."""

    def __init__(
        self,
        *,
        residual_length: int = 128,
        rotation: str = "dense",
        head_dim: int = 64,
        bits: int = 4,
        seed: int = 0,
        sink_length: int = 0,
        key_quant: str = "per_token",
        key_group_size: int = 32,
        pre_rope: bool = False,
        bf16_layers: int = 0,
        key_outliers: int = 0,
    ) -> None:
        super().__init__()
        if bits != 4:
            raise ValueError("TurboKVCache MVP supports int4 only (bits=4)")
        if key_quant not in ("per_token", "per_channel"):
            raise ValueError("key_quant must be 'per_token' or 'per_channel'")
        self.residual_length = int(residual_length)
        self.rotation_kind = rotation
        self.head_dim = int(head_dim)
        self.bits = int(bits)
        self.seed = int(seed)
        self.sink_length = int(sink_length)
        # Key quantization axis: per-token (original) or per-channel (KIVI-style,
        # the fix for outlier key channels). Values stay per-token. Per-channel
        # keys are quantized over blocks of ``key_group_size`` evicted tokens.
        self.key_quant = key_quant
        self.key_group_size = int(key_group_size)
        self._key_axis = "channel" if key_quant == "per_channel" else "token"
        # Pre-RoPE key quantization (KVQuant fix): keys are stored/quantized in
        # their RAW pre-rotary basis (RoPE injects position-dependent variation
        # that hurts quantization) and RoPE is re-applied to the reconstructed
        # keys at attention time using per-token positions + the rotary base
        # frequencies. Values are unaffected by RoPE.
        self.pre_rope = bool(pre_rope)
        self._rope_inv_freq: Optional[Any] = None  # [D/2] rotary base freqs
        # Per-layer bit allocation (QJL "more bits for early layers"): the first
        # ``bf16_layers`` layers are the most quantization-sensitive (M2 found
        # layer 0 a huge inner-product RMSE outlier), so keep their KV entirely in
        # BF16 (never evict/compress) and int4 the rest. ``bf16_layers=0`` is
        # byte-for-byte identical to the all-int4 cache.
        self.bf16_layers = int(bf16_layers)
        # Dense-and-sparse outliers (KVQuant + QJL): for the PER-TOKEN key path
        # only, keep the top ``key_outliers`` coordinates per key vector in fp16
        # (a sparse side-channel) and quantize the dense rest, so a few extreme
        # coordinates no longer inflate the whole token's int4 scale (the M4
        # per-token key failure mode). No-op for per_channel keys (per-channel
        # already isolates channel outliers with its own scale) and for values.
        # ``key_outliers=0`` is byte-for-byte identical to the all-dense cache.
        self.key_outliers = int(key_outliers)

        self._rot: Optional[R.Rotation] = None  # built lazily on first update
        # Per-layer compressed store (packed codes + per-token scale/zero).
        self._cK: List[Optional[dict]] = []
        self._cV: List[Optional[dict]] = []
        # Per-layer BF16 recent window.
        self._wK: List[Optional[Any]] = []
        self._wV: List[Optional[Any]] = []
        # Per-layer BF16 attention-sink buffer (first ``sink_length`` tokens, kept
        # exact — sink tokens receive massive attention and are catastrophic to
        # quantize; StreamingLLM/KIVI keep them in full precision).
        self._sK: List[Optional[Any]] = []
        self._sV: List[Optional[Any]] = []
        # Per-layer int32 positions buffer (pre_rope only): absolute token
        # position of each stored token, in arrival order. Cheap (T int32 values)
        # and preserves the memory win; used to re-apply RoPE on reconstruction.
        self._pos: List[Optional[Any]] = []
        self._seen_tokens = 0  # total tokens seen by layer 0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _is_bf16_layer(self, layer_idx: int) -> bool:
        """True if ``layer_idx`` is kept entirely in BF16 (no quantization)."""
        return layer_idx < self.bf16_layers

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self._wK) <= layer_idx:
            self._cK.append(None)
            self._cV.append(None)
            self._wK.append(None)
            self._wV.append(None)
            self._sK.append(None)
            self._sV.append(None)
            self._pos.append(None)

    @staticmethod
    def _rotate_half(x):
        """HuggingFace ``rotate_half``: split last dim in halves, rotate."""
        import torch

        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, k, positions, out_dtype):
        """Apply RoPE to a full reconstructed pre-RoPE key ``[B,H,T,D]``.

        cos/sin are built EXACTLY like HuggingFace from the stored positions and
        the rotary base frequencies ``inv_freq`` ``[D/2]``.
        """
        import torch

        inv_freq = self._rope_inv_freq
        assert inv_freq is not None, (
            "pre_rope=True requires rope_inv_freq in cache_kwargs on first update")
        pos = positions.to(device=k.device, dtype=torch.float32)
        inv = inv_freq.to(device=k.device, dtype=torch.float32)
        freqs = pos[:, None] * inv[None, :]            # [T, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)        # [T, D]
        cos = emb.cos()[None, None, :, :]              # [1, 1, T, D]
        sin = emb.sin()[None, None, :, :]
        kf = k.to(torch.float32)
        k_post = kf * cos + self._rotate_half(kf) * sin
        return k_post.to(out_dtype)

    def _ensure_rotation(self, device, dtype) -> None:
        if self._rot is None:
            self._rot = R.make_rotation(
                self.rotation_kind, self.head_dim, seed=self.seed, device=device, dtype=dtype
            )

    def _compress(self, x, axis: str = "token", n_outliers: int = 0):
        """[B,H,T,D] BF16 → packed int4 dict (rotate → quant → pack).

        ``axis='token'`` (one scale per token) or ``'channel'`` (one scale per
        coordinate over the whole evicted block — the KIVI key scheme).
        ``n_outliers>0`` (token axis only, keys only) keeps the top-N per-token
        coordinates in fp16 (dense-and-sparse, KVQuant/QJL) and quantizes the rest.
        """
        import torch

        xr = self._rot.rotate(x.to(torch.float32))
        if axis == "channel":
            codes, scale, lo = P.quantize_int4_per_channel(xr)   # scale/lo [B,H,1,D]
            return {
                "packed": P.pack_int4(codes),
                "scale": scale.to(torch.bfloat16),
                "lo": lo.to(torch.bfloat16),
                "sizes": [x.shape[2]],
                "T": x.shape[2],
                "D": x.shape[3],
                "axis": "channel",
            }
        if n_outliers > 0:
            codes, scale, lo, out_idx, out_val = P.quantize_int4_per_token_outliers(
                xr, n_outliers)
            return {
                "packed": P.pack_int4(codes),
                "scale": scale.to(torch.bfloat16),
                "lo": lo.to(torch.bfloat16),
                "out_idx": out_idx,                  # [B,H,T,n_outliers] int16
                "out_val": out_val.to(torch.bfloat16),
                "outliers": int(n_outliers),
                "T": x.shape[2],
                "D": x.shape[3],
                "axis": "token",
            }
        codes, scale, lo = P.quantize_int4_per_token(xr)
        return {
            "packed": P.pack_int4(codes),
            "scale": scale.to(torch.bfloat16),
            "lo": lo.to(torch.bfloat16),
            "T": x.shape[2],
            "D": x.shape[3],
            "axis": "token",
        }

    def _decompress(self, store, out_dtype):
        """Packed int4 dict → [B,H,T,D] reconstruction (unpack → dequant → inv-rot)."""
        import torch

        codes = P.unpack_int4(store["packed"], store["D"]).to(torch.float32)
        if store.get("axis") == "channel":
            # expand each block's per-channel scale [B,H,n_blocks,D] over its tokens.
            sizes = torch.tensor(store["sizes"], device=codes.device)
            scale = store["scale"].to(torch.float32).repeat_interleave(sizes, dim=2)
            lo = store["lo"].to(torch.float32).repeat_interleave(sizes, dim=2)
            deq = codes * scale + lo
        elif store.get("outliers", 0) > 0:
            deq = P.dequantize_int4_per_token_outliers(
                codes, store["scale"].to(torch.float32), store["lo"].to(torch.float32),
                store["out_idx"], store["out_val"].to(torch.float32))
        else:
            deq = P.dequantize_int4_per_token(
                codes, store["scale"].to(torch.float32), store["lo"].to(torch.float32))
        rec = self._rot.inverse(deq)  # float32, orthogonal inverse
        return rec.to(out_dtype)

    @staticmethod
    def _append_store(dst, src):
        import torch

        if dst is None:
            return src
        if src.get("axis") == "channel":
            return {
                "packed": torch.cat([dst["packed"], src["packed"]], dim=2),
                "scale": torch.cat([dst["scale"], src["scale"]], dim=2),
                "lo": torch.cat([dst["lo"], src["lo"]], dim=2),
                "sizes": dst["sizes"] + src["sizes"],
                "T": dst["T"] + src["T"],
                "D": dst["D"],
                "axis": "channel",
            }
        if src.get("outliers", 0) > 0:
            return {
                "packed": torch.cat([dst["packed"], src["packed"]], dim=2),
                "scale": torch.cat([dst["scale"], src["scale"]], dim=2),
                "lo": torch.cat([dst["lo"], src["lo"]], dim=2),
                "out_idx": torch.cat([dst["out_idx"], src["out_idx"]], dim=2),
                "out_val": torch.cat([dst["out_val"], src["out_val"]], dim=2),
                "outliers": src["outliers"],
                "T": dst["T"] + src["T"],
                "D": dst["D"],
                "axis": "token",
            }
        return {
            "packed": torch.cat([dst["packed"], src["packed"]], dim=2),
            "scale": torch.cat([dst["scale"], src["scale"]], dim=2),
            "lo": torch.cat([dst["lo"], src["lo"]], dim=2),
            "T": dst["T"] + src["T"],
            "D": dst["D"],
            "axis": "token",
        }

    # ------------------------------------------------------------------ #
    # Cache API
    # ------------------------------------------------------------------ #
    def update(
        self,
        key_states,
        value_states,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Any]:
        import torch

        self._ensure_layer(layer_idx)
        self._ensure_rotation(key_states.device, torch.float32)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        # pre-RoPE bookkeeping: track per-token positions (for re-applying RoPE
        # on reconstruction) and the rotary base frequencies (captured once).
        if self.pre_rope:
            ck = cache_kwargs or {}
            inv = ck.get("rope_inv_freq")
            if inv is not None and self._rope_inv_freq is None:
                self._rope_inv_freq = inv.detach()
            existing = 0 if self._pos[layer_idx] is None else self._pos[layer_idx].shape[0]
            T_new = key_states.shape[2]
            cp = ck.get("cache_position")
            if cp is None:
                cp = torch.arange(existing, existing + T_new, device=key_states.device)
            cp = cp.reshape(-1).to(device=key_states.device, dtype=torch.int32)
            self._pos[layer_idx] = (
                cp if self._pos[layer_idx] is None
                else torch.cat([self._pos[layer_idx], cp], dim=0))

        # 1) append new tokens to the BF16 window.
        wK = key_states if self._wK[layer_idx] is None else torch.cat([self._wK[layer_idx], key_states], dim=2)
        wV = value_states if self._wV[layer_idx] is None else torch.cat([self._wV[layer_idx], value_states], dim=2)

        # 2) evict the oldest overflow tokens into the compressed store.
        n_overflow = wK.shape[2] - self.residual_length
        # per-channel keys evict in whole ``key_group_size`` blocks so each block's
        # per-channel scale sees real token statistics, not a single decode token.
        if self._is_bf16_layer(layer_idx):
            # BF16 bit-allocation layer: keep ALL tokens in the BF16 window, never
            # compress, so reconstruction is the exact BF16 history+window.
            n_evict = 0
        elif self.key_quant == "per_channel" and n_overflow > 0:
            n_evict = (n_overflow // self.key_group_size) * self.key_group_size
        else:
            n_evict = n_overflow
        if n_evict > 0:
            evK = wK[:, :, :n_evict, :]
            evV = wV[:, :, :n_evict, :]
            # route the first ``sink_length`` tokens of the stream to the BF16 sink.
            if self.sink_length > 0:
                cur = 0 if self._sK[layer_idx] is None else self._sK[layer_idx].shape[2]
                room = self.sink_length - cur
                if room > 0:
                    take = min(room, evK.shape[2])
                    sK, sV = evK[:, :, :take, :], evV[:, :, :take, :]
                    self._sK[layer_idx] = sK if self._sK[layer_idx] is None else torch.cat([self._sK[layer_idx], sK], dim=2)
                    self._sV[layer_idx] = sV if self._sV[layer_idx] is None else torch.cat([self._sV[layer_idx], sV], dim=2)
                    evK, evV = evK[:, :, take:, :], evV[:, :, take:, :]
            if evK.shape[2] > 0:
                self._cK[layer_idx] = self._append_store(
                    self._cK[layer_idx],
                    self._compress(evK, self._key_axis, n_outliers=self.key_outliers))
                self._cV[layer_idx] = self._append_store(
                    self._cV[layer_idx], self._compress(evV, "token"))
            wK = wK[:, :, n_evict:, :].contiguous()
            wV = wV[:, :, n_evict:, :].contiguous()
        self._wK[layer_idx] = wK
        self._wV[layer_idx] = wV

        # 3) return the full reconstructed K/V for attention: sink + history + window.
        out_dtype = key_states.dtype
        parts_K = []
        parts_V = []
        if self._sK[layer_idx] is not None:
            parts_K.append(self._sK[layer_idx])
            parts_V.append(self._sV[layer_idx])
        if self._cK[layer_idx] is not None:
            parts_K.append(self._decompress(self._cK[layer_idx], out_dtype))
            parts_V.append(self._decompress(self._cV[layer_idx], out_dtype))
        parts_K.append(wK)
        parts_V.append(wV)
        full_K = parts_K[0] if len(parts_K) == 1 else torch.cat(parts_K, dim=2)
        full_V = parts_V[0] if len(parts_V) == 1 else torch.cat(parts_V, dim=2)
        if self.pre_rope:
            # Re-apply RoPE to the reconstructed pre-RoPE keys. The stored
            # positions buffer is appended in arrival order, which is exactly the
            # reconstruction order (sink → history → window), so it aligns 1:1
            # with the concatenated keys. Values are NOT rotated.
            positions = self._pos[layer_idx]
            assert positions is not None and positions.shape[0] == full_K.shape[2], (
                f"pre_rope positions ({0 if positions is None else positions.shape[0]}) "
                f"must align with reconstructed key length ({full_K.shape[2]})")
            full_K = self._apply_rope(full_K, positions, out_dtype)
        return full_K, full_V

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._wK) or self._wK[layer_idx] is None:
            return 0
        comp = self._cK[layer_idx]["T"] if self._cK[layer_idx] is not None else 0
        sink = self._sK[layer_idx].shape[2] if self._sK[layer_idx] is not None else 0
        return sink + comp + self._wK[layer_idx].shape[2]

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx) -> None:  # pragma: no cover - beam search
        import torch

        for layer_idx in range(len(self._wK)):
            if self._wK[layer_idx] is not None:
                self._wK[layer_idx] = self._wK[layer_idx].index_select(0, beam_idx.to(self._wK[layer_idx].device))
                self._wV[layer_idx] = self._wV[layer_idx].index_select(0, beam_idx.to(self._wV[layer_idx].device))
            for store in (self._cK[layer_idx], self._cV[layer_idx]):
                if store is not None:
                    dev = store["packed"].device
                    store["packed"] = store["packed"].index_select(0, beam_idx.to(dev))
                    store["scale"] = store["scale"].index_select(0, beam_idx.to(dev))
                    store["lo"] = store["lo"].index_select(0, beam_idx.to(dev))

    @property
    def seen_tokens(self) -> int:  # transformers back-compat
        return self._seen_tokens

    def __len__(self) -> int:
        return len(self._wK)

    # ------------------------------------------------------------------ #
    # memory accounting
    # ------------------------------------------------------------------ #
    def memory_bytes(self) -> dict:
        """Resident storage bytes: packed codes + scale/zero + BF16 window.

        When ``pre_rope`` is on, the per-layer int32 positions buffer is also
        counted (``position_bytes``); it is a handful of bytes per token and does
        not change the ~4x memory win.

        When ``key_outliers`` is on (per-token keys), the sparse fp16 outlier
        side-channel is counted in ``outlier_bytes`` (and folded into the total):
        a real cost of ``n_outliers*(2 bytes idx + 2 bytes val)`` per stored key
        token, the price paid to rescue the dense int4 grid.
        """
        packed = scale_zero = window = position = outlier = 0
        for layer_idx in range(len(self._wK)):
            for store in (self._cK[layer_idx], self._cV[layer_idx]):
                if store is not None:
                    packed += store["packed"].numel() * store["packed"].element_size()
                    scale_zero += store["scale"].numel() * store["scale"].element_size()
                    scale_zero += store["lo"].numel() * store["lo"].element_size()
                    if store.get("outliers", 0) > 0:
                        outlier += store["out_idx"].numel() * store["out_idx"].element_size()
                        outlier += store["out_val"].numel() * store["out_val"].element_size()
            for w in (self._wK[layer_idx], self._wV[layer_idx],
                      self._sK[layer_idx], self._sV[layer_idx]):
                if w is not None:
                    window += w.numel() * w.element_size()
            p = self._pos[layer_idx] if layer_idx < len(self._pos) else None
            if p is not None:
                position += p.numel() * p.element_size()
        return {
            "packed_bytes": packed,
            "scale_zero_bytes": scale_zero,
            "window_bytes": window,
            "position_bytes": position,
            "outlier_bytes": outlier,
            "total_bytes": packed + scale_zero + window + position + outlier,
        }
