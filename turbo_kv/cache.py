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
    ) -> None:
        super().__init__()
        if bits != 4:
            raise ValueError("TurboKVCache MVP supports int4 only (bits=4)")
        self.residual_length = int(residual_length)
        self.rotation_kind = rotation
        self.head_dim = int(head_dim)
        self.bits = int(bits)
        self.seed = int(seed)
        self.sink_length = int(sink_length)

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
        self._seen_tokens = 0  # total tokens seen by layer 0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self._wK) <= layer_idx:
            self._cK.append(None)
            self._cV.append(None)
            self._wK.append(None)
            self._wV.append(None)
            self._sK.append(None)
            self._sV.append(None)

    def _ensure_rotation(self, device, dtype) -> None:
        if self._rot is None:
            self._rot = R.make_rotation(
                self.rotation_kind, self.head_dim, seed=self.seed, device=device, dtype=dtype
            )

    def _compress(self, x):
        """[B,H,T,D] BF16 → packed int4 dict (rotate → per-token quant → pack)."""
        import torch

        xr = self._rot.rotate(x.to(torch.float32))
        codes, scale, lo = P.quantize_int4_per_token(xr)
        packed = P.pack_int4(codes)
        return {
            "packed": packed,
            "scale": scale.to(torch.bfloat16),
            "lo": lo.to(torch.bfloat16),
            "T": x.shape[2],
            "D": x.shape[3],
        }

    def _decompress(self, store, out_dtype):
        """Packed int4 dict → [B,H,T,D] reconstruction (unpack → dequant → inv-rot)."""
        import torch

        codes = P.unpack_int4(store["packed"], store["D"])
        deq = P.dequantize_int4_per_token(
            codes.to(torch.float32),
            store["scale"].to(torch.float32),
            store["lo"].to(torch.float32),
        )
        rec = self._rot.inverse(deq)  # float32, orthogonal inverse
        return rec.to(out_dtype)

    @staticmethod
    def _append_store(dst, src):
        import torch

        if dst is None:
            return src
        return {
            "packed": torch.cat([dst["packed"], src["packed"]], dim=2),
            "scale": torch.cat([dst["scale"], src["scale"]], dim=2),
            "lo": torch.cat([dst["lo"], src["lo"]], dim=2),
            "T": dst["T"] + src["T"],
            "D": dst["D"],
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

        # 1) append new tokens to the BF16 window.
        wK = key_states if self._wK[layer_idx] is None else torch.cat([self._wK[layer_idx], key_states], dim=2)
        wV = value_states if self._wV[layer_idx] is None else torch.cat([self._wV[layer_idx], value_states], dim=2)

        # 2) evict the oldest overflow tokens into the compressed store.
        n_evict = wK.shape[2] - self.residual_length
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
                self._cK[layer_idx] = self._append_store(self._cK[layer_idx], self._compress(evK))
                self._cV[layer_idx] = self._append_store(self._cV[layer_idx], self._compress(evV))
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
        if len(parts_K) == 1:
            return parts_K[0], parts_V[0]
        return torch.cat(parts_K, dim=2), torch.cat(parts_V, dim=2)

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
        """Resident storage bytes: packed codes + scale/zero + BF16 window."""
        packed = scale_zero = window = 0
        for layer_idx in range(len(self._wK)):
            for store in (self._cK[layer_idx], self._cV[layer_idx]):
                if store is not None:
                    packed += store["packed"].numel() * store["packed"].element_size()
                    scale_zero += store["scale"].numel() * store["scale"].element_size()
                    scale_zero += store["lo"].numel() * store["lo"].element_size()
            for w in (self._wK[layer_idx], self._wV[layer_idx],
                      self._sK[layer_idx], self._sV[layer_idx]):
                if w is not None:
                    window += w.numel() * w.element_size()
        return {
            "packed_bytes": packed,
            "scale_zero_bytes": scale_zero,
            "window_bytes": window,
            "total_bytes": packed + scale_zero + window,
        }
