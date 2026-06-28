"""M4 — patch ``Qwen2Attention`` for rotate-query TurboQuant int4 KV.

The TurboQuant identity ``qᵀk = (Rq)ᵀ(Rk)`` lets us keep the cached keys in the
*rotated* basis and never inverse-rotate them: instead we rotate the **query**.
For values, ``o = Σ pᵢ vᵢ`` and storing ``ṽᵢ = R vᵢ`` gives ``Σ pᵢ ṽᵢ = R o``, so a
**single** inverse rotation of the attention *output* (per head, not per token)
recovers ``o``. This is the structure the fused int4 kernels (M5) will exploit.

Concretely the patched ``forward`` (SDPA path) does, after RoPE:

    q, k, v  ->  R q, R k, R v        (orthogonal rotation on head_dim)
    k, v      = cache.update(R k, R v) (int4-quantized, rotated, NOT inverse-rotated)
    o_rot     = softmax(q_rot k_rotᵀ / √d) · v_rot      (SDPA, rotated space)
    o         = Rᵀ o_rot                                 (one inverse rotation)

The cache is a :class:`turbo_kv.cache.TurboKVCache` with ``rotation="none"`` — the
rotation lives here in the patch, so the cache only does int4 pack/window. Because
``(Rq)·dequant(quant(Rk))`` is exactly what M3 measured, M4 quality matches M3 for
the same rotation (use **RHT**; dense is inner-product-broken, see M3).

``torch`` / ``transformers`` are imported lazily so this module imports on CPU.
"""
from __future__ import annotations

import types
from typing import Optional, Tuple

from turbo_kv import rotations as R


def _turbo_sdpa_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    position_embeddings: Optional[Tuple] = None,
    **kwargs,
):
    """Drop-in replacement for ``Qwen2SdpaAttention.forward`` with rotate-query int4 KV."""
    import torch
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    bsz, q_len, _ = hidden_states.size()

    q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(v, position_ids)
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # --- TurboQuant rotation (rotate-query identity) on head_dim ---
    rot = self._turbo_rot
    in_dtype = q.dtype
    q = rot.rotate(q.float()).to(in_dtype)   # query rotated; logits = (Rq)·(Rk)
    k = rot.rotate(k.float()).to(in_dtype)   # keys rotated, stored int4, never inverse-rotated
    v = rot.rotate(v.float()).to(in_dtype)   # values rotated, inverse-rotated once at the output

    # GQA shape assertions: [B, H_kv, T, D].
    assert k.shape[1] == self.num_key_value_heads and k.shape[-1] == self.head_dim
    assert v.shape[1] == self.num_key_value_heads and v.shape[-1] == self.head_dim

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : k.shape[-2]]

    if q.device.type == "cuda" and causal_mask is not None:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    is_causal = causal_mask is None and q_len > 1
    attn_out_rot = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=causal_mask, dropout_p=0.0, is_causal=is_causal
    )

    # --- single inverse rotation of the attention output (per head, not per token) ---
    attn_out = rot.inverse(attn_out_rot.float()).to(in_dtype)

    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
    attn_out = self.o_proj(attn_out)
    return attn_out, None, past_key_value


def patch_qwen2_attention(model, *, rotation: str = "rht", seed: int = 0):
    """Patch every ``self_attn`` of a Qwen2 model with the rotate-query int4 forward.

    A single orthogonal rotation ``R`` (shared across layers and heads) is built on
    ``head_dim`` and attached to each attention module. Returns ``model`` (mutated).
    Call :func:`unpatch_qwen2_attention` to restore.
    """
    import torch

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    dev = next(model.parameters()).device
    rot = R.make_rotation(rotation, head_dim, seed=seed, device=dev, dtype=torch.float32)

    for layer in model.model.layers:
        attn = layer.self_attn
        if not hasattr(attn, "_turbo_orig_forward"):
            attn._turbo_orig_forward = attn.forward
        attn._turbo_rot = rot
        attn.forward = types.MethodType(_turbo_sdpa_forward, attn)

    model._turbo_patched = True
    model._turbo_rotation = rotation
    return model


def unpatch_qwen2_attention(model):
    """Restore the original ``self_attn.forward`` on every layer."""
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_turbo_orig_forward"):
            attn.forward = attn._turbo_orig_forward
            del attn._turbo_orig_forward
        if hasattr(attn, "_turbo_rot"):
            del attn._turbo_rot
    model._turbo_patched = False
    return model
