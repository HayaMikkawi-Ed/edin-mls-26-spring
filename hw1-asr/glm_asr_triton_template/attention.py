"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['seq_q', 'seq_k', 'head_dim'],
)
@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    scale,
    seq_q, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    is_causal: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    q_offset = pid_q * BLOCK_Q
    d_offsets = tl.arange(0, BLOCK_D)
    q_offsets = q_offset + tl.arange(0, BLOCK_Q)
    d_mask = d_offsets < head_dim
    q_mask = q_offsets < seq_q

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + q_offsets[:, None] * stride_q1 + d_offsets[None, :],
        mask=q_mask[:, None] & d_mask[None, :], other=0.0
    ).to(tl.bfloat16)

    m = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for k_start in range(0, seq_k, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < seq_k

        k = tl.load(
            k_ptr + pid_bh * stride_k0 + k_offsets[:, None] * stride_k1 + d_offsets[None, :],
            mask=k_mask[:, None] & d_mask[None, :], other=0.0
        ).to(tl.bfloat16)

        scores = tl.dot(q, tl.trans(k)) * scale

        if is_causal:
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            scores = tl.where(causal_mask, scores, float('-inf'))

        scores = tl.where(k_mask[None, :], scores, float('-inf'))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.math.exp(m - m_new)
        scores_exp = tl.math.exp(scores - m_new[:, None])

        v = tl.load(
            v_ptr + pid_bh * stride_v0 + k_offsets[:, None] * stride_v1 + d_offsets[None, :],
            mask=k_mask[:, None] & d_mask[None, :], other=0.0
        ).to(tl.bfloat16)

        l = alpha * l + tl.sum(scores_exp, axis=1)
        acc = alpha[:, None] * acc + tl.dot(scores_exp.to(tl.bfloat16), v)
        m = m_new

    acc = acc / l[:, None]

    tl.store(
        output_ptr + pid_bh * stride_o0 + q_offsets[:, None] * stride_o1 + d_offsets[None, :],
        acc,
        mask=q_mask[:, None] & d_mask[None, :]
    )


@triton.jit
def attention_scores_kernel(
    q_ptr, k_ptr, scores_ptr,
    scale, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim, other=0.0,
    ).to(tl.bfloat16)
    k = tl.load(
        k_ptr + pid_bh * stride_k0 + offs_k[:, None] * stride_k1 + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0,
    ).to(tl.bfloat16)
    scores = (tl.sum(k * q[None, :], axis=1) * scale).to(tl.float32)
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        scores, mask=offs_k < seq_k,
    )


@triton.jit
def softmax_weighted_sum_fused_kernel(
    scores_ptr, v_ptr, output_ptr,
    seq_k, head_dim,
    stride_s0, stride_s1, stride_s2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILE_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)

    s = tl.load(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        mask=offs_k < seq_k, other=-float("inf"),
    )
    s = s - tl.max(s, axis=0)
    exp_s = tl.exp(s)
    w = exp_s / tl.sum(exp_s, axis=0)

    for d in range(0, BLOCK_D, TILE_D):
        offs_d = d + tl.arange(0, TILE_D)
        v_tile = tl.load(
            v_ptr + pid_bh * stride_v0 + offs_k[:, None] * stride_v1 + offs_d[None, :] * stride_v2,
            mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0,
        )
        out_tile = tl.sum(w[:, None] * v_tile, axis=0)
        tl.store(
            output_ptr + pid_bh * stride_o0 + pid_q * stride_o1 + offs_d * stride_o2,
            out_tile, mask=offs_d < head_dim,
        )


@triton.jit
def causal_mask_kernel(
    scores_ptr, seq_k, offset,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        mask=mask, other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        scores, mask=mask,
    )


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    def __init__(self, hidden_size, num_heads, num_kv_heads=None, head_dim=None):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(self, q, k, v, attention_mask=None, is_causal=False):
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape
        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)
        return scaled_dot_product_attention(q, k, v, attention_mask, is_causal, self.scale)

    def _expand_kv(self, x, num_repeats):
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(batch, num_kv_heads, num_repeats, seq_len, head_dim)
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x):
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 2048
TILE_D = 16


def scaled_dot_product_attention(q, k, v, attention_mask=None, is_causal=False, scale=None):
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    seq_k_padded = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    # Primary path: FlashAttention (for seq_q >= 16)
    if q.is_cuda and seq_q >= 16 and seq_k >= 16:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.bfloat16).contiguous()
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.bfloat16).contiguous()
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.bfloat16).contiguous()

        output = torch.zeros(
            (batch * num_heads, seq_q, head_dim),
            dtype=torch.float32, device=q.device
        )

        BLOCK_D = next_power_of_two(head_dim)
        grid = lambda meta: (batch * num_heads, triton.cdiv(seq_q, meta['BLOCK_Q']))
        flash_attention_kernel[grid](
            q_flat, k_flat, v_flat, output,
            float(scale),
            seq_q, seq_k, head_dim,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            is_causal=is_causal,
            BLOCK_D=BLOCK_D,
        )
        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    # Secondary path: fused softmax+weighted_sum kernel (for seq_q < 16, e.g. decoding)
    if (q.is_cuda
            and seq_k_padded <= MAX_ATTENTION_DIM
            and head_dim_padded <= MAX_ATTENTION_DIM
            and head_dim_padded >= TILE_D):

        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32)
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)

        if seq_k_padded != seq_k or head_dim_padded != head_dim:
            k_padded = torch.zeros((batch * num_heads, seq_k_padded, head_dim_padded),
                                   dtype=torch.float32, device=q.device)
            v_padded = torch.zeros_like(k_padded)
            q_padded = torch.zeros((batch * num_heads, seq_q, head_dim_padded),
                                   dtype=torch.float32, device=q.device)
            k_padded[:, :seq_k, :head_dim] = k_flat
            v_padded[:, :seq_k, :head_dim] = v_flat
            q_padded[:, :, :head_dim] = q_flat
            k_flat, v_flat, q_flat = k_padded, v_padded, q_padded

        scores = torch.empty((batch * num_heads, seq_q, seq_k_padded),
                             dtype=torch.float32, device=q.device)
        output = torch.empty((batch * num_heads, seq_q, head_dim_padded),
                             dtype=torch.float32, device=q.device)

        grid = (batch * num_heads, seq_q)

        attention_scores_kernel[grid](
            q_flat, k_flat, scores,
            float(scale), seq_k_padded, head_dim_padded,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            scores.stride(0), scores.stride(1), scores.stride(2),
            BLOCK_K=seq_k_padded, BLOCK_D=head_dim_padded,
        )

        if seq_k_padded != seq_k:
            scores[:, :, seq_k:] = -1e9

        if is_causal:
            causal_mask = torch.triu(
                torch.ones((seq_q, seq_k_padded), dtype=torch.float32, device=q.device),
                diagonal=1,
            ) * -1e9
            scores = scores + causal_mask[None, :, :]

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                attention_mask = attention_mask.reshape(batch * num_heads, seq_q, seq_k)
            if seq_k_padded != seq_k:
                mask_padded = torch.zeros((batch * num_heads, seq_q, seq_k_padded),
                                         dtype=torch.float32, device=q.device)
                mask_padded[:, :, :seq_k] = attention_mask
                mask_padded[:, :, seq_k:] = -1e9
                attention_mask = mask_padded
            scores = scores + attention_mask

        softmax_weighted_sum_fused_kernel[grid](
            scores, v_flat, output,
            seq_k_padded, head_dim_padded,
            scores.stride(0), scores.stride(1), scores.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=seq_k_padded, BLOCK_D=head_dim_padded, TILE_D=TILE_D,
        )

        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    # PyTorch fallback
    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device), diagonal=1,
        ) * -1e9
        scores = scores + mask[None, None, :, :]
    if attention_mask is not None:
        scores = scores + attention_mask
    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)
    return output.to(q.dtype)


if __name__ == "__main__":
    import math
    print("Testing Triton Attention...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def ref_attn(q, k, v, is_causal=False):
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
        if is_causal:
            sq, sk = q.shape[-2], k.shape[-2]
            m = torch.triu(torch.ones(sq, sk, device=q.device), diagonal=1) * -1e9
            scores = scores + m[None, None]
        scores = scores - scores.amax(-1, keepdim=True)
        w = torch.exp(scores)
        w = w / w.sum(-1, keepdim=True)
        return torch.einsum("bnqk,bnkd->bnqd", w, v)

    B, H, Sq, Sk, D = 1, 28, 100, 100, 128
    torch.manual_seed(0)
    q = torch.randn(B, H, Sq, D, device=device)
    k = torch.randn(B, H, Sk, D, device=device)
    v = torch.randn(B, H, Sk, D, device=device)

    for causal in (False, True):
        ref = ref_attn(q, k, v, is_causal=causal)
        out = scaled_dot_product_attention(q, k, v, is_causal=causal)
        err = (ref - out).abs().max().item()
        tag = "causal    " if causal else "non-causal"
        status = "PASS" if err < 1e-3 else f"FAIL err={err:.2e}"
        print(f"  {tag}  max_err={err:.2e}  {status}")

    print("\nTriton Attention working!")