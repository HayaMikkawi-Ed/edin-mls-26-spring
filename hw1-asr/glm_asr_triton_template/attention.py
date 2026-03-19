"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

FUSION: softmax_inplace_kernel + attention_output_kernel
     → softmax_weighted_sum_fused_kernel (tiled over D)

Previous attempt loaded V as [BLOCK_K × BLOCK_D] all at once, causing
register spilling (e.g. 128×128 = 16 384 fp32 = 64 KB per program).

Fix: tile over the head_dim dimension in steps of TILE_D=16.
  - softmax weights w [BLOCK_K] stay in registers the whole time
  - each V tile is [BLOCK_K × TILE_D] = 128×16 = 2048 fp32 = 8 KB — fits
  - attention weights never written to HBM between the two ops
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
# Triton Kernels
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    )
    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    scores = tl.sum(k * q[None, :], axis=1) * scale
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q  * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=offs_k < seq_k,
    )


@triton.jit
def softmax_weighted_sum_fused_kernel(
    scores_ptr,
    v_ptr,
    output_ptr,
    seq_k,
    head_dim,
    stride_s0,
    stride_s1,
    stride_s2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,  # = seq_k_padded  (whole row of scores fits in registers)
    BLOCK_D: tl.constexpr,  # = head_dim_padded (loop-bound, fully unrolled by compiler)
    TILE_D:  tl.constexpr,  # tile width over D (16 keeps register pressure low)
):
    """
    FUSION: numerically-stable softmax + attention-weighted sum.
    Grid: (batch_heads, seq_q)

    Replaces two-kernel sequence:
        softmax_inplace_kernel   — read scores, wrote softmax weights to HBM
        attention_output_kernel  — read softmax weights from HBM, wrote output

    Register budget per CUDA thread (BLOCK_K=128, TILE_D=16, warp=32 threads):
        w        : BLOCK_K / 32 =  4 floats / thread  →   16 B
        v_tile   : (BLOCK_K / 32) × TILE_D = 4 × 16 = 64 floats  → 256 B
        Total    : ~80 registers — well within GPU limits
    """
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)

    # ── Step 1: load (masked) scores row, compute softmax weights ────────────
    # Padding positions were set to -1e9 in Python before this kernel runs.
    s = tl.load(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        mask=offs_k < seq_k,
        other=-float("inf"),
    )
    s     = s - tl.max(s, axis=0)          # numerical stability
    exp_s = tl.exp(s)
    w     = exp_s / tl.sum(exp_s, axis=0)  # [BLOCK_K] — stays in registers

    # ── Step 2: tiled weighted sum over V ────────────────────────────────────
    # Loop over head_dim in TILE_D chunks. Both bounds are constexprs so the
    # compiler fully unrolls this loop — no runtime branch overhead.
    for d in range(0, BLOCK_D, TILE_D):
        offs_d = d + tl.arange(0, TILE_D)

        v_tile = tl.load(
            v_ptr
            + pid_bh          * stride_v0
            + offs_k[:, None] * stride_v1
            + offs_d[None, :] * stride_v2,
            mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
            other=0.0,
        )   # [BLOCK_K, TILE_D]

        out_tile = tl.sum(w[:, None] * v_tile, axis=0)   # [TILE_D]

        tl.store(
            output_ptr
            + pid_bh        * stride_o0
            + pid_q         * stride_o1
            + offs_d        * stride_o2,
            out_tile,
            mask=offs_d < head_dim,
        )


@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask   = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q  * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q  * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim    = head_dim or (hidden_size // num_heads)
        self.scale       = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256
TILE_D            = 16   # D-tile size inside the fused kernel


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention using Triton kernels.

    FUSION: softmax + weighted sum run in one kernel (tiled over D).
    Attention weights never touch HBM between the two operations.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    seq_k_padded    = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    use_triton = (
        q.is_cuda
        and seq_k_padded    <= MAX_ATTENTION_DIM
        and head_dim_padded <= MAX_ATTENTION_DIM
        and head_dim_padded >= TILE_D            # must have at least one tile
    )

    if use_triton:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32)
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)

        if seq_k_padded != seq_k or head_dim_padded != head_dim:
            k_padded = torch.zeros(
                (batch * num_heads, seq_k_padded, head_dim_padded),
                dtype=torch.float32, device=q.device,
            )
            v_padded = torch.zeros_like(k_padded)
            q_padded = torch.zeros(
                (batch * num_heads, seq_q, head_dim_padded),
                dtype=torch.float32, device=q.device,
            )
            k_padded[:, :seq_k, :head_dim] = k_flat
            v_padded[:, :seq_k, :head_dim] = v_flat
            q_padded[:, :, :head_dim]      = q_flat
            k_flat = k_padded
            v_flat = v_padded
            q_flat = q_padded

        scores = torch.empty(
            (batch * num_heads, seq_q, seq_k_padded),
            dtype=torch.float32, device=q.device,
        )
        output = torch.empty(
            (batch * num_heads, seq_q, head_dim_padded),
            dtype=torch.float32, device=q.device,
        )

        grid = (batch * num_heads, seq_q)

        # Kernel 1: Q @ K^T
        attention_scores_kernel[grid](
            q_flat, k_flat, scores,
            float(scale), seq_k_padded, head_dim_padded,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            scores.stride(0), scores.stride(1), scores.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        # Apply masks in Python before fused kernel
        if seq_k_padded != seq_k:
            scores[:, :, seq_k:] = -1e9

        if is_causal:
            causal_mask = (
                torch.triu(
                    torch.ones((seq_q, seq_k_padded), dtype=torch.float32, device=q.device),
                    diagonal=1,
                ) * -1e9
            )
            scores = scores + causal_mask[None, :, :]

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                attention_mask = attention_mask.reshape(batch * num_heads, seq_q, seq_k)
            if seq_k_padded != seq_k:
                mask_padded = torch.zeros(
                    (batch * num_heads, seq_q, seq_k_padded),
                    dtype=torch.float32, device=q.device,
                )
                mask_padded[:, :, :seq_k] = attention_mask
                mask_padded[:, :, seq_k:] = -1e9
                attention_mask = mask_padded
            scores = scores + attention_mask

        # Kernel 2 (FUSED, tiled): softmax + weighted sum
        #
        # Previously:
        #   softmax_inplace_kernel[N,](scores_2d, ...)  → wrote attn weights to HBM
        #   attention_output_kernel[grid](scores, v_flat, output, ...)  → read back
        #
        # Now: one kernel, attn weights stay in registers, V loaded in TILE_D chunks.
        softmax_weighted_sum_fused_kernel[grid](
            scores, v_flat, output,
            seq_k_padded, head_dim_padded,
            scores.stride(0), scores.stride(1), scores.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
            TILE_D=TILE_D,
        )

        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    # ── CPU / large-seq fallback (unchanged) ─────────────────────────────────
    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = (
            torch.triu(
                torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
                diagonal=1,
            ) * -1e9
        )
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores       = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output       = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)

    return output.to(q.dtype)


if __name__ == "__main__":
    import math

    print("Testing Triton Attention...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def ref_attn(q, k, v, is_causal=False):
        scale  = 1.0 / math.sqrt(q.shape[-1])
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
        status = "PASS ✓" if err < 1e-3 else f"FAIL  err={err:.2e}"
        print(f"  {tag}  max_err={err:.2e}  {status}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 4
    k_gqa = torch.randn(B, num_kv_heads, Sk, D, device=device)
    v_gqa = torch.randn(B, num_kv_heads, Sk, D, device=device)
    attn  = MultiHeadAttention(hidden_size=H*D, num_heads=H, num_kv_heads=num_kv_heads)
    out_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {out_gqa.shape}")

    print("\nTriton Attention working!")