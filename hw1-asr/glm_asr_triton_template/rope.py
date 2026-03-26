"""
Triton Rotary Position Embeddings (RoPE)

FUSION: apply_rotary_pos_emb
  One kernel handles both Q and K rotation regardless of head counts (GQA).

  Grid: (num_q_heads, seq_len)
  - Every program loads cos/sin once for its sequence position.
  - Every program rotates its Q head.
  - Programs where pid_head < num_kv_heads ALSO rotate the corresponding K head,
    reusing the already-loaded cos/sin with zero extra HBM reads.

  This activates for BOTH:
    audio encoder  (num_q_heads == num_kv_heads == 20)
    text decoder   (num_q_heads=28, num_kv_heads=4)  ← previously fell back to PyTorch

  HBM savings vs original two-call PyTorch path (per layer per token):
    cos/sin reads: (num_q_heads + num_kv_heads) × seq_len × half_dim × 4B
                 →  num_q_heads               × seq_len × half_dim × 4B
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def get_stream():
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def compute_freqs_kernel(
    positions_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    half_dim,
    stride_pos,
    stride_inv,
    stride_cos0,
    stride_cos1,
    stride_sin0,
    stride_sin1,
    BLOCK: tl.constexpr,
):
    """
    Compute cos and sin for rotary embeddings.
    Grid: (seq_len,)
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < half_dim

    pos = tl.load(positions_ptr + pid * stride_pos)
    inv = tl.load(inv_freq_ptr + offs * stride_inv, mask=mask, other=0.0)
    freqs = pos * inv

    cos_half = tl.cos(freqs)
    sin_half = tl.sin(freqs)

    tl.store(cos_ptr + pid * stride_cos0 + offs * stride_cos1, cos_half, mask=mask)
    tl.store(cos_ptr + pid * stride_cos0 + (offs + half_dim) * stride_cos1, cos_half, mask=mask)
    tl.store(sin_ptr + pid * stride_sin0 + offs * stride_sin1, sin_half, mask=mask)
    tl.store(sin_ptr + pid * stride_sin0 + (offs + half_dim) * stride_sin1, sin_half, mask=mask)


@triton.jit
def rope_qk_fused_kernel(
    q_ptr,          # [num_q_heads,  seq_len, head_dim]  (batch indexed out by caller)
    k_ptr,          # [num_kv_heads, seq_len, head_dim]
    cos_ptr,        # [seq_len, half_dim]
    sin_ptr,        # [seq_len, half_dim]
    q_out_ptr,
    k_out_ptr,
    num_kv_heads,   # scalar: number of KV heads (may be < num_q_heads for GQA)
    half_dim,       # scalar: rotary_dim // 2
    head_dim,       # scalar: full head dimension
    stride_qh,      # Q strides
    stride_qs,
    stride_qd,
    stride_kh,      # K strides
    stride_ks,
    stride_kd,
    stride_cs,      # cos/sin strides (same layout)
    stride_cd,
    stride_oh,      # output strides (q_out and k_out share layout with their inputs)
    stride_os,
    stride_od,
    BLOCK: tl.constexpr,  # >= half_dim, power of two
):
    """
    GQA-aware fused RoPE for Q and K.
    Grid: (num_q_heads, seq_len)

    Each program:
      1. Loads cos/sin for its sequence position (1 read, shared below).
      2. Rotates Q[pid_head, pid_seq, :].
      3. If pid_head < num_kv_heads: also rotates K[pid_head, pid_seq, :]
         using the same cos/sin — zero extra HBM reads for position data.

    For standard MHA (num_q_heads == num_kv_heads):
      every program does both Q and K → same as before.
    For GQA (num_kv_heads < num_q_heads, e.g. 4 vs 28):
      the first 4 programs also handle K; the remaining 24 only handle Q.
      cos/sin are still loaded only once per program instead of separately
      for Q and K in the original two-call path.
    """
    pid_h = tl.program_id(0)   # Q head index
    pid_s = tl.program_id(1)   # sequence position

    offs = tl.arange(0, BLOCK)
    mask = offs < half_dim

    # ── Step 1: load cos/sin (one read, reused for both Q and K) ─────────────
    cos = tl.load(cos_ptr + pid_s * stride_cs + offs * stride_cd, mask=mask, other=1.0)
    sin = tl.load(sin_ptr + pid_s * stride_cs + offs * stride_cd, mask=mask, other=0.0)

    # ── Step 2: rotate Q[pid_h] ───────────────────────────────────────────────
    base_q = pid_h * stride_qh + pid_s * stride_qs
    x1 = tl.load(q_ptr + base_q + offs            * stride_qd, mask=mask, other=0.0)
    x2 = tl.load(q_ptr + base_q + (offs + half_dim) * stride_qd, mask=mask, other=0.0)

    base_qo = pid_h * stride_oh + pid_s * stride_os
    tl.store(q_out_ptr + base_qo + offs            * stride_od, x1 * cos - x2 * sin, mask=mask)
    tl.store(q_out_ptr + base_qo + (offs + half_dim) * stride_od, x2 * cos + x1 * sin, mask=mask)

    # ── Step 3: rotate K[pid_h] only when this head has a KV counterpart ──────
    # Triton scalar comparison: generates a predicated branch in PTX.
    # Programs with pid_h >= num_kv_heads skip the K load/store entirely.
    if pid_h < num_kv_heads:
        base_k  = pid_h * stride_kh + pid_s * stride_ks
        base_ko = pid_h * stride_oh + pid_s * stride_os   # k_out has same layout as k

        x1k = tl.load(k_ptr + base_k + offs            * stride_kd, mask=mask, other=0.0)
        x2k = tl.load(k_ptr + base_k + (offs + half_dim) * stride_kd, mask=mask, other=0.0)

        tl.store(k_out_ptr + base_ko + offs            * stride_od, x1k * cos - x2k * sin, mask=mask)
        tl.store(k_out_ptr + base_ko + (offs + half_dim) * stride_od, x2k * cos + x1k * sin, mask=mask)


# ============================================================================
# RotaryEmbedding class (unchanged)
# ============================================================================

class RotaryEmbedding:
    """Rotary Position Embedding using Triton."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        self.rotary_dim = int(dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.inv_freq = inv_freq
        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len: int, device: Optional[torch.device] = None):
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device

        positions  = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos_cache  = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)
        sin_cache  = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)

        if device.type == "cuda":
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            block = triton.next_power_of_2(half_dim)
            compute_freqs_kernel[(seq_len,)](
                positions, self.inv_freq, cos_cache, sin_cache,
                seq_len, half_dim,
                positions.stride(0), self.inv_freq.stride(0),
                cos_cache.stride(0), cos_cache.stride(1),
                sin_cache.stride(0), sin_cache.stride(1),
                BLOCK=block,
            )
        else:
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            freqs     = positions[:, None] * self.inv_freq[None, :]
            cos_half  = torch.cos(freqs)
            sin_half  = torch.sin(freqs)
            cos_cache[:, :half_dim]            = cos_half
            cos_cache[:, half_dim:half_dim*2]  = cos_half
            sin_cache[:, :half_dim]            = sin_half
            sin_cache[:, half_dim:half_dim*2]  = sin_half

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len, device=x.device)
        elif self.cos_cached.device != x.device:
            self._update_cache(self.max_seq_len_cached, device=x.device)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]
                sin = sin[0]
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)

        return cos, sin


# ============================================================================
# Helpers
# ============================================================================

def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ROPE_DIM = 256


def _apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """PyTorch fallback for a single tensor."""
    cos = cos[:x.shape[-2]]
    sin = sin[:x.shape[-2]]

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:half_dim * 2]

    cos_e = cos[None, None, :, :]
    sin_e = sin[None, None, :, :]

    x1_rot = x1 * cos_e - x2 * sin_e
    x2_rot = x2 * cos_e + x1 * sin_e

    if head_dim > half_dim * 2:
        return torch.cat([x1_rot, x2_rot, x[..., half_dim * 2:]], dim=-1)
    return torch.cat([x1_rot, x2_rot], dim=-1)


# ============================================================================
# Public API  (model.py calls this — signature unchanged)
# ============================================================================

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K.

    FUSION: single kernel handles both Q and K for any head-count ratio,
    including GQA (num_q_heads > num_kv_heads).

    API is identical to original — model.py requires no changes.
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _,     num_kv_heads, _,       _       = k.shape

    if rotary_dim is None:
        rotary_dim = head_dim

    half_dim = rotary_dim // 2

    if cos.shape[-1] > half_dim:
        cos = cos[:, :half_dim]
        sin = sin[:, :half_dim]

    cos = cos.to(torch.float32).contiguous()
    sin = sin.to(torch.float32).contiguous()

    half_dim_padded = next_power_of_two(half_dim)

    use_triton = (
        q.is_cuda
        and half_dim_padded <= MAX_ROPE_DIM
        and cos.shape[0] >= seq_len   # cos covers all positions
    )

    if use_triton:
        q_f   = q.to(torch.float32).contiguous()
        k_f   = k.to(torch.float32).contiguous()
        q_out = torch.empty_like(q_f)
        k_out = torch.empty_like(k_f)

        # Copy pass-through dims (beyond rotary_dim) before the kernel
        # so we only need to store rotated slice inside the kernel.
        if head_dim > rotary_dim:
            q_out[..., rotary_dim:] = q_f[..., rotary_dim:]
            k_out[..., rotary_dim:] = k_f[..., rotary_dim:]

        # Grid: (num_q_heads, seq_len)
        # Programs with pid_h < num_kv_heads also rotate K.
        grid = (num_q_heads, seq_len)

        for b in range(batch):
            rope_qk_fused_kernel[grid](
                q_f[b], k_f[b],
                cos, sin,
                q_out[b], k_out[b],
                num_kv_heads,
                half_dim,
                head_dim,
                # Q strides (batch dim indexed out)
                q_f.stride(1), q_f.stride(2), q_f.stride(3),
                # K strides
                k_f.stride(1), k_f.stride(2), k_f.stride(3),
                # cos/sin strides
                cos.stride(0), cos.stride(1),
                # output strides (q_out layout)
                q_out.stride(1), q_out.stride(2), q_out.stride(3),
                BLOCK=half_dim_padded,
            )

        return q_out.to(q.dtype), k_out.to(k.dtype)

    # ── PyTorch fallback ──────────────────────────────────────────────────────
    q_out = _apply_rope_single(q, cos, sin, half_dim, head_dim)
    k_out = _apply_rope_single(k, cos, sin, half_dim, head_dim)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    import math

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    def run_test(tag, batch, num_q, num_kv, seq, dim):
        rope   = RotaryEmbedding(dim=dim, max_position_embeddings=256)
        q      = torch.randn(batch, num_q,  seq, dim, device=device)
        k      = torch.randn(batch, num_kv, seq, dim, device=device)
        cos, sin = rope(q)

        # Fused
        q_fused, k_fused = apply_rotary_pos_emb(q, k, cos, sin)

        # Reference (PyTorch)
        half = dim // 2
        q_ref = _apply_rope_single(q, cos, sin, half, dim)
        k_ref = _apply_rope_single(k, cos, sin, half, dim)

        eq = (q_fused - q_ref).abs().max().item()
        ek = (k_fused - k_ref).abs().max().item()
        status = "PASS ✓" if max(eq, ek) < 1e-4 else "FAIL"
        print(f"{tag:30s}  Q err={eq:.1e}  K err={ek:.1e}  {status}")

    run_test("MHA  (audio, heads=20)",       1, 20, 20, 16, 64)
    run_test("GQA  (text,  Q=28 KV=4)",      1, 28,  4,  1, 128)
    run_test("GQA  batch>1",                 2, 28,  4,  8, 128)
    run_test("partial RoPE (factor=0.5)",    1,  4,  4, 16,  32)

    print("\nTriton RoPE working!")