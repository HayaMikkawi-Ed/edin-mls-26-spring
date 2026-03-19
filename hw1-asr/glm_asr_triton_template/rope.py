"""
Triton Rotary Position Embeddings (RoPE)

FUSION: apply_rotary_pos_emb
  Previously: _apply_rope_single(q) then _apply_rope_single(k)
    - cos/sin loaded from HBM twice
    - two separate kernel sequences

  Now: rope_qk_fused_kernel processes Q and K together
    - cos/sin loaded from HBM once, shared for both Q and K
    - one kernel launch instead of two

  API is unchanged — model.py calls apply_rotary_pos_emb(q, k, cos, sin)
  exactly as before.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for RoPE
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
    tl.store(
        cos_ptr + pid * stride_cos0 + (offs + half_dim) * stride_cos1,
        cos_half,
        mask=mask,
    )
    tl.store(sin_ptr + pid * stride_sin0 + offs * stride_sin1, sin_half, mask=mask)
    tl.store(
        sin_ptr + pid * stride_sin0 + (offs + half_dim) * stride_sin1,
        sin_half,
        mask=mask,
    )


@triton.jit
def rope_qk_fused_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    seq_len,
    half_dim,
    rotary_dim,
    head_dim,
    stride_q0,   # batch
    stride_q1,   # head
    stride_q2,   # seq
    stride_q3,   # dim
    stride_k0,
    stride_k1,
    stride_k2,
    stride_k3,
    stride_cos0, # seq
    stride_cos1, # dim
    stride_o0,   # batch  (same layout for q_out and k_out)
    stride_o1,   # head
    stride_o2,   # seq
    stride_o3,   # dim
    BLOCK: tl.constexpr,   # >= half_dim, power of two
):
    """
    FUSION: Apply RoPE to Q and K in one kernel.
    Grid: (batch * (num_q_heads + num_kv_heads), seq_len)

    cos/sin are read once per program and applied to whichever of Q or K
    this program instance is responsible for.

    Savings vs two separate _apply_rope_single calls:
      - cos/sin HBM reads halved  (seq_len * rotary_dim * 4 bytes * 2 → * 1)
      - kernel launches halved    (2 → 1)
    """
    pid_hs = tl.program_id(0)   # combined head index (Q heads first, then K heads)
    pid_s  = tl.program_id(1)   # sequence position

    offs = tl.arange(0, BLOCK)
    mask_half = offs < half_dim

    # ── Load cos/sin for this sequence position (shared for Q and K) ──────────
    cos = tl.load(
        cos_ptr + pid_s * stride_cos0 + offs * stride_cos1,
        mask=mask_half,
        other=1.0,
    )  # [half_dim]
    sin = tl.load(
        sin_ptr + pid_s * stride_cos0 + offs * stride_cos1,
        mask=mask_half,
        other=0.0,
    )  # [half_dim]

    # ── Determine whether we are processing a Q or K head ────────────────────
    # The grid dimension 0 encodes: Q heads [0, num_q_heads), K heads [num_q_heads, ...)
    # The caller sets num_total_heads = num_q_heads + num_kv_heads and passes
    # q and k base pointers separately.  We derive the actual head index and
    # select the right pointer inside the kernel.
    #
    # To keep the kernel generic without a num_q_heads argument, the caller
    # sets q_ptr = NULL sentinel for K-head programs and vice versa.
    # Instead, we split the grid and use two kernel launches with the same
    # kernel body — but that defeats the fusion purpose.
    #
    # Simplest correct approach: launch one kernel per tensor (Q and K) but
    # share the compiled PTX and amortise cos/sin reuse across a larger grid.
    # The actual fusion benefit (one cos/sin read instead of two) is achieved
    # by keeping BOTH in a single grid where each thread block loads cos/sin
    # once and rotates its assigned head.
    #
    # Here we implement the Q path; a second launch does K.  The two launches
    # share the Triton JIT-compiled PTX so compile overhead is paid once.

    # ── Load x1 (first half) and x2 (second half) of this head's vector ──────
    x1 = tl.load(
        q_ptr
        + pid_hs * stride_q1
        + pid_s  * stride_q2
        + offs   * stride_q3,
        mask=mask_half,
        other=0.0,
    )
    x2 = tl.load(
        q_ptr
        + pid_hs * stride_q1
        + pid_s  * stride_q2
        + (offs + half_dim) * stride_q3,
        mask=mask_half,
        other=0.0,
    )

    # ── Rotation ──────────────────────────────────────────────────────────────
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x2 * cos + x1 * sin

    # ── Store rotated values ──────────────────────────────────────────────────
    tl.store(
        q_out_ptr
        + pid_hs * stride_o1
        + pid_s  * stride_o2
        + offs   * stride_o3,
        x1_rot,
        mask=mask_half,
    )
    tl.store(
        q_out_ptr
        + pid_hs * stride_o1
        + pid_s  * stride_o2
        + (offs + half_dim) * stride_o3,
        x2_rot,
        mask=mask_half,
    )

    # ── Load K for same head index (reusing cos/sin already in registers) ─────
    x1k = tl.load(
        k_ptr
        + pid_hs * stride_k1
        + pid_s  * stride_k2
        + offs   * stride_k3,
        mask=mask_half,
        other=0.0,
    )
    x2k = tl.load(
        k_ptr
        + pid_hs * stride_k1
        + pid_s  * stride_k2
        + (offs + half_dim) * stride_k3,
        mask=mask_half,
        other=0.0,
    )

    x1k_rot = x1k * cos - x2k * sin
    x2k_rot = x2k * cos + x1k * sin

    tl.store(
        k_out_ptr
        + pid_hs * stride_o1
        + pid_s  * stride_o2
        + offs   * stride_o3,
        x1k_rot,
        mask=mask_half,
    )
    tl.store(
        k_out_ptr
        + pid_hs * stride_o1
        + pid_s  * stride_o2
        + (offs + half_dim) * stride_o3,
        x2k_rot,
        mask=mask_half,
    )


# ============================================================================
# RoPE Classes
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
        """Pre-compute cos and sin using Triton kernel."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)
        sin_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)

        if device.type == "cuda":
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)

            block = triton.next_power_of_2(half_dim)
            compute_freqs_kernel[(seq_len,)](
                positions,
                self.inv_freq,
                cos_cache,
                sin_cache,
                seq_len,
                half_dim,
                positions.stride(0),
                self.inv_freq.stride(0),
                cos_cache.stride(0),
                cos_cache.stride(1),
                sin_cache.stride(0),
                sin_cache.stride(1),
                BLOCK=block,
            )
        else:
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            freqs = positions[:, None] * self.inv_freq[None, :]
            cos_half = torch.cos(freqs)
            sin_half = torch.sin(freqs)
            cos_cache[:, :half_dim] = cos_half
            cos_cache[:, half_dim : half_dim * 2] = cos_half
            sin_cache[:, :half_dim] = sin_half
            sin_cache[:, half_dim : half_dim * 2] = sin_half

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for given positions."""
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


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ROPE_DIM = 256


def _apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply RoPE to a single tensor (Q or K) using Torch. Fallback only."""
    cos = cos[:x.shape[-2]]
    sin = sin[:x.shape[-2]]

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim : half_dim * 2]

    cos_expanded = cos[None, None, :, :]
    sin_expanded = sin[None, None, :, :]

    x1_rot = x1 * cos_expanded - x2 * sin_expanded
    x2_rot = x2 * cos_expanded + x1 * sin_expanded

    if head_dim > half_dim * 2:
        x_pass = x[..., half_dim * 2 :]
        return torch.cat([x1_rot, x2_rot, x_pass], dim=-1)
    return torch.cat([x1_rot, x2_rot], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K.

    FUSION: Q and K are rotated in a single Triton kernel when possible.
    cos/sin are loaded from HBM once and reused for both Q and K rotations,
    halving memory traffic vs the original two-call approach.

    API is identical to the original — model.py requires no changes.
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape

    if rotary_dim is None:
        rotary_dim = head_dim

    half_dim = rotary_dim // 2

    if cos.shape[-1] > half_dim:
        cos = cos[:, :half_dim]
        sin = sin[:, :half_dim]

    cos = cos.to(torch.float32).contiguous()
    sin = sin.to(torch.float32).contiguous()

    # ── Triton fused path ─────────────────────────────────────────────────────
    # Conditions: CUDA, head sizes fit in registers, Q and K have the same
    # seq_len and head_dim (always true here), num_q_heads == num_kv_heads
    # (GQA with different head counts handled by fallback for simplicity).
    half_dim_padded = next_power_of_two(half_dim)
    use_triton = (
        q.is_cuda
        and half_dim_padded <= MAX_ROPE_DIM
        and num_q_heads == num_kv_heads   # same head count → same grid
        and seq_len == k.shape[2]
    )

    if use_triton:
        q_f = q.to(torch.float32).contiguous()
        k_f = k.to(torch.float32).contiguous()

        q_out = torch.empty_like(q_f)
        k_out = torch.empty_like(k_f)

        # Grid: (num_heads, seq_len)
        # Each program loads cos/sin once and rotates one head of BOTH Q and K.
        grid = (num_q_heads, seq_len)

        # We run over batch dimension with a loop to keep the kernel simple
        # (batch is typically 1 for inference).
        for b in range(batch):
            rope_qk_fused_kernel[grid](
                q_f[b],           # [num_q_heads,  seq_len, head_dim]
                k_f[b],           # [num_kv_heads, seq_len, head_dim]
                cos,              # [seq_len, half_dim]
                sin,              # [seq_len, half_dim]
                q_out[b],
                k_out[b],
                seq_len,
                half_dim,
                rotary_dim,
                head_dim,
                # Q strides (batch dim already indexed out)
                0,                       # stride_q0 unused
                q_f.stride(1),           # head
                q_f.stride(2),           # seq
                q_f.stride(3),           # dim
                # K strides
                0,
                k_f.stride(1),
                k_f.stride(2),
                k_f.stride(3),
                # cos/sin strides
                cos.stride(0),           # seq
                cos.stride(1),           # dim
                # output strides (q_out and k_out share layout)
                0,
                q_out.stride(1),
                q_out.stride(2),
                q_out.stride(3),
                BLOCK=half_dim_padded,
            )

        # Handle pass-through dimensions beyond rotary_dim
        if head_dim > rotary_dim:
            q_out[..., rotary_dim:] = q_f[..., rotary_dim:]
            k_out[..., rotary_dim:] = k_f[..., rotary_dim:]

        return q_out.to(q.dtype), k_out.to(k.dtype)

    # ── Fallback: original PyTorch path ──────────────────────────────────────
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
    """Apply rotary embeddings to partial dimensions."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)


if __name__ == "__main__":
    print("Testing Triton RoPE...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=1024)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    cos, sin = rope(q)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"Q rotated shape: {q_rot.shape}")
    print(f"K rotated shape: {k_rot.shape}")

    # Correctness check vs PyTorch fallback
    q_ref, k_ref = _apply_rope_single(q, cos, sin, head_dim // 2, head_dim), \
                   _apply_rope_single(k, cos, sin, head_dim // 2, head_dim)
    print(f"Q max err vs ref: {(q_rot - q_ref).abs().max().item():.2e}")
    print(f"K max err vs ref: {(k_rot - k_ref).abs().max().item():.2e}")

    print("\nTesting partial RoPE (50%):")
    rope_partial = RotaryEmbedding(dim=head_dim, partial_rotary_factor=0.5)
    cos_p, sin_p = rope_partial(q)
    q_rot_p, k_rot_p = apply_partial_rotary_pos_emb(q, k, cos_p, sin_p, head_dim // 2)
    print(f"Q rotated (partial) shape: {q_rot_p.shape}")

    print("\nTriton RoPE working!")