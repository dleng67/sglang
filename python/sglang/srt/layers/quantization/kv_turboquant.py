# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
TurboQuant KV cache quantization for SGLang.

Algorithm (per block of BLOCK_SIZE=32 along head_dim):
  1. Normalize: scale = ||block||₂
  2. Rotate: golden-ratio sign flips + Walsh-Hadamard Transform
  3. Quantize: nearest Lloyd-Max centroid (3-bit or 4-bit)
  4. Pack: 3-bit → 8 values per 3 bytes; 4-bit → 2 values per byte

After rotation each coordinate is approximately N(0, 1/sqrt(BLOCK_SIZE)),
making a single precomputed Lloyd-Max codebook optimal for all layers/models.

Reference: "TurboQuant: Efficient KV Cache Quantization via Randomized
Codebooks" (Google Research, 2025) https://arxiv.org/abs/2504.19874

Community implementations:
  - tonbistudio/turboquant-pytorch (MIT)
  - TheTom/turboquant_plus (Apache 2.0)
  - Aaryan-Kapoor/llama.cpp TQ3_0 format
"""

import math
from typing import Tuple

import torch

# Block size for per-block quantization. Must be a power of 2 for WHT.
# Matches the validated llama.cpp TQ3_0 block size.
TURBOQUANT_BLOCK_SIZE = 32


def _compute_lloyd_max_centroids(bits: int, n_iter: int = 200) -> torch.Tensor:
    """
    Compute Lloyd-Max optimal scalar quantization centroids for the
    N(0, 1/sqrt(BLOCK_SIZE)) distribution that arises after WHT rotation
    of a unit-norm block.

    Runs once at startup; takes ~0.1s.
    """
    import numpy as np
    from scipy.stats import norm as gaussian

    n_levels = 2**bits
    sigma = 1.0 / math.sqrt(TURBOQUANT_BLOCK_SIZE)

    # Initialize decision boundaries via uniform quantile spacing
    quantiles = np.linspace(0, 1, n_levels + 1)[1:-1]
    boundaries = gaussian.ppf(quantiles, scale=sigma)

    for _ in range(n_iter):
        b_ext = np.concatenate([[-np.inf], boundaries, [np.inf]])
        centroids = np.zeros(n_levels)
        for j in range(n_levels):
            lo, hi = b_ext[j], b_ext[j + 1]
            num = sigma * (gaussian.pdf(lo / sigma) - gaussian.pdf(hi / sigma))
            den = gaussian.cdf(hi / sigma) - gaussian.cdf(lo / sigma)
            centroids[j] = num / max(den, 1e-12)
        boundaries = (centroids[:-1] + centroids[1:]) / 2

    return torch.tensor(centroids, dtype=torch.float32)


def _golden_ratio_signs(n: int) -> torch.Tensor:
    """
    Deterministic sign-flip pattern from the golden ratio hash.
    Matches the sign pattern used in llama.cpp TQ3_0 (Aaryan-Kapoor fork).
    """
    PHI = (1.0 + math.sqrt(5.0)) / 2.0
    signs = torch.ones(n)
    for i in range(n):
        # Use integer arithmetic to avoid float precision issues
        if int(i * PHI * (2**20)) % 2 == 1:
            signs[i] = -1.0
    return signs


def _fast_wht(x: torch.Tensor) -> torch.Tensor:
    """
    In-place Walsh-Hadamard Transform over the last dimension.
    x: (..., N) where N must be a power of 2.
    Returns the unnormalized WHT.
    """
    N = x.shape[-1]
    h = 1
    while h < N:
        x = x.view(*x.shape[:-1], N // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :].clone()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.view(*x.shape[:-3], N)
        h *= 2
    return x


def _pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack 3-bit indices (stored as uint8, values 0-7) into dense 3-bit format.
    Packs 8 consecutive values into 3 bytes (24 bits).

    Bit layout per 8-value group (v0..v7, each 3 bits):
      byte0 = v0[2:0] | v1[2:0]<<3 | v2[1:0]<<6
      byte1 = v2[2]   | v3[2:0]<<1 | v4[2:0]<<4 | v5[1:0]<<7
      byte2 = v5[2:1] | v6[2:0]<<2 | v7[2:0]<<5

    Args:
        indices: (..., N) uint8, N divisible by 8, values in [0, 7]
    Returns:
        packed: (..., N*3//8) uint8
    """
    N = indices.shape[-1]
    assert N % 8 == 0, f"N={N} must be divisible by 8 for 3-bit packing"
    x = indices.view(*indices.shape[:-1], N // 8, 8).long()

    byte0 = (x[..., 0] & 7) | ((x[..., 1] & 7) << 3) | ((x[..., 2] & 3) << 6)
    byte1 = (
        ((x[..., 2] >> 2) & 1)
        | ((x[..., 3] & 7) << 1)
        | ((x[..., 4] & 7) << 4)
        | ((x[..., 5] & 3) << 7)
    )
    byte2 = ((x[..., 5] >> 1) & 3) | ((x[..., 6] & 7) << 2) | ((x[..., 7] & 7) << 5)

    packed = torch.stack([byte0, byte1, byte2], dim=-1).to(torch.uint8)
    return packed.view(*indices.shape[:-1], N * 3 // 8)


def _unpack_3bit(packed: torch.Tensor, N: int) -> torch.Tensor:
    """
    Unpack 3-bit packed data back to uint8 indices (values 0-7).

    Args:
        packed: (..., N*3//8) uint8
        N: original number of values (must be divisible by 8)
    Returns:
        indices: (..., N) uint8
    """
    assert N % 8 == 0
    x = packed.view(*packed.shape[:-1], N // 8, 3).long()

    v = torch.empty(
        *packed.shape[:-1], N // 8, 8, dtype=torch.long, device=packed.device
    )
    v[..., 0] = x[..., 0] & 7
    v[..., 1] = (x[..., 0] >> 3) & 7
    v[..., 2] = ((x[..., 0] >> 6) & 3) | ((x[..., 1] & 1) << 2)
    v[..., 3] = (x[..., 1] >> 1) & 7
    v[..., 4] = (x[..., 1] >> 4) & 7
    v[..., 5] = ((x[..., 1] >> 7) & 1) | ((x[..., 2] & 3) << 1)
    v[..., 6] = (x[..., 2] >> 2) & 7
    v[..., 7] = (x[..., 2] >> 5) & 7

    return v.view(*packed.shape[:-1], N).to(torch.uint8)


def _pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack 4-bit indices (values 0-15) into nibble format: 2 values per byte.

    Args:
        indices: (..., N) uint8, N even, values in [0, 15]
    Returns:
        packed: (..., N//2) uint8
    """
    even = indices[..., 0::2].to(torch.uint8) & 0xF
    odd = (indices[..., 1::2].to(torch.uint8) & 0xF) << 4
    return even | odd


def _unpack_4bit(packed: torch.Tensor, N: int) -> torch.Tensor:
    """
    Unpack nibble-packed data to uint8 indices.

    Args:
        packed: (..., N//2) uint8
        N: original number of values (even)
    Returns:
        indices: (..., N) uint8
    """
    out = torch.empty(*packed.shape[:-1], N, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = packed & 0xF
    out[..., 1::2] = (packed >> 4) & 0xF
    return out


class KVTurboQuantUtil:
    """
    TurboQuant KV cache quantization utility.

    Quantizes [num_tokens, head_num, head_dim] KV tensors using per-block
    Lloyd-Max codebook quantization with randomized WHT rotation.

    Memory layout per token per head:
      3-bit: head_dim * 3/8 bytes (indices) + head_dim/32 * 2 bytes (bf16 scales)
             Effective: ~3.5 bits/value  →  ~4.6x compression vs bfloat16
      4-bit: head_dim / 2 bytes (indices) + head_dim/32 * 2 bytes (bf16 scales)
             Effective: ~4.5 bits/value  →  ~3.6x compression vs bfloat16

    Example (head_dim=128):
      bf16 baseline:    128 * 2 = 256 bytes/head
      turboquant3:       48 +  8 =  56 bytes/head  (4.57x)
      turboquant4:       64 +  8 =  72 bytes/head  (3.56x)
    """

    _instance_cache: dict = {}

    def __init__(self, bits: int, head_dim: int, device: str = "cuda"):
        if bits not in (3, 4):
            raise ValueError(f"TurboQuant supports bits=3 or bits=4, got {bits}")
        if head_dim % TURBOQUANT_BLOCK_SIZE != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by "
                f"TURBOQUANT_BLOCK_SIZE={TURBOQUANT_BLOCK_SIZE}"
            )

        self.bits = bits
        self.head_dim = head_dim
        self.n_levels = 2**bits
        self.device = device
        self.num_blocks = head_dim // TURBOQUANT_BLOCK_SIZE
        self.wht_norm = 1.0 / math.sqrt(TURBOQUANT_BLOCK_SIZE)

        # Precompute codebook centroids for N(0, 1/sqrt(BLOCK_SIZE)) distribution
        centroids = _compute_lloyd_max_centroids(bits)
        self.centroids = centroids.to(device)  # [n_levels]

        # Precompute deterministic sign-flip pattern
        signs = _golden_ratio_signs(TURBOQUANT_BLOCK_SIZE)
        self.signs = signs.to(device)  # [BLOCK_SIZE]

    @classmethod
    def get_instance(cls, bits: int, head_dim: int, device: str) -> "KVTurboQuantUtil":
        """Return a cached instance to avoid recomputing codebooks."""
        key = (bits, head_dim, device)
        if key not in cls._instance_cache:
            cls._instance_cache[key] = cls(bits, head_dim, device)
        return cls._instance_cache[key]

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a KV cache tensor.

        Args:
            x: [num_tokens, head_num, head_dim] in bf16/fp16/fp32
        Returns:
            packed:  [num_tokens, head_num, packed_dim] uint8
                     packed_dim = head_dim*3//8 (3-bit) or head_dim//2 (4-bit)
            scales:  [num_tokens, head_num, num_blocks] bfloat16
                     num_blocks = head_dim // TURBOQUANT_BLOCK_SIZE
        """
        T, H, D = x.shape
        NB = self.num_blocks

        # Split into blocks: [T, H, NB, BLOCK_SIZE]
        blocks = x.float().view(T, H, NB, TURBOQUANT_BLOCK_SIZE)

        # Per-block ℓ₂ norm (scale)
        scales = blocks.norm(dim=-1, keepdim=True)  # [T, H, NB, 1]
        blocks_normed = blocks / (scales + 1e-8)

        # Randomized Hadamard Transform:
        #   1. Apply sign flips (the "randomization")
        #   2. WHT, normalized by 1/sqrt(BLOCK_SIZE) → orthogonal transform
        blocks_signed = blocks_normed * self.signs  # broadcast over last dim
        blocks_rotated = (
            _fast_wht(blocks_signed) * self.wht_norm
        )  # [T, H, NB, BLOCK_SIZE]

        # Nearest-centroid quantization
        # [T, H, NB, BLOCK_SIZE, 1] vs [n_levels] → argmin over n_levels dim
        diffs = (blocks_rotated.unsqueeze(-1) - self.centroids).abs()
        indices = diffs.argmin(dim=-1).to(torch.uint8)  # [T, H, NB, BLOCK_SIZE]

        # Flatten blocks back to head_dim: [T, H, D]
        indices_flat = indices.view(T, H, D)

        # Bit-pack
        if self.bits == 3:
            packed = _pack_3bit(indices_flat)  # [T, H, D*3//8]
        else:
            packed = _pack_4bit(indices_flat)  # [T, H, D//2]

        scales_out = scales.squeeze(-1).to(torch.bfloat16)  # [T, H, NB]
        return packed, scales_out

    def dequantize(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Reconstruct a KV cache tensor from its compressed representation.

        Args:
            packed: [T, H, packed_dim] uint8
            scales: [T, H, num_blocks] bfloat16
            dtype:  output dtype (default bfloat16)
        Returns:
            x: [T, H, head_dim] in dtype
        """
        T, H = packed.shape[:2]
        D = self.head_dim
        NB = self.num_blocks

        # Unpack indices
        if self.bits == 3:
            indices_flat = _unpack_3bit(packed, D)  # [T, H, D] uint8
        else:
            indices_flat = _unpack_4bit(packed, D)  # [T, H, D] uint8

        # Reshape: [T, H, NB, BLOCK_SIZE]
        indices = indices_flat.view(T, H, NB, TURBOQUANT_BLOCK_SIZE).long()

        # Centroid lookup
        blocks_rotated = self.centroids[indices]  # [T, H, NB, BLOCK_SIZE]

        # Inverse WHT: the normalized WHT is its own inverse
        # (H_norm @ H_norm = I for orthogonal normalization)
        blocks_signed = _fast_wht(blocks_rotated) * self.wht_norm

        # Undo sign flips
        blocks_normed = blocks_signed * self.signs  # [T, H, NB, BLOCK_SIZE]

        # Apply per-block scales: [T, H, NB, 1] * [T, H, NB, BLOCK_SIZE]
        scales_fp32 = scales.float().unsqueeze(-1)  # [T, H, NB, 1]
        blocks = blocks_normed * scales_fp32

        return blocks.view(T, H, D).to(dtype)


def packed_dim(bits: int, head_dim: int) -> int:
    """Return the packed index storage size (in uint8 elements) per head."""
    if bits == 3:
        return head_dim * 3 // 8
    else:  # 4-bit
        return head_dim // 2


def scale_dim(head_dim: int) -> int:
    """Return the number of bfloat16 scales per head."""
    return head_dim // TURBOQUANT_BLOCK_SIZE


def bytes_per_token_per_head(bits: int, head_dim: int) -> float:
    """Total bytes per token per head including packed indices and scales."""
    index_bytes = packed_dim(bits, head_dim)  # uint8 elements = bytes
    scale_bytes = scale_dim(head_dim) * 2  # bfloat16 = 2 bytes each
    return float(index_bytes + scale_bytes)


def compression_ratio_vs_bf16(bits: int, head_dim: int) -> float:
    """Compression ratio relative to bfloat16 baseline."""
    bf16_bytes = head_dim * 2
    return bf16_bytes / bytes_per_token_per_head(bits, head_dim)
