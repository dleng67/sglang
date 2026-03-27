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
Unit tests for TurboQuant KV cache quantization.

Tests cover:
  1. Pack / unpack round-trip for 3-bit and 4-bit
  2. Quantize → dequantize cosine similarity (quality check)
  3. MHATokenToKVPoolTurboQuant: set_kv_buffer → get_key/value_buffer
  4. Memory savings vs bfloat16 baseline
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-test-1-gpu-small")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HEAD_DIM = 128
V_HEAD_DIM = 128
HEAD_NUM = 4
NUM_TOKENS = 64
LAYER_NUM = 2
PAGE_SIZE = 16
POOL_SIZE = NUM_TOKENS * 4


class TestTurboQuantPackUnpack(unittest.TestCase):
    """Test 3-bit and 4-bit pack/unpack round-trips."""

    def _test_pack_unpack_3bit(self, N: int):
        from sglang.srt.layers.quantization.kv_turboquant import (
            _pack_3bit,
            _unpack_3bit,
        )

        indices = torch.randint(0, 8, (4, 2, N), dtype=torch.uint8, device=DEVICE)
        packed = _pack_3bit(indices)
        unpacked = _unpack_3bit(packed, N)
        self.assertEqual(packed.shape, (*indices.shape[:-1], N * 3 // 8))
        self.assertTrue(torch.all(unpacked == indices))

    def _test_pack_unpack_4bit(self, N: int):
        from sglang.srt.layers.quantization.kv_turboquant import (
            _pack_4bit,
            _unpack_4bit,
        )

        indices = torch.randint(0, 16, (4, 2, N), dtype=torch.uint8, device=DEVICE)
        packed = _pack_4bit(indices)
        unpacked = _unpack_4bit(packed, N)
        self.assertEqual(packed.shape, (*indices.shape[:-1], N // 2))
        self.assertTrue(torch.all(unpacked == indices))

    def test_pack_unpack_3bit_32(self):
        self._test_pack_unpack_3bit(32)

    def test_pack_unpack_3bit_128(self):
        self._test_pack_unpack_3bit(128)

    def test_pack_unpack_3bit_256(self):
        self._test_pack_unpack_3bit(256)

    def test_pack_unpack_4bit_32(self):
        self._test_pack_unpack_4bit(32)

    def test_pack_unpack_4bit_128(self):
        self._test_pack_unpack_4bit(128)


class TestKVTurboQuantUtil(unittest.TestCase):
    """Test quantize/dequantize quality and correctness."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for TurboQuant tests")

    def _make_kv(self, T=NUM_TOKENS, H=HEAD_NUM, D=HEAD_DIM):
        """Generate realistic KV tensors (standard-normal)."""
        return torch.randn(T, H, D, dtype=torch.bfloat16, device=DEVICE)

    def _cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Mean cosine similarity between reconstructed and original vectors."""
        a_f = a.float().view(-1, a.shape[-1])
        b_f = b.float().view(-1, b.shape[-1])
        cos = torch.nn.functional.cosine_similarity(a_f, b_f, dim=-1)
        return float(cos.mean().item())

    def test_quantize_dequantize_3bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import KVTurboQuantUtil

        tq = KVTurboQuantUtil(bits=3, head_dim=HEAD_DIM, device=DEVICE)
        x = self._make_kv()
        packed, scales = tq.quantize(x)
        x_hat = tq.dequantize(packed, scales)

        cos_sim = self._cosine_sim(x, x_hat)
        self.assertGreater(cos_sim, 0.99, f"3-bit cosine sim too low: {cos_sim:.4f}")

    def test_quantize_dequantize_4bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import KVTurboQuantUtil

        tq = KVTurboQuantUtil(bits=4, head_dim=HEAD_DIM, device=DEVICE)
        x = self._make_kv()
        packed, scales = tq.quantize(x)
        x_hat = tq.dequantize(packed, scales)

        cos_sim = self._cosine_sim(x, x_hat)
        self.assertGreater(cos_sim, 0.995, f"4-bit cosine sim too low: {cos_sim:.4f}")

    def test_output_shapes_3bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            KVTurboQuantUtil,
            packed_dim,
            scale_dim,
        )

        tq = KVTurboQuantUtil(bits=3, head_dim=HEAD_DIM, device=DEVICE)
        x = self._make_kv()
        packed, scales = tq.quantize(x)

        expected_packed = packed_dim(3, HEAD_DIM)
        expected_scales = scale_dim(HEAD_DIM)
        self.assertEqual(packed.shape, (NUM_TOKENS, HEAD_NUM, expected_packed))
        self.assertEqual(scales.shape, (NUM_TOKENS, HEAD_NUM, expected_scales))

    def test_output_shapes_4bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            KVTurboQuantUtil,
            packed_dim,
            scale_dim,
        )

        tq = KVTurboQuantUtil(bits=4, head_dim=HEAD_DIM, device=DEVICE)
        x = self._make_kv()
        packed, scales = tq.quantize(x)

        expected_packed = packed_dim(4, HEAD_DIM)
        expected_scales = scale_dim(HEAD_DIM)
        self.assertEqual(packed.shape, (NUM_TOKENS, HEAD_NUM, expected_packed))
        self.assertEqual(scales.shape, (NUM_TOKENS, HEAD_NUM, expected_scales))

    def test_3bit_better_compression_than_4bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            compression_ratio_vs_bf16,
        )

        ratio3 = compression_ratio_vs_bf16(3, HEAD_DIM)
        ratio4 = compression_ratio_vs_bf16(4, HEAD_DIM)
        self.assertGreater(ratio3, ratio4)
        # Sanity: 3-bit should be at least 3x (well above 1x)
        self.assertGreater(ratio3, 3.0)

    def test_instance_cache(self):
        from sglang.srt.layers.quantization.kv_turboquant import KVTurboQuantUtil

        a = KVTurboQuantUtil.get_instance(3, HEAD_DIM, DEVICE)
        b = KVTurboQuantUtil.get_instance(3, HEAD_DIM, DEVICE)
        self.assertIs(a, b)


class TestMHATokenToKVPoolTurboQuant(unittest.TestCase):
    """Test the KV cache memory pool with TurboQuant compression."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for TurboQuant pool tests")

    def _make_pool(self, bits: int):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolTurboQuant

        return MHATokenToKVPoolTurboQuant(
            bits=bits,
            size=POOL_SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            layer_num=LAYER_NUM,
            device=DEVICE,
            enable_memory_saver=False,
        )

    def _cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a_f = a.float().view(-1, a.shape[-1])
        b_f = b.float().view(-1, b.shape[-1])
        return float(
            torch.nn.functional.cosine_similarity(a_f, b_f, dim=-1).mean().item()
        )

    def _dummy_layer(self, layer_id: int = 0):
        """Minimal RadixAttention stub with a layer_id attribute."""

        class FakeLayer:
            pass

        layer = FakeLayer()
        layer.layer_id = layer_id
        return layer

    def _run_set_get(self, bits: int, threshold: float):
        pool = self._make_pool(bits)
        layer = self._dummy_layer(layer_id=0)

        # Generate random KV tensors and write to pool
        loc = torch.arange(NUM_TOKENS, device=DEVICE)
        cache_k = torch.randn(
            NUM_TOKENS, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
        )
        cache_v = torch.randn(
            NUM_TOKENS, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
        )

        pool.set_kv_buffer(layer, loc, cache_k, cache_v)

        # Read back full buffer and slice out written tokens
        k_full = pool.get_key_buffer(layer.layer_id)
        v_full = pool.get_value_buffer(layer.layer_id)
        k_read = k_full[loc]
        v_read = v_full[loc]

        cos_k = self._cosine_sim(cache_k, k_read)
        cos_v = self._cosine_sim(cache_v, v_read)
        self.assertGreater(cos_k, threshold, f"K cosine sim {cos_k:.4f} < {threshold}")
        self.assertGreater(cos_v, threshold, f"V cosine sim {cos_v:.4f} < {threshold}")

    def test_set_get_3bit(self):
        self._run_set_get(bits=3, threshold=0.98)

    def test_set_get_4bit(self):
        self._run_set_get(bits=4, threshold=0.99)

    def test_memory_savings_3bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            compression_ratio_vs_bf16,
        )

        pool = self._make_pool(bits=3)
        k_bytes, v_bytes = pool.get_kv_size_bytes()
        actual_bytes = k_bytes + v_bytes

        # What bf16 would use
        bf16_bytes = (
            POOL_SIZE
            * HEAD_NUM
            * HEAD_DIM
            * 2  # bfloat16 = 2 bytes
            * 2  # K and V
            * LAYER_NUM
        )
        actual_ratio = bf16_bytes / actual_bytes
        expected_ratio = compression_ratio_vs_bf16(3, HEAD_DIM)

        self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.1)
        self.assertGreater(actual_ratio, 3.0)

    def test_kv_buffer_dtype(self):
        pool = self._make_pool(bits=3)
        k_buf = pool.get_key_buffer(0)
        self.assertEqual(k_buf.dtype, torch.bfloat16)

    def test_multi_layer(self):
        pool = self._make_pool(bits=3)

        for layer_id in range(LAYER_NUM):
            layer = self._dummy_layer(layer_id=layer_id)
            loc = torch.arange(8, device=DEVICE)
            cache_k = torch.randn(
                8, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
            )
            cache_v = torch.randn(
                8, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
            )
            pool.set_kv_buffer(layer, loc, cache_k, cache_v)

        # Verify each layer is stored independently
        k0 = pool.get_key_buffer(0)[torch.arange(8, device=DEVICE)]
        k1 = pool.get_key_buffer(1)[torch.arange(8, device=DEVICE)]
        # Different layers should have different (independent) data
        self.assertFalse(torch.allclose(k0, k1))


if __name__ == "__main__":
    unittest.main()
