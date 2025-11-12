import math
import pytest
import torch

from triton_int4.triton_kernels.quant_i4_pack8 import (
    dequantize_i4_pack8,
    quantize_i4_pack8,
)

@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return "cuda"

def test_memory_reduction(device):
    x = torch.randn(64, 512, dtype=torch.float16, device=device)
    packed, _ = quantize_i4_pack8(x)
    orig_bytes = x.numel() * x.element_size()
    packed_bytes = packed.numel() * packed.element_size()
    assert math.isclose(packed_bytes / orig_bytes, 0.25, rel_tol=1e-3)

def test_roundtrip_accuracy(device):
    x = torch.randn(32, 256, dtype=torch.float16, device=device)
    packed, scales = quantize_i4_pack8(x)
    restored = dequantize_i4_pack8(packed, scales)
    x_fp32 = x.to(torch.float32)
    diff = (x_fp32 - restored).abs().mean()
    assert diff < 0.08
