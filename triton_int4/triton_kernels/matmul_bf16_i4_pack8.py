import torch
from triton_int4.triton_kernels.quant_i4_pack8 import dequantize_i4_pack8

def matmul_bf16_i4(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    scales: torch.Tensor,
    pack_dtype: str = "int32",
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    """Stub: dequant on CPU and matmul via torch; matches final API."""
    if a.dtype != torch.bfloat16:
        raise TypeError("expected bf16 activations")
    a16 = a.to(torch.float16)
    w = dequantize_i4_pack8(b_packed, scales, pack_dtype=pack_dtype).to(torch.float16)
    return torch.matmul(a16, w.t()).to(torch.float32)
