from typing import Literal, Tuple

import torch
import triton
import triton.language as tl

DTYPE_TO_PACK = {"int8": 2, "int32": 8}
EPS = 1e-8

@triton.jit
def _quant_pack(
    x_ptr, scales_ptr, out_ptr,
    M, N,
    stride_x_m, stride_x_n,
    stride_scales_m, stride_scales_p,
    stride_out_m,
    elems_per_pack: tl.constexpr,
    block_packs: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_pack = tl.program_id(1)
    if pid_m >= M:
        return
    pack_ids = pid_pack * block_packs + tl.arange(0, block_packs)
    num_packs = (N + elems_per_pack - 1) // elems_per_pack
    mask_pack = pack_ids < num_packs
    lanes = tl.arange(0, elems_per_pack)[None, :]
    base_cols = pack_ids[:, None] * elems_per_pack + lanes
    mask_cols = base_cols < N

    s_ptrs = scales_ptr + pid_m * stride_scales_m + pack_ids * stride_scales_p
    scales = tl.load(s_ptrs, mask=mask_pack, other=1.0).to(tl.float32)[:, None]

    ptrs = x_ptr + pid_m * stride_x_m + base_cols * stride_x_n
    vals = tl.load(ptrs, mask=mask_cols, other=0.0).to(tl.float32)

    # ties-to-even rounding for halves; clamp to [-8, 7]
    x = vals / scales
    f = tl.math.floor(x)
    r = tl.math.floor(x + 0.5)
    is_half = (x - f) == 0.5
    r_adj = tl.where(is_half & ((r.to(tl.int32) & 1) == 1), r - 1.0, r)
    q = tl.maximum(tl.minimum(r_adj, 7.0), -8.0).to(tl.int32) + 8

    shifts = (tl.arange(0, elems_per_pack)[None, :] * 4).to(tl.int32)
    nibbles = (q & 0xF) << shifts
    pack_vals = tl.sum(nibbles, axis=1).to(tl.int32)
    out_ptrs = out_ptr + pid_m * stride_out_m + pack_ids
    tl.store(out_ptrs, pack_vals, mask=mask_pack)

def quantize_i4_pack8(
    x: torch.Tensor,
    pack_dtype: Literal["int8", "int32"] = "int32",
    block_packs: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dtype != torch.float16:
        raise TypeError("expected fp16 input")
    if not x.is_contiguous():
        x = x.contiguous()
    M, N = x.shape
    elems_per_pack = DTYPE_TO_PACK[pack_dtype]
    if N % elems_per_pack != 0:
        raise ValueError("cols must align to pack size")
    num_packs = N // elems_per_pack
    x32 = x.float()
    scales = (
        x32.abs().view(M, num_packs, elems_per_pack).amax(dim=2).clamp_min(EPS) / 7.0
    ).contiguous()
    out_dtype = torch.int8 if pack_dtype == "int8" else torch.int32
    out = torch.empty((M, num_packs), dtype=out_dtype, device=x.device)
    grid = (M, triton.cdiv(num_packs, block_packs))
    _quant_pack[grid](
        x, scales, out,
        M, N,
        x.stride(0), x.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0),
        elems_per_pack=elems_per_pack,
        block_packs=block_packs,
    )
    return out, scales

def dequantize_i4_pack8(packed: torch.Tensor, scales: torch.Tensor, pack_dtype: Literal["int8","int32"]="int32") -> torch.Tensor:
    elems_per_pack = DTYPE_TO_PACK[pack_dtype]
    M, num_packs = packed.shape
    K = num_packs * elems_per_pack
    out = torch.empty((M, K), dtype=torch.float32, device=packed.device)
    data = packed.to(torch.int32)
    # lanes write into columns lane, lane+L, ...
    for lane in range(elems_per_pack):
        shift = lane * 4
        vals = (data >> shift) & 0xF
        chunk = vals.to(torch.float32) - 8.0
        out[:, lane::elems_per_pack] = chunk * scales
    return out
