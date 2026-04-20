import torch
import torch.nn as nn
import triton
import triton.language as tl

# Define the custom Triton kernel for element-wise addition
@triton.jit
def add_kernel(a_ptr, b_ptr, o_ptr, N, TILE_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    off = tl.arange(0, TILE_SIZE) + pid * TILE_SIZE
    mask = off < N

    a = tl.load(a_ptr + off, mask=mask)
    b = tl.load(b_ptr + off, mask=mask)

    o = a + b

    tl.store(o_ptr + off, o, mask=mask)


# wrap Triton kernel
def add(a: torch.Tensor, b: torch.Tensor):
    o = torch.empty_like(a)
    N = a.numel()
    TILE_SIZE = 512
    grid = (triton.cdiv(N, TILE_SIZE), 1, 1)
    add_kernel[grid](a, b, o, N, TILE_SIZE)
    return o


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return add(a, b)
