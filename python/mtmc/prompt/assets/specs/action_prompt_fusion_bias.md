You need to fuse **only the bias terms** of gemm or convolution-related operators (conv and conv_transpose) with the subsequent selected operators. 

Note that gemm operations = matmul + gemm_bias, and convolution-related operations = conv + conv_bias. Only fuse gemm_bias and conv_bias with subsequent operators, and do not fuse the main computational parts (matmul and conv).

The overall steps are as follows:
1. Convert the gemm or convolution-related operators into the format of nn.functional, and explicitly write out weight and bias of gemm or convolution-related operators.
```
def __init__(self, ...):
    super(Model, self).__init__()
    # Maintain the original implementation
    # Example 1: conv_transpose
    self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
    # Example 2: gemm
    self.gemm = nn.Linear(in_features, out_features)
    # Example 3: conv
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    
def forward(self, x, ...):
    # Explicitly write out the bias and weight, then call the nn.functional function with bias=None

    # Example 1: conv_transpose
    conv_trans_weight = self.conv_transpose.weight
    conv_trans_bias = self.conv_transpose.bias
    x = nn.functional.conv_transpose2d(x, conv_trans_weight, None, stride=self.conv_transpose.stride, padding=self.conv_transpose.padding, output_padding=self.conv_transpose.output_padding)
    ... fuse conv_trans_bias with other ops in triton kernel ...

    # Example 2: gemm
    gemm_weight = self.matmul.weight
    gemm_bias = self.matmul.bias
    x = nn.functional.linear(x, gemm_weight, None)
    ... fuse gemm_bias with other ops in triton kernel ...

    # Example 3: conv
    conv_weight = self.conv.weight
    conv_bias = self.conv.bias
    x = nn.functional.conv2d(x, conv_weight, None, self.conv.stride, self.conv.padding, self.conv.dilation)
    ... fuse conv_bias with other ops in triton kernel ...
```

2. Fuse the gemm_bias or conv_bias with the subsequent selected operators, and write the fused kernel code. From the perspective of the tile, check the correctness of the calculation.

3. Output architecture ModelNew. Ensure that the computational accuracy and results after fusion are consistent with the given architecture.
```
# architecture ModelNew 
class ModelNew(nn.Module):
    def __init__(self, ...):   # Keep the same function signature as in Model
        ...
    
    def forward(self, ...):    # Keep the same function signature as in Model
        ...
```

Here are some examples and analyses of bias fusion. 

Example 1 - conv_transpose:

Given architecture:
```
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.bias
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x / self.scaling_factor
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x / self.scaling_factor
        return x
```
Select the operators for fusion: conv_transpose, bias, clamp, div, clamp and div.

First, rewrite conv_transpose using nn.functional with explicit conv_weight and conv_bias.

Second, fuse conv_bias with subsequent element-wise operators directly in the kernel. **Never write Triton code for matmul/conv/conv_transpose**. These must use PyTorch's `nn.functional`.
```
@triton.jit
def fused_op_kernel(
    x_ptr, conv_bias_ptr, bias_ptr, o_ptr,
    n_elements,
    C, H, W,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < n_elements
    
    x = tl.load(x_ptr + off, mask=mask)
    
    hw = H * W
    c_idx = (off // hw) % C
    bias = tl.load(bias_ptr + c_idx)
    conv_bias = tl.load(conv_bias_ptr + c_idx)
    x += conv_bias

    # Fused operators
    x += bias   # fuse bias
    x = tl.minimum(tl.maximum(x, 0.0), 1.0) # fuse clamp
    x = x / scaling_factor  # fuse div
    x = tl.minimum(tl.maximum(x, 0.0), 1.0) # fuse clamp
    x = x / scaling_factor  # fuse div
    
    tl.store(o_ptr + off, x, mask=mask)

def fused_ops(x: torch.Tensor, conv_bias, bias: torch.Tensor, scaling_factor: float):
    output = torch.empty_like(x)
    N = x.numel()
    C, H, W = x.shape[1], x.shape[2], x.shape[3]
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fused_op_kernel[grid](x, conv_bias, bias, output, N, C, H, W, scaling_factor, BLOCK_SIZE)
    return output
```

Finally, validate against the original architecture and output the code (output as ModelNew).
```
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        # Note: Maintain the original implementation
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        weight = self.conv_transpose.weight
        conv_bias = self.conv_transpose.bias
        x = F.conv_transpose2d(x, weight, None, stride=self.conv_transpose.stride, padding=self.conv_transpose.padding, output_padding=self.conv_transpose.output_padding)
        # fuse conv_bias with other selected ops
        x = fused_ops(x, conv_bias, self.bias, self.scaling_factor)
        return x
```

Example 2 - gemm:

Given architecture:
```
class Model(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape)) 

    def forward(self, x):
        x = self.gemm(x)
        x = x + self.add_value
        x = torch.relu(x)
        x = torch.nn.functional.leaky_relu(x)
        x = torch.nn.functional.gelu(x)
        x = torch.max(x, dim=1, keepdim=True)
        return x
```
Select the operators for fusion: gemm_bias, add, relu, leakyrelu, gelu and max.

First, rewrite gemm using nn.functional with explicit gemm_bias and gemm_weight.
```
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape)) 

    def forward(self, x):
        w = self.matmul.weight
        gemm_bias = self.matmul.bias
        x = nn.functional.linear(x, w, None)
        # only fuse the gemm_bias
        x = fuse_ops_triton(x, gemm_bias, self.add_value)
        return x
```

Second, fuse gemm_bias with subsequent element-wise operators directly in the kernel. **Never write Triton code for matmul/conv/conv_transpose**. These must use PyTorch's `nn.functional`.
```
@triton.jit
def fuse_ops_kernel(
    a_ptr, c_ptr, gemm_bias_ptr, v_add_ptr,
    M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + offs_m[:, None] * N + offs_n[None, :]
    inp = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    gemm_bias = tl.load(gemm_bias_ptr + bias_offs, mask=bias_offs < N, other=0.0)
    bias2 = tl.load(v_add_ptr + bias_offs, mask=bias_offs < N, other=0.0)
    inp += gemm_bias[None, :]
    inp += bias2[None, :]

    # fuse: relu + leakyrelu = relu
    inp = tl.where(inp > 0.0, inp, 0.0)

    # fuse gelu
    sqrt_2_over_pi = 0.7978845608028654
    tanh_in = sqrt_2_over_pi * (inp + 0.044715 * (inp * inp * inp))
    exp2x = tl.exp(2.0 * tanh_in)
    tanh_x = (exp2x - 1.0) / (exp2x + 1.0)  # Explicitly write out the tanh formula
    inp = inp * 0.5 * (1.0 + tanh_x)

    # fuse max, use max reduction within tile
    inp = tl.max(inp, axis=1)
    # NOTE: use tl.atomic_max to slove cross-tile reductions
    tl.atomic_max(c_ptr + offs_m, inp)

def fuse_ops_triton(x, gemm_bias, v_add):
    M, N = x.shape
    # Match the result of the final fused max operator
    c = torch.full((M, 1), fill_value=float("-inf") ,device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    # NOTE：This grid partitioning method will lead to cross-tile reductions. 
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    # only fuse the bias term of gemm
    fuse_ops_kernel[grid](
        x, c, gemm_bias, v_add,
        M, N, 
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_stages=5,
    )

    return c
```

Finally, validate against the original process and output the code (output as ModelNew)
