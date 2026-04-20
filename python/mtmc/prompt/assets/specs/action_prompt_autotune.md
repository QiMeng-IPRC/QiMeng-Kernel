The following will illustrate how to add autotune config to Triton code for performance optimization.

Key Requirements:
1. Define tunable parameters (e.g., BLOCK_SIZE, num_warps, num_stages) with valid ranges.
2. Adjust config parameters using `triton.Config` based on input dimensions and tiling approach.
3. Modify the grid calculation method


Example 1 - GEMM:
Give GEMM solution:
```
@triton.jit
def triton_kernel(
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    # ...
```

Step 1: Identify Tunable Parameters
- Required Parameters: 
  - `tl.constexpr` variables (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K， GROUP_SIZE_M)
  - `num_warps` (4, 8, 16 or 32; default=4)
  - `num_stages` (2–6; default=2)
- Constraints: 
  - `kwargs` values must be multiples of powers of 2.  
  - Avoid `num_stages > 6` to prevent pipeline overhead.

Step 2: Adjust config parameters
If parallelism is low, consider increasing num_warps or decreasing the block size used to compute the grid.

Below are some reference configurations for GEMM autotuning. Ensure that the number of autotune configs for a single kernel does not exceed 6.
```
# Square input
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),

# large k
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        
# small k
triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4)

# tiny m, n, k
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 2}, num_stages=4, num_warps=4)
```

Step 3: Modify the grid calculation method
```
@triton.autotune(
    configs=[  
        # Config 1: Small block size (for small input shapes)  
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4},  
            num_stages=3,  
            num_warps=4  
        ),  
        # Config 2: Large block size (for large input shapes)  
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},  
            num_stages=4,  
            num_warps=8  
        ),  
        # Add up to 6 configurations  
    ],  
    key=['M', 'N', 'K'],  # Tuning keys (input parameters to differentiate configs)  
)  
@triton.jit  
def optimized_kernel(...):  
    # Kernel logic using configured parameters  
    ...

# call kernel
def matmul(a, b):
    # NOTE: should use `lambda META` to set grid
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
```

Example 2 - BatchNorm:
Give BatchNorm solution:
```
@triton.jit
def batch_norm_forward_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    inv_std_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    output_batch_stride,
    output_feat_stride,
    output_spatial_stride,
    momentum,
    eps,
    fuse_bias_add_ptr,
    fuse_divide_value,
    affine: tl.constexpr,     # bool, don't tune
    save_stats: tl.constexpr, # bool, don't tune
    is_train: tl.constexpr,   # bool, don't tune
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # ...
```

Step 1: Identify Tunable Parameters
- Required Parameters: 
  - `tl.constexpr` variables (BLOCK_M, BLOCK_N)
  - `num_warps` (4, 8, 16 or 32; default=4)
  - `num_stages` (2–6; default=2)
- Constraints: 
  - `kwargs` values must be multiples of powers of 2.  
  - Avoid `num_stages > 6` to prevent pipeline overhead.

Step 2: Adjust config parameters
Input Shape: (B, C, L)

Below are common BatchNorm tuning parameters organized in autotune config format. Ensure that the number of autotune configs for a single kernel does not exceed 6.

```
# for large L (more than 16k)
BLOCK_M=2, BLOCK_M=2048, num_warps=16
BLOCK_M=8, BLOCK_M=2048, num_warps=8

# for middle L
BLOCK_M=8, BLOCK_M=1024, num_warps=16

# for small L
BLOCK_M=8, BLOCK_M=16, num_warps=4
```

Step 3: Modify the grid calculation method - use `lambda META` to set grid