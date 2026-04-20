The general process of optimizing operators with Triton is as follows:

1. Analysis of the original architecture: clarify the calculation process and input shape.
2. Design of the tiling strategy: analyze the data flow from the perspective of tiles and formulate a tiling strategy based on the analysis results.
3. Kernel implementation: write kernel code based on the tiling method and check for data dependencies within tile.

Below are some examples demonstrating how to optimize operators using Triton.

Example 1 - Optimizing Softmax: 

1.1 Selection of torch.softmax(x, dim=1) with input dimensions (batch=32, elem=8192), reduction axis=1, and output dimensions (batch=32, elem=8192)

Softmax computation involves data dependencies and can be divided into three stages: max, expsum, norm.

For each batch, expsum depends on the reduction result of max, and norm depends on the results of both max and expsum.

1.2 Tile strategy

Tiling: There are no data dependencies between batches, but reduction operations are involved within each batch. Tiling is performed along the batch dimension.

In-tile: The data dependency chain is max → expsum → norm. The max operator is a reduction. Use a loop within the tile to compute each step sequentially.

Wait, the number of elements for reduction (elem=8192) is less than 16384, the entire batch of data can be loaded in one go, meaning the loop runs exactly once

Therefore, the tile strategy is:
```
TILE_SIZE = triton.next_power_of_2(n)
grid = (batch, )
```

1.3 Kernel implementation
```
@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, 
                   output_row_stride, n_cols, TILE_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    row_start_ptr = input_ptr + row_id * input_row_stride
    
    col_offsets = tl.arange(0, TILE_SIZE)
    input_ptr = row_start_ptr + col_offsets

    data = tl.load(input_ptr, mask=col_offsets < n_cols, other=0.0)
    # Loop once and omit the loop statement.
    data_minus_max = data - tl.max(data, axis=0)
    data_exp = tl.exp(data_minus_max)
    data_sum = tl.sum(data_exp, axis=0)
    output = data_exp / data_sum
    output_start_ptr = output_ptr + row_id * output_row_stride
    
    tl.store(output_start_ptr + col_offsets, output, mask=(col_offsets < n_cols))
                
def softmax(x):
    m,n = x.shape
    output=torch.empty_like(x)
    # Load the entire batch; otherwise, explicitly write loops within kernel.
    TILE_SIZE = triton.next_power_of_2(n)
    if TILE_SIZE >= 2048: num_warps = 8
    if TILE_SIZE >= 4096: num_warps = 16
    # Only the kernel with dim = 1 needs to be written.
    softmax_kernel[(m, )](x, 
        output, 
        x.stride(0), 
        output.stride(0), 
        n, 
        TILE_SIZE = TILE_SIZE
    )
    return output

# Integrate into ModelNew
class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Just implement the case where dim = 1.
        return softmax(x)
```


example 2 - Upper triangular matmul (same applies to lower triangular matmul): 

2.1 Selected torch.triu(torch.matmul(A, B)), input shape (N, N).

2.2 Tile strategy: grid=(BLOCK_M, BLOCK_N), but only the upper half need to be calculated.
```
# returning in advance 
pid_m > pid_n can skip lower triangle blocks
pid_m < pid_n can skip upper triangle blocks
```

2.3 Kernel implementation. 
```
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m > pid_n:   # skip lower triangle blocks
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k) & (offs_m[:, None] < M), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        # set allow_tf32=False for computational precision
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

example 3 - Pooling operator: 

3.1 select AvgPool3d, the input shape is (N, C, D, H, W)

3.2 Tile strategy
```
BLOCK_SIZE = 256  
grid = (triton.cdiv(N * C * D_out * H_out * W_out, BLOCK_SIZE),)
```
3.3 Kernel implementation
``` 
@triton.jit
def avg_pool3d_kernel(
    x_ptr, o_ptr, 
    N, C, D, H, W, 
    D_out, H_out, W_out,
    K, S, P,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    elements = N * C * D_out * H_out * W_out
    mask = offsets < elements
    
    # compute output tensor indices
    n = offsets // (C * D_out * H_out * W_out)
    c = (offsets % (C * D_out * H_out * W_out)) // (D_out * H_out * W_out)
    d_out_idx = (offsets % (D_out * H_out * W_out)) // (H_out * W_out)
    h_out_idx = (offsets % (H_out * W_out)) // W_out
    w_out_idx = offsets % W_out
    
    # calculate starting indices in input tensor
    d_in_start = d_out_idx * S - P
    h_in_start = h_out_idx * S - P
    w_in_start = w_out_idx * S - P
    
    # Initialize sum accumulator for pooling window
    sum_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # NOTE: `tl.full((BLOCK_SIZE,), float('-inf'), tl.float32)` for max_pooling window
    
    # sum of one pooling_kernel
    for kd in range(K):  # Ensure that the loop range is scalar
        d_in = d_in_start + kd
        d_valid = (d_in >= 0) & (d_in < D)
        
        for kh in range(K):
            h_in = h_in_start + kh
            h_valid = (h_in >= 0) & (h_in < H)
            
            for kw in range(K):
                w_in = w_in_start + kw
                w_valid = (w_in >= 0) & (w_in < W)
                
                valid = d_valid & h_valid & w_valid & mask
                
                x_idx = n * (C * D * H * W) + c * (D * H * W) + d_in * (H * W) + h_in * W + w_in
                
                x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
                sum_vals += x_val
    
    # avg of one pooling_kernel
    avg_vals = sum_vals / (K * K * K)
    tl.store(o_ptr + offsets, avg_vals, mask=mask)
```

example 4 - BatchNorm:

4.1 select BatchNorm operator, the input shape is (B, C, H, W).

BatchNorm involves data dependencies and can be divided into three stages: mean, var and norm.

4.2 Tile strategy

In BatchNorm, the mean is reduced along the C dimension, meaning there is no data dependency across different C dimensions.

In-tile: The data dependency chain is mean → var → norm. The mean operator is a reduction. Use a loop within the tile to compute each step sequentially.

Therefore, the tile strategy is:
```
grid = (feat_dim,)
# Loop in tile
BLOCK_M=8,
BLOCK_N=2048,
```

4.3 Kernel implementation
```
def make_3d_for_bn(input: torch.Tensor):
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = input.flatten(2, -1)
    return input

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
    affine: tl.constexpr,
    save_stats: tl.constexpr,
    is_train: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    if is_train:
        mean = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        var = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        cnt = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        m_num_steps = tl.cdiv(batch_dim, BLOCK_M)
        n_num_steps = tl.cdiv(spatial_dim, BLOCK_N)

        for m_step in range(0, m_num_steps):
            for n_step in range(0, n_num_steps):
                spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
                spatial_mask = spatial_offset < spatial_dim

                batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
                batch_mask = batch_offset < batch_dim

                curr_input_ptr = (
                    input_ptr
                    + input_feat_stride * feat_pid
                    + input_batch_stride * batch_offset[:, None]
                    + input_spatial_stride * spatial_offset[None, :]
                )

                mask = batch_mask[:, None] & spatial_mask[None, :]
                curr_input = tl.load(curr_input_ptr, mask=mask).to(tl.float32)

                step = m_step * n_num_steps + n_step + 1
                new_mean = tl.where(mask, mean + (curr_input - mean) / step, mean)
                new_var = tl.where(
                    mask, var + (curr_input - new_mean) * (curr_input - mean), var
                )
                cnt += mask.to(tl.int32)
                mean = new_mean
                var = new_var

        final_mean = tl.sum(mean * cnt) / (batch_dim * spatial_dim)
        var = tl.sum(var + cnt * (mean - final_mean) * (mean - final_mean)) / (
            batch_dim * spatial_dim
        )
        inv_std = tl.rsqrt(var + eps)
        mean = final_mean

        if save_stats:
            tl.store(feat_pid + mean_ptr, mean)
            tl.store(feat_pid + inv_std_ptr, inv_std)

        running_mean_ptr += feat_pid
        running_var_ptr += feat_pid

        running_mean = tl.load(running_mean_ptr)
        running_var = tl.load(running_var_ptr)

        n = batch_dim * spatial_dim
        tl.store(running_mean_ptr, (1 - momentum) * running_mean + momentum * mean)
        tl.store(
            running_var_ptr,
            (1 - momentum) * running_var + momentum * var * n / (n - 1),
        )

    else:
        mean = tl.load(feat_pid + running_mean_ptr)
        inv_std = tl.rsqrt(tl.load(feat_pid + running_var_ptr) + eps)

    if affine:
        weight = tl.load(feat_pid + weight_ptr)
        bias = tl.load(feat_pid + bias_ptr)

    else:
        weight = 1.0
        bias = 0.0

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_input_ptr = (
                input_ptr
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_output_ptr = (
                output_ptr
                + output_feat_stride * feat_pid
                + output_batch_stride * batch_offset[:, None]
                + output_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_ptr, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            output = weight * (curr_input - mean) * inv_std + bias

            tl.store(
                curr_output_ptr,
                output,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        # keep the necessary information
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_3d = make_3d_for_bn(x)
        batch_dim, feat_dim, spatial_dim = input_3d.shape
        output = torch.empty_like(input_3d)

        mean = torch.empty(feat_dim, device=x.device, dtype=x.dtype)
        inv_std = torch.empty(feat_dim, device=x.device, dtype=x.dtype)

        batch_norm_forward_kernel[(feat_dim,)](
            input_3d,
            self.gamma,
            self.beta,
            mean,
            inv_std,
            output,
            self.running_mean,
            self.running_var,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *output.stride(),
            self.momentum,
            self.eps,
            affine=False,       # No affine
            save_stats=False,
            is_train=True,      # Must set is_train=True
            BLOCK_M=8,
            BLOCK_N=2048,
            num_warps=8,
        )

        return output.view_as(x)
```