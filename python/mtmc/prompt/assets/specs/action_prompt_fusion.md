The general steps of fusion are as follows:
1. Analyze the computation process among operators and the data dependencies within operators from a tiling perspective.
2. Fusion strategy: Based on the data analysis under tiling, identify the fusible parts (a single operator may be decomposed into multiple computational parts, e.g. layernorm -> mean, var, norm), and write the fusible parts into one kernel.
3. Output architecture ModelNew. Ensure that the computational accuracy and results after fusion are consistent with the given architecture.

Here are some examples and analyses of fusion.

Example 1 - fuse linear and a reduction operator:

Given architecture:
```
x = self.linear(x)  # (batch_size, out_features)
x = torch.sum(x, dim=1, keepdim=True) # (batch_size, 1)
x = torch.max(x, dim=1, keepdim=True)[0]
x = torch.mean(x, dim=1, keepdim=True)
x = torch.logsumexp(x, dim=1, keepdim=True)
```

Select to fuse linear and sum.

1.1 Analysis of tiling dataflow

Determine the input and output dimensions before and after fusion. Here, the input dimension is (B, N) and the output dimension is (B, 1).

The linear operation computes a BLOCK_M × BLOCK_N block each time. Fusing the sum reduction under the (BLOCK_M, BLOCK_N) tile requires cross-tile operations. Therefore, directly fuse the sum operation within the (BLOCK_M, BLOCK_N) tile.

1.2 Fusion strategy: fuse sum into the end of linear tile computation
```
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M           
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # (BLOCK_M, BLOCK_N)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k)), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, accumulator, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    bias_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_mask = bias_offs < N
    bias = tl.load(bias_ptr + bias_offs, mask=bias_mask, other=0.0)
    bias = bias[None, :]
    c = accumulator.to(tl.float32)
    c += bias 

    # fuse sum operator before writing back
    row_sums = tl.sum(c, axis=1)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + offs_cm
    mask = offs_cm < M

    # use tl.atomic_add to solve the cross-tile reduction problem
    tl.atomic_add(c_ptrs, row_sums, mask=mask)


def matmul_fused_triton(a, b, bias):
    M, K = a.shape
    _, N = b.shape
    c = torch.zeros((M, 1), device=a.device, dtype=a.dtype)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    matmul_kernel[grid](
        a, b, c, bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_K = BLOCK_SIZE_K,
        GROUP_SIZE_M = 8,
    )
    return c
```

1.3 Integration into ModelNew
```
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        w = self.linear.weight.T  # NOTE Here
        b = self.linear.bias
        x = matmul_fused_triton(x, w, b)
        # Output dimension (B, 1)
        # Subsequent operations are meaningless with (B, 1) input, thus omitted
        # x = torch.max(x, dim=1, keepdim=True)[0]
        # x = torch.mean(x, dim=1, keepdim=True)
        # x = torch.logsumexp(x, dim=1, keepdim=True)
        return x
```

Example 2 - fuse batchnorm and element-wise operators:
Given architecture:
```
def forward(self, x):
    x = self.matmul(x)
    x = self.bn(x)      # nn.BatchNorm1d
    x = x + self.bias
    x = x / self.divide_value
    x = x * torch.sigmoid(x)
    return x
```

Select to fuse batchnorm1d, add, div, swish(mul, sigmoid)
2.1 Analysis of tiling dataflow
BatchNorm involves data dependencies and can be divided into three stages: mean, var and norm.
Since all subsequent operators are element-wise, they can be directly fused after batch normalization

2.2 Fusion strategy: fuse before writing back
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
    fuse_bias_add_ptr,
    fuse_divide_value,
    affine: tl.constexpr,
    save_stats: tl.constexpr,
    is_train: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    # traning mode default track_running_stat
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
            
            # fuse before writing back
            fuse_bias_add = tl.load(fuse_bias_add_ptr)  
            output += fuse_bias_add                 # fuse bias_add, shape of bias is (1,)
            output /= fuse_divide_value             # fuse div
            output = output / (1 + tl.exp(-output)) # fuse swish

            # write back
            tl.store(
                curr_output_ptr,
                output,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


def batch_norm_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                       running_mean: torch.Tensor, running_var: torch.Tensor,
                       momentum: float, eps: float, affine: bool,
                       save_stats: bool, is_train: bool,
                       fuse_bias_add, fuse_divide_value):
    inp = make_3d_for_bn(x)     # use new variables
    batch_dim, feat_dim, spatial_dim = inp.shape
    output = torch.empty_like(inp)

    mean = torch.empty(feat_dim, device=x.device, dtype=x.dtype)
    inv_std = torch.empty(feat_dim, device=x.device, dtype=x.dtype)

    batch_norm_forward_kernel[(feat_dim,)](
        inp,
        weight,
        bias,
        mean,
        inv_std,
        output,
        running_mean,
        running_var,
        batch_dim,
        spatial_dim,
        *inp.stride(),
        *output.stride(),
        momentum=momentum,
        eps=eps,
        fuse_bias_add_ptr=fuse_bias_add,
        fuse_divide_value=fuse_divide_value,
        affine=affine,
        save_stats=save_stats,
        is_train=is_train,
        BLOCK_M=8,
        BLOCK_N=16,
    )

    return output.view_as(x)
```

3.3 Integration into ModelNew
```
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        # keep raw batchnorm info
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.momentum = bn_momentum
        self.eps = bn_eps
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.divide_value = divide_value

    def forward(self, x):
        x = self.matmul(x)
        x = batch_norm_forward(
            x,
            None,
            None,
            self.running_mean,
            self.running_var,
            self.momentum,
            self.eps,
            affine=False,
            save_stats=False,
            is_train=True,      # must set is_train=True
            # the input of fused elements
            fuse_bias_add=self.bias,
            fuse_divide_value=self.divide_value
        )
        return x
```