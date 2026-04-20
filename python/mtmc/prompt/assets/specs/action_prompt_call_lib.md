When you convert the selected operator to library functions, please adhere to the following conversion criteria:
1. You only need to replace the selected operator with library functions (nn.functional and torch interfaces)
2. Prioritize using nn.functional. If a specific operator is not supported by nn.functional, consider using torch or other library functions instead
3. If the selected operator is already an implementation of library functions, please leave it as it is
4. Please use library functions within nn.Module, and do not use it in other ordinary functions or Triton wrapper functions
5. Name your output architecture ModelNew

Here are some conversion examples and steps
Example 1: conv
```
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
```
First, selected operator is nn.Conv3d, can use nn.functional
Second, since the weight and bias won't be explicitly given, when converting operator such as nn.Conv3d or nn.Linear to calls under nn.functional, you need to retain the original nn operator in the __init__ method to obtain the same weight and bias
Finally, Name output with architecture "ModelNew"
```
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        # Maintain the original implementation
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # Explicitly write out the weight and bias for conv3d
        w = self.conv.weight
        b = self.conv.bias
        x = nn.functional.conv3d(x, w, b, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
```

Example 2: conv_transpose
```
class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor):
        return self.conv_transpose3d(x)
```
Since the weights and biases aren't explicitly provided, when converting operator like conv_transpose to nn.functional calls, you should keep the original nn operator in the __init__ method to ensure you get the same weights and biases. The parameters of __init__ should be consistent with the original ones.

Name output with architecture ModelNew:
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Maintain the original implementation
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicitly write out the weight and bias for conv_transpose
        w = self.conv_transpose3d.weight
        b = self.conv_transpose3d.bias
        return F.conv_transpose3d(x, w, b, self.conv_transpose3d.stride, self.conv_transpose3d.padding, self.conv_transpose3d.output_padding, self.conv_transpose3d.groups, self.conv_transpose3d.dilation)
```

Example 3: torch.matmul
```
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)
```

Since the API for matrix multiplication in nn.functional is linear. However, nn.functional.linear performs a linear transformation of the form y=xA^T+b. Therefore, when using it, you need to transpose the matrix and set bias=None.

Finally, name output with architecture "ModelNew"
```
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # there is no bias in the torch.matmul operator
        return nn.functional.linear(A, B.T, None)
```

Example 4：Leave selected operator as it is
select linear operator, 'nn.functional.linear(A, B.T, None)' -> 'nn.functional.linear(A, B.T, None)', Don't change it.
select bmm in torch: 'torch.bmm(A, B)' -> 'torch.bmm(A, B)', Don't change it.
select mul in torch: 'x = A * s' -> 'x = A * s', Don't change it.

Example 5 - Multiple operators on a single line: 
Select operator: linear, relu
The corresponding line of code is: x = F.relu(self.fc1(x))

You should split different operators into separate lines, that is, one line represents one computation.
```
# Explicitly write out the weights and bias of the linear layer
linear_weight = self.fc1.weight
linear_bias = self.fc1.bias
x = F.linear(x, linear_weight, linear_bias)  # linear
x = F.relu(x)    # ReLu
```
