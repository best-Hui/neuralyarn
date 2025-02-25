from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)

import torch
import torch.nn as nn



# 这段代码定义了一个将 PyTorch 模型 转换为 Mitsuba （基于 drjit）兼容格式的类，
# 并且对不同类型的网络层进行适配，使其能够在 drjit 中执行。这是为了将 PyTorch 中训练好的模型转换为一个可以在 Mitsuba 渲染器中执行的模型。


# 用于矩阵乘法的实现（类似于 torch.matmul），但使用 drjit 实现，适用于 GPU 加速。
# 它采用了 dr.gather 和 dr.scatter_reduce 等 drjit 操作来进行矩阵的逐元素相乘和加法操作
def matmul(a: mi.TensorXf, b: mi.TensorXf) -> mi.TensorXf:
    # Check conditions
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[0]

    # Matrix sizes
    N = a.shape[0]
    M = b.shape[1]
    K = b.shape[0]

    # Indices of the final matrix c repeat K times
    i, j = dr.arange(mi.UInt, N), dr.arange(mi.UInt, M)
    i, j = dr.meshgrid(i, j, indexing='ij')
    i, j = dr.repeat(i, K), dr.repeat(j, K)

    # [0, 1, ..., K - 1] repeated N * M times
    offset = dr.tile(dr.arange(mi.UInt, K), N * M)

    # Compute [a[0][0] * b[0][0], a[0][1] * b[1][0], ... ]
    tmp = dr.gather(mi.Float, a.array, i * K + offset) * dr.gather(mi.Float, b.array, offset * M + j)

    # Compute [c[0][0], c[0][1], ... ]
    c = dr.zeros(mi.TensorXf, shape=(N, M))
    dr.scatter_reduce(dr.ReduceOp.Add, c.array, tmp, dr.repeat(dr.arange(mi.UInt, N * M), K))
    return c


# 模拟了一个全连接层（线性层），该类包含了权重和偏置的属性，并且实现了 forward 方法来执行矩阵乘法和加法操作
class Linear:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        y = matmul(self.weight, x) + self.bias
        return y


# 模拟了 PReLU（Parametric ReLU）激活函数，其中包含一个可训练的参数 weight，并实现了 forward 方法来计算激活值
class PReLU:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x):
        y = dr.maximum(0.0, x) + self.weight * dr.minimum(0.0, x)
        return y


# 这是主要的包装类，它将一个训练好的 PyTorch 模型转换为 drjit 兼容的模型。
# 它首先遍历 PyTorch 模型的各个层，检查每一层的类型，然后根据类型进行相应的转换：
# 对于线性层（torch.nn.Linear），提取权重和偏置，并将其转换为 mi.TensorXf 格式，以便在 Mitsuba 渲染中使用。
# 对于 PReLU 激活层（torch.nn.PReLU），提取激活权重并转换为 mi.TensorXf。
# 其他层类型会抛出 NotImplementedError。
# test 方法用来验证转换后的 drjit 模型输出与 PyTorch 模型输出是否一致
class MiModelWrapper():
    def __init__(self, torch_model, activation):
        self.torch_model = torch_model

        self.activation = activation
        self.layers = []

        for layer in torch_model.sequential:
            if type(layer) == torch.nn.modules.linear.Linear:
                state_dict = layer.state_dict()
                weight, bias = state_dict['weight'], state_dict['bias']
                weight, bias = mi.TensorXf(weight.cpu().numpy()), mi.TensorXf(bias.unsqueeze(1).cpu().numpy())
                self.layers.append(Linear(weight, bias))

            elif type(layer) == torch.nn.modules.activation.PReLU:
                weight = layer.state_dict()['weight']
                weight = mi.TensorXf(weight.unsqueeze(1).cpu().numpy())
                self.layers.append(PReLU(weight))

            else:
                raise NotImplementedError(f"Layer type not supported: {type(layer)}")
            
        self.test()

    # 实现了将输入数据通过模型的各层进行前向传播。
    # 在 MiModelWrapper 中，数据通过 Linear 层和 PReLU 层依次传递，最后通过指定的激活函数（通常是 ReLU 或类似的函数）进行处理
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return self.activation(x)
    
    def test(self, samples=42900):
        nn_in = torch.randn([samples, self.layers[0].weight.shape[1]], device=device)
        
        torch_out = self.torch_model(nn_in)
        drjit_out = self.forward(mi.TensorXf(nn_in.cpu().numpy().T)).torch().T

        assert torch.allclose(torch_out, drjit_out, rtol=1e-03, atol=1e-03), f'{torch_out}\n{drjit_out}'
    


