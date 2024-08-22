from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)

import torch
import torch.nn as nn
    
# Adapted from https://github.com/mitsuba-renderer/mitsuba3/discussions/579#discussioncomment-5652452
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

class Linear:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        y = matmul(self.weight, x) + self.bias
        return y

class PReLU:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x):
        y = dr.maximum(0.0, x) + self.weight * dr.minimum(0.0, x)
        return y
    
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

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return self.activation(x)
    
    def test(self, samples=42900):
        nn_in = torch.randn([samples, self.layers[0].weight.shape[1]], device=device)
        
        torch_out = self.torch_model(nn_in)
        drjit_out = self.forward(mi.TensorXf(nn_in.cpu().numpy().T)).torch().T

        assert torch.allclose(torch_out, drjit_out, rtol=1e-03, atol=1e-03), f'{torch_out}\n{drjit_out}'
    


