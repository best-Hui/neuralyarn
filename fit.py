# python fit.py --checkpoint_name fleece --yarn_name fleece
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import argparse
import os
from config import device, variant
import mitsuba as mi
import drjit as dr

mi.set_variant(variant)
from mitsuba import Transform4f as T
from mitsuba import ScalarTransform4f as sT

from config.parameters import get_fiber_parameters

from bsdf.neuralyarn import NeuralYarn

from network.model import Model_M, Model_T
from network.wrapper import MiModelWrapper

from utils.pdf import fit_pdf
from utils.rdm import compute_rdm
from utils.torch import cart2sph
from utils.np import to_marschner, uct_gaussian_pdf, from_spherical
from utils.geometry import create_single_yarn



parser = argparse.ArgumentParser(description="Fitting the RDM")

parser.add_argument("--checkpoint_name", help="Checkpoint name to store outputs")
parser.add_argument("--yarn_name", help="Name of the yarn defined in config/parameters.py")
parser.add_argument("--epochs_m", type=int, help="Number of epochs to train for model_m", default=25000)
parser.add_argument("--epochs_t", type=int,  help="Number of epochs to train for model_t", default=25000)
parser.add_argument("--lr_m", type=float,  help="Learning rate for model_m", default=0.005)
parser.add_argument("--lr_t", type=float, help="Learning rate for model_t", default=0.002)

args = parser.parse_args()

output_dir = os.path.join('checkpoints/', args.checkpoint_name)

# Load rdm data
print('Loading RDM data from', output_dir)
data = np.load(os.path.join(output_dir, 'rdm.npz'))

rdm_t = data['rdm_t']
rdm_r = data['rdm_r']
rdm_m = data['rdm_m']
count_t = data['count_t']
count_r = data['count_r']
count_m = data['count_m']
x = data['x']
sa = data['sa']  # solid_angles

print('Training Model M')

# .to(device)：将模型移动到指定的设备上。
# device 通常是 'cpu' 或 'cuda'（如果你使用的是 GPU）。
# PyTorch 的模型和张量可以被分配到不同的设备（CPU 或 GPU），这是为了利用硬件加速（尤其是在使用 GPU 时）。
# to(device) 使得模型在适当的设备上执行，从而加速训练和推理。
# 如果 device = 'cuda'，则模型会被转移到 GPU 上进行计算（适合加速训练过程）。
# 如果 device = 'cpu'，则模型会在 CPU 上执行。
model_m = Model_M().to(device)

n_epochs = args.epochs_m
print_every = 100

# 使用 Adam 优化算法来优化 model_m 的参数，model_m.parameters() 是指 model_m 模型中的所有可训练参数（包括权重和偏置）。
# 这些参数会在训练过程中更新。
# lr=args.lr_m 是学习率，args.lr_m 来自命令行参数。
# 这是控制模型在每次更新时“步长”的超参数。
# 较小的学习率可以使模型更加稳定，但可能需要更多的训练步骤，而较大的学习率则可能加速收敛，但也可能导致训练不稳定。
optimizer = torch.optim.Adam(model_m.parameters(), lr=args.lr_m)


# 将输入数据 x 转换为 PyTorch 的张量，并对其形状进行调整
nn_in = torch.tensor(x).reshape(-1, 4).float().to(device)

# 将输入张量 nn_in 转换为球面坐标系
nn_in = torch.cat([cart2sph(*nn_in[:, [0, 1]].T), cart2sph(*nn_in[:, [2, 3]].T)], dim=-1).float().to(device)

# 将真实的 RDM 数据（rdm_m）转换为张量，并对其形状进行调整,实际上是rgb值(辐照度)
nn_true = torch.tensor(rdm_m).reshape(-1, 3).float().to(device)

# 定义了损失函数：mse
criterion = nn.MSELoss()

# 记录训练过程中的损失值
history = []

# 启动一个 训练循环，该循环会重复 n_epochs 次，每次更新模型参数
for epoch in range(n_epochs):
    # Zero the gradients
    # 清除上一轮迭代中的 梯度信息
    # 在 PyTorch 中，梯度是累加的，即每次 backward() 被调用时，梯度会累积在模型的参数中。
    # 因此，在每一轮训练开始之前，必须清空之前的梯度，以确保新计算的梯度不会与旧的梯度混合。
    # optimizer.zero_grad() 会把所有模型参数的梯度清零，准备进行新的反向传播。
    optimizer.zero_grad()


    # 这行代码使用训练数据 nn_in 通过模型 model_m 进行 前向传播。
    # nn_in 是输入数据，通常是特征向量，包含了输入样本的所有特征。
    # model_m(nn_in) 将输入数据传入神经网络模型 model_m，得到模型的预测结果 nn_pred。这一步是神经网络的前向计算过程。
    nn_pred = model_m(nn_in)

    # 计算 损失函数 的值
    loss = criterion(nn_pred, nn_true)

    # 反向传播，计算损失函数关于模型参数的梯度
    loss.backward()

    # optimizer.step() 会使用计算出的梯度，按照优化算法（在这里是 Adam）来更新模型的参数（如权重和偏置）。
    optimizer.step()

    # 每隔一段时间打印一次训练信息
    if (epoch + 1) % print_every == 0:
        # 将当前的 损失值 记录到 history 列表中
        history.append(loss.item())
        # 每隔一定 epoch 打印一次训练的状态
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.7f}, Log Loss: {np.log(loss.item()):.7f}')

# model_m.eval() 会通知模型进入评估（推理）模式。
# 这样做的原因是某些层（如 Dropout 和 BatchNorm）在训练和推理时的行为不同。
# 在评估模式下，Dropout 层会关闭，而 BatchNorm 层会使用训练过程中计算的均值和方差。
# 通常在训练结束后，调用 eval() 方法，以便进行验证或测试。
model_m.eval()

print('Training Model T')
# 创建 Model_T 的实例
model_t = Model_T().to(device)

# n_epochs = args.epochs_t：获取 epochs_t 参数，即 Model_T 需要训练的总轮数。
# print_every = 100：每隔 100 轮训练打印一次损失值，以便监控训练进度。
n_epochs = args.epochs_t
print_every = 100

# torch.optim.Adam(...)：使用 Adam 优化器进行梯度更新。
# model_t.parameters()：获取 Model_T 的所有可训练参数，以便优化器更新它们。
# lr=args.lr_t：设置学习率（lr），用于控制优化器每次更新参数的步长。
optimizer = torch.optim.Adam(model_t.parameters(), lr=args.lr_t)

nn_in = torch.tensor(x)[:, :, 0, 0, :2].reshape(-1, 2).float().to(device)
nn_in = cart2sph(*nn_in.T).float().to(device)

# 处理真实标签(透射概率)
nn_true = count_t / (count_t + count_r + count_m)
nn_true = torch.tensor(nn_true).reshape(-1, 1).to(device)

criterion = nn.MSELoss()

for epoch in range(n_epochs):
    # Zero the gradients
    optimizer.zero_grad()
    
    nn_pred = model_t(nn_in)
    
    loss = criterion(nn_pred, nn_true)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % print_every == 0:
        history.append(loss.item())
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.7f}, Log Loss: {np.log(loss.item()):.7f}')

model_t.eval()


print('Fitting PDF')

# 拟合 PDF（概率密度函数）
parameters = get_fiber_parameters(args.yarn_name)
kappa_R, beta_M, gamma_M, kappa_M = fit_pdf(rdm_t, rdm_r, rdm_m, x, sa, parameters)
print('Parameters:', kappa_R, beta_M, gamma_M, kappa_M)

print('Saving to', output_dir)
torch.save(model_m.state_dict(), os.path.join(output_dir, 'model_m.pth'))
torch.save(model_t.state_dict(), os.path.join(output_dir, 'model_t.pth'))
np.savez(os.path.join(output_dir, 'pdf.npz'), kappa_R=kappa_R, beta_M=beta_M, gamma_M=gamma_M, kappa_M=kappa_M)
