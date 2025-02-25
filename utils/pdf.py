from utils.np import to_marschner, uct_gaussian_pdf, from_spherical

import numpy as np
import scipy as sp

# 计算 kappa_R（反射比例）
def compute_kappa_R(rdm_t, rdm_r, rdm_m, sa):
    # energy_t：透射部分的加权能量
    # energy_r：反射部分的加权能量
    # energy_m：多次散射部分的加权能量
    energy_t = np.sum(rdm_t.mean(axis=-1) * sa[None, None, :, :])
    energy_r = np.sum(rdm_r.mean(axis=-1) * sa[None, None, :, :])
    energy_m = np.sum(rdm_m.mean(axis=-1) * sa[None, None, :, :])

    kappa_R = energy_r / (energy_t + energy_r + energy_m)
    return kappa_R

def prepare_data(S, x, sa, parameters):
    # 对数据进行归一化，确保 S 的总能量归一化到 1，避免数值不稳定。
    S /= np.sum(S * sa, axis=(2, 3))[:, :, None, None]

    # x 是一个包含入射角、出射角数据的数组，transpose(4, 0, 1, 2, 3) 重新排列维度，使得：
    # THT_I（入射天顶角）
    # PHI_I（入射方位角）
    # THT_O（出射天顶角）
    # PHI_O（出射方位角）
    THT_I, PHI_I, THT_O, PHI_O = x.transpose(4, 0, 1, 2, 3)

    # Convert to cartesian
    # from_spherical() 将极坐标 (θ, φ) 转换为笛卡尔坐标 (x, y, z)。
    # WI 是 入射方向向量，WO 是 出射方向向量。
    WI = np.stack(from_spherical(THT_I, PHI_I), axis=-1)
    WO = np.stack(from_spherical(THT_O, PHI_O), axis=-1)

    # Create fiber frame
    # 创建旋转矩阵：
    # parameters['Alpha'] 表示旋转角度的参数，angle = -π * Alpha 计算旋转弧度。
    # 旋转矩阵 旋转 Alpha 角，使得数据转换到 fiber frame。
    angle = -np.pi * parameters['Alpha']
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), +np.cos(angle), 0.0],
        [0.0,           0.0,            1.0]
    ])

    # 旋转基向量：
    # frame_x：沿 X 轴旋转
    # frame_y：沿 Y 轴旋转
    # frame_z：沿 Z 轴旋转
    frame_x = rotation @ [1.0, 0.0, 0.0]
    frame_y = rotation @ [0.0, 1.0, 0.0]
    frame_z = rotation @ [0.0, 0.0, 1.0]

    # Convert to fiber frame
    # 应用旋转矩阵：
    # 将 入射向量 WI 和 出射向量 WO 旋转到纤维参考系。
    WI[:, :, :, :, 0] = np.dot(WI, frame_x)
    WI[:, :, :, :, 1] = np.dot(WI, frame_y)
    WI[:, :, :, :, 2] = np.dot(WI, frame_z)
    WO[:, :, :, :, 0] = np.dot(WO, frame_x)
    WO[:, :, :, :, 1] = np.dot(WO, frame_y)
    WO[:, :, :, :, 2] = np.dot(WO, frame_z)

    # Convert to Marschner
    # Marschner 模型转换：
    # to_marschner() 将旋转后的向量转换到 Marschner 坐标系，用于纤维光照建模
    THT_I, PHI_I = to_marschner(*np.transpose(WI, [4, 0, 1, 2, 3]))
    THT_O, PHI_O = to_marschner(*np.transpose(WO, [4, 0, 1, 2, 3]))

    return (THT_I, PHI_I, THT_O, PHI_O), S


# 定义了一个 概率密度函数 (PDF)，用于拟合光照中的散射分布。它的输入参数包括：
# X：包含角度信息 (θ_i, φ_i, θ_o, φ_o)，分别代表入射和出射的天顶角 (θ) 和方位角 (φ)。
# p：控制权重的参数，表示该 PDF 的混合比例。
# p、beta 和 gamma 不是直接传入的输入数据，而是通过 curve_fit 进行拟合的 参数
def pdf(X, p, beta, gamma):
    tht_i, phi_i, tht_o, phi_o = X

    M = uct_gaussian_pdf(tht_o, -tht_i, beta, -np.pi/2, +np.pi/2)
    N = uct_gaussian_pdf(phi_o, 0.0, gamma, -np.pi, +np.pi)
    # 论文中pdf_m的公式定义,p实际上是kappa_M
    return p * M * N / np.cos(tht_o) + (1-p) * (1/(4*np.pi))


# 这个函数用于 拟合 PDF 以找到最优参数（p, beta, gamma）。
#
# input：将输入数据 展平 (ravel)，确保其形状符合 curve_fit 需求。
# label：标签数据，同样展平成 1D 数组。
# sp.optimize.curve_fit()：
# pdf：目标拟合的函数。
# p0=[0.5, 1, 1]：初始猜测值 (p=0.5, beta=1, gamma=1)。
# bounds=[(0, 0, 0), (1, 5, 10)]：设定参数范围：
# p 在 [0, 1] 之间。
# beta 在 [0, 5] 之间（较小的 beta 表示更集中的分布）。
# gamma 在 [0, 10] 之间（控制 φ_o 的扩散程度）。
# 返回值 popt：最优拟合参数 (p, beta, gamma)。
def fit_pdf_m(input, label):
    # input :THT_I, PHI_I, THT_O, PHI_O
    # label :rdm_m
    input = [data.ravel() for data in input]
    label = label.ravel()

    # 将input传入pdf,
    # curve_fit 会自动调用 pdf 函数，并通过传入的 input（四个角度数据）和初始参数 p0 来计算 pdf 的输出。
    # pdf 输出的值将与目标数据 label（rdm_m）进行比较。
    # curve_fit 调整 p, beta, 和 gamma，直到最小化输出与目标之间的差异。
    # 最终，popt 返回最优的拟合参数 [p, beta, gamma]，这些参数可以用于进一步的计算。
    # p、beta 和 gamma 不是直接传入的输入数据，而是通过 curve_fit 进行拟合的 参数
    popt, pcov = sp.optimize.curve_fit(pdf, input, label, p0=[0.5, 1, 1], bounds=[(0, 0, 0), (1, 5, 10)])

    return popt


# 该函数的目的是 计算 PDF 的所有参数，包括 kappa_R 和 kappa_M
# 步骤 1：先用 fit_pdf_m() 拟合 rdm_m，得到 beta_M、gamma_M、kappa_M。
# 步骤 2：计算 kappa_R 反射比例。
# 步骤 3：返回 (kappa_R, beta_M, gamma_M, kappa_M)，用于进一步的光照计算。
def fit_pdf(rdm_t, rdm_r, rdm_m, x, sa, parameters):
    # prepare_data: return (THT_I, PHI_I, THT_O, PHI_O), S .s是rdm_m
    kappa_M, beta_M, gamma_M = fit_pdf_m(*prepare_data(rdm_m.mean(axis=-1), x, sa, parameters))
    kappa_R = compute_kappa_R(rdm_t, rdm_r, rdm_m, sa)

    return kappa_R, beta_M, gamma_M, kappa_M