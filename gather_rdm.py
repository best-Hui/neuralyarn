# python gather_rdm.py --checkpoint_name fleece --yarn_name fleece

import numpy as np
import argparse # 用于解析命令行参数的一个模块
import os

from config.parameters import get_fiber_parameters
from utils.rdm import compute_rdm




parser = argparse.ArgumentParser(description="Gather RDM for a yarn.") # 创建 ArgumentParser 对象

# 这是一个可选参数，用户可以通过 --checkpoint_name 指定一个检查点名称，用于存储输出结果
parser.add_argument("--checkpoint_name", help="Checkpoint name to store outputs")

# 用户可以通过 --yarn_name 指定一个 yarn 的名称，该名称应该在 config/parameters.py 文件中定义
parser.add_argument("--yarn_name", help="Name of the yarn defined in config/parameters.py")

# 用户可以通过 --batch_size 指定每个批次的大小（即每个批次中包含的样本数）
parser.add_argument("--batch_size", type=int, help='Samples per batch for RDM collection', default=1024*8)

# --num_batches 指定收集 RDM 时的批次数量
parser.add_argument("--num_batches", type=int, help='Number of batches for RDM collection', default=1024)

# 指定 RDM 的 theta 方向的bin 的数量
parser.add_argument("--theta_bins", type=int, help='Theta resolution of the RDM', default=18)

# --phi_bins 指定 RDM 的 phi 方向的bin 的数量
parser.add_argument("--phi_bins", type=int, help='Phi resolution of the RDM', default=36)

args = parser.parse_args()

# 根据用户输入的 checkpoint_name，构造输出目录路径。路径格式为 ./checkpoints/<checkpoint_name>。
output_dir = os.path.join('./checkpoints', args.checkpoint_name)

os.makedirs(output_dir, exist_ok=True)

# 读取一个包含布料纤维参数的数据
parameters = get_fiber_parameters(args.yarn_name)

# 计算 RDM，\ 是一个续行符，用于将一行代码分成多行书写，以提高代码的可读性
# 得到的 透射、反射、多重散射的RDM 数据、相关的计数或统计数据、bin 中心坐标和立体角分布
rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa \
    = compute_rdm(parameters, theta_bins=args.theta_bins, phi_bins=args.theta_bins, num_batches=args.num_batches, batch_size=args.batch_size)

# 转换为 NumPy 数组，使用列表推导式对每个变量调用 .numpy() 方法
# rdm这里应该是透射率的相关统计
rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa = [x.numpy() for x in [rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa]]

# 保存计算结果到./checkpoints/<checkpoint_name>
print('Saving to', output_dir)

# 将计算结果保存为一个 .npz 文件。.npz 是 NumPy 的多数组存储格式
np.savez(os.path.join(output_dir, 'rdm.npz'), rdm_t=rdm_t, rdm_r=rdm_r, rdm_m=rdm_m, 
         count_t=count_t, count_r=count_r, count_m=count_m, 
         x=x, sa=sa)

print('Done')

