from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)
import time
import torch

import numpy as np
from mitsuba import ScalarTransform4f as sT

from utils.geometry import create_single_yarn

# 计算单位球面上某个角度网格（由 n 和 m 给定）的每个小单元的固体角
# n：表示极角（θ，theta）的分段数，即球面在极角方向的分割数量。
# m：表示方位角（φ，phi）的分段数，即球面在方位角方向的分割数量。
def calculate_solid_angle_grid(n, m):
    # Define the ranges for theta (polar) and phi (azimuthal) angles
    # theta_edges：生成 n+1 个等距的点，范围从 0 到 π，即极角（theta）的边界。theta = 0 对应球面北极，theta = π 对应球面南极。
    # phi_edges：生成 m+1 个等距的点，范围从 0 到 2π，即方位角（phi）的边界。
    # np.linspace 是 NumPy 库中的一个函数，用于生成一个等间隔的数值序列,这里分别是0到pi和0到2pi
    theta_edges = np.linspace(0, np.pi, n + 1)
    phi_edges = np.linspace(0, 2 * np.pi, m + 1)
    
    # Calculate bin centers
    # theta_bins：通过计算 theta_edges 中相邻两个边界的平均值来获取每个角度区间的中心。[:-1] 和 [1:] 是去掉第一个和最后一个边界来计算每两个相邻边界的中点。
    # phi_bins：同样的处理方法，用于计算方位角区间的中心。
    theta_bins = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_bins = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    
    # Calculate the solid angle for each bin
    # 初始化一个 n x m 的零矩阵，用来存储每个角度bin的固体角值
    solid_angle_grid = np.zeros((n, m))


    for i in range(n):
        # 对于每个 i（表示极角的索引），计算极角间隔的余弦差值：delta_cos_theta = np.cos(theta_edges[i]) - np.cos(theta_edges[i+1])。
        # 这代表极角方向上每个角度区间的角度宽度。
        delta_cos_theta = np.cos(theta_edges[i]) - np.cos(theta_edges[i+1])
        for j in range(m):
            # 对于每个 j（表示方位角的索引），计算方位角的区间宽度：delta_phi = phi_edges[j+1] - phi_edges[j]。
            delta_phi = phi_edges[j+1] - phi_edges[j]
            # 固体角值存储在 solid_angle_grid[i, j]
            solid_angle_grid[i, j] = delta_cos_theta * delta_phi

    # solid_angle_grid：一个 n x m 的数组，每个元素表示对应 theta_bins[i] 和 phi_bins[j] 位置的固体角。
    # theta_bins：包含每个极角区间中心的数组。
    # phi_bins：包含每个方位角区间中心的数组。
    return solid_angle_grid, theta_bins, phi_bins


def trace_paths(scene, rays, max_depth=100):
    # n_samples：rays.o 的第二维度的大小，即射线数量。实际上是一批光线
    # sampler：创建一个独立的随机采样器，用于生成随机数。mi.load_dict 加载采样器的配置，'type': 'independent' 表示使用独立的随机数生成器。
    # sampler.seed：为采样器设置一个随机种子，wavefront_size=n_samples 确保采样器的大小与射线数量相同。
    n_samples = dr.shape(rays.o)[1]
    sampler = mi.load_dict({'type' : 'independent'})
    sampler.seed(np.random.randint(np.iinfo(np.int32).max), wavefront_size=n_samples)

    # si = scene.ray_intersect(rays)：计算射线与场景中物体的交点（si 表示交点信息）。si 是一个结构体，包含交点位置、法线、材质等信息。
    # valid = si.is_valid()：检查交点是否有效（即射线是否与物体相交）。
    si = scene.ray_intersect(rays)
    valid = si.is_valid()

    # Randomly sample initial wi
    # wi：为当前交点随机采样一个入射方向。
    # mi.warp.square_to_uniform_hemisphere 根据二维均匀随机数生成一个单位半球上的方向（即从天空采样入射光方向）。
    wi = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())

    # frame_i：基于交点的切线平面创建一个局部坐标系（Frame3f）。它用来将方向向量从世界坐标系转换为交点的局部坐标系。
    frame_i = mi.Frame3f(si.sh_frame)

    # si.wi = wi：将采样到的入射光方向 wi 存储在交点 si 中。
    # throughput = mi.Spectrum(1.0)：初始化透射率为 1（表示路径追踪开始时光强为 1）。
    # active = mi.Bool(valid)：表示光线是否仍然有效的布尔数组，初始值为 valid。
    # depth = mi.UInt32(0) 和 i = mi.UInt32(0)：用于记录当前路径的深度。
    si.wi = wi
    throughput = mi.Spectrum(1.0)
    active = mi.Bool(valid)
    depth = mi.UInt32(0)
    i = mi.UInt32(0)

    # loop：mi.Loop 是一种用于控制路径追踪迭代的机制。它会在路径追踪过程中不断更新各个变量（如 i、throughput、rays 等）
    loop = mi.Loop(name="", state=lambda:(i, sampler, rays, si, throughput, active, depth))

    # 这部分代码执行路径追踪的主循环，loop(active & (i < max_depth)) 确保只有有效的光线且深度未超过 max_depth 时才继续追踪。
    while loop(active & (i < max_depth)):
        # depth[active] += mi.UInt32(1)：每次路径迭代时，深度加 1。
        depth[active] += mi.UInt32(1)

        # Sample new direction
        # ctx = mi.BSDFContext()：创建一个 BSDF 上下文，用于后续采样。
        # bs, bsdf_val = si.bsdf().sample(ctx, si, sampler.next_1d(), sampler.next_2d(), True)：使用 BSDF（双向散射分布函数）在交点处采样散射方向 wo 和散射值 bsdf_val。
        ctx = mi.BSDFContext()
        bs, bsdf_val = si.bsdf().sample(ctx, si, sampler.next_1d(), sampler.next_2d(), True)

        # throughput[active] *= bsdf_val：更新透射率，乘以散射值（即每次反射/折射时光强衰减）。
        # bsdf_val 是 BSDF（双向散射分布函数）采样结果，表示材质对于特定入射方向的散射强度。
        # 每次光线与物体表面相互作用时，反射或折射 会消耗一部分光能，bsdf_val 就是这部分能量的衰减因子
        # rays[active] = si.spawn_ray(si.to_world(bs.wo))：根据 wo（新的散射方向）生成新的光线。
        throughput[active] *= bsdf_val
        rays[active] = si.spawn_ray(si.to_world(bs.wo))

        # Find next intersection
        # si = scene.ray_intersect(rays, active)：计算新光线与场景的交点。
        # active &= si.is_valid()：更新 active 标志，确保只有与物体相交的光线继续传播。
        si = scene.ray_intersect(rays, active)
        active &= si.is_valid()

        # Increase loop iteration counter
        # 路径深度
        i += 1

    # 多次光线追踪的最后的出射方向
    wo = frame_i.to_local(rays.d)
    
    return wi, wo, throughput, depth, frame_i, si

# Flatten N-dimensional indices to a flat array
# 将多维索引（index）转换为一维索引（ravel_idx）。它的目的是根据给定的形状（shape），将一个多维数组的坐标索引转换成一个对应的一维索引
# 举个例子
# 假设你有一个 3D 数组，其形状为 (3, 4, 5)，即它有 3 个维度，分别有 3、4、5 个元素。如果你传入一个 3D 索引，比如 (2, 1, 3)，这个函数将会将其转换为对应的一维索引。
# 输入
# shape = (3, 4, 5) 表示这个数组的维度。
# index = (2, 1, 3) 表示多维数组中的索引位置。
# 最终的 ravel_idx 数组是 [48, 48, 48]，也就是对应的线性索引。
def ravel_index(index, shape):
    ravel_idx = dr.zeros(mi.UInt32, dr.shape(index[0])[0])
    step = 1
    for i in range(len(shape) - 1, 0 - 1, -1):
        ravel_idx += index[i] * step
        step *= shape[i]

    return ravel_idx

# 计算一个4维直方图，它涉及到角度空间（如极角和方位角）的离散化，并通过这些角度的组合来统计一个量（如透射率或反射率）的分布情况。
# 这个分布被用来生成一个4D的直方图和一个2D的计数图
# outputs传入的是透射率的变量
def compute_histogram_4d(omega_i, omega_o, outputs, theta_bins=180, phi_bins=360):

    # 将wi和wo转换为球坐标系
    theta_i = dr.acos(omega_i.z)
    phi_i = dr.atan2(omega_i.y, omega_i.x)
    theta_o = dr.acos(omega_o.z)
    phi_o = dr.atan2(omega_o.y, omega_o.x)

    # 将连续的角度值映射到离散的 bin 中，theta_bins 和 phi_bins 分别代表极角和方位角的离散化数量（即 bin 的数量）
    # theta_i_idx 和 phi_i_idx 是 omega_i 的角度对应的 bin 索引。
    # theta_o_idx 和 phi_o_idx 是 omega_o 的角度对应的 bin 索引
    # 按照作者的命令执行时,theta_bins为18,phi_bins为36
    # θ（极角）：表示从球心到表面点的垂直角度，范围是 [0, π]
    # theta_bins//2,作者可能是想将角度映射到半个球面上的网格，所以将theta_bins // 2作为一个缩放因子，可能是为了限制θ的范围只到π/2
    # phi_i_idx = mi.UInt32((phi_i + dr.pi) / dr.two_pi * phi_bins)：这部分操作是将φ角度从[-π, π]映射到[0, 2π]范围内，之后将其映射到phi_bins个离散的bin中。这里没有除以2，因为整个范围是2π
    theta_i_idx = mi.UInt32(theta_i / (dr.pi / 2) * (theta_bins//2))
    phi_i_idx = mi.UInt32((phi_i + dr.pi) / dr.two_pi * phi_bins)
    theta_o_idx = mi.UInt32(theta_o / dr.pi * theta_bins)
    phi_o_idx = mi.UInt32((phi_o + dr.pi) / dr.two_pi * phi_bins)
    
    # Filter out out-of-bound indices (i.e., -1 or indices equal to the number of bins)
    # 确保计算出来的 bin 索引在有效范围内。如果索引越界（如小于零或者大于最大 bin 数量），则它们会被过滤掉
    valid_mask = mi.Bool(
        (theta_i_idx >= 0) & (theta_i_idx < theta_bins//2) & \
        (phi_i_idx >= 0) & (phi_i_idx < phi_bins) & \
        (theta_o_idx >= 0) & (theta_o_idx < theta_bins) & \
        (phi_o_idx >= 0) & (phi_o_idx < phi_bins))

    # Apply the mask to indices and outputs
    # 应用 valid_mask 来筛选出有效的 theta_i_idx、phi_i_idx、theta_o_idx、phi_o_idx 和 outputs。只有有效的角度和输出值才会进入直方图的计算
    theta_i_idx = theta_i_idx[valid_mask]
    phi_i_idx = phi_i_idx[valid_mask]
    theta_o_idx = theta_o_idx[valid_mask]
    phi_o_idx = phi_o_idx[valid_mask]  
    outputs = outputs[valid_mask]

    # Create a 4D array to accumulate the sum of outputs for each bin
    # 创建一个形状为 (theta_bins//2, phi_bins, theta_bins, phi_bins, 3) 的4D零数组 s，用于存储每个 bin 对应的输出值
    # 3 表示每个网格点的维度，通常在计算光照、颜色、方向等三维向量时，涉及到 RGB 或 3D 方向向量
    shape = (theta_bins//2, phi_bins, theta_bins, phi_bins, 3)
    s = dr.zeros(mi.Float, dr.prod(shape))

    # Accumulate outputs into the bins
    # scatter_reduce 用于将 outputs.x、outputs.y 和 outputs.z 的值按 theta_i_idx、phi_i_idx、theta_o_idx 和 phi_o_idx 索引累积到数组 s 中
    # dr.ReduceOp.Add 指定了累加操作（即将每个 bin 中的值相加）
    dr.scatter_reduce(dr.ReduceOp.Add, s, outputs.x, ravel_index([theta_i_idx, phi_i_idx, theta_o_idx, phi_o_idx, mi.UInt32(0)], shape))
    dr.scatter_reduce(dr.ReduceOp.Add, s, outputs.y, ravel_index([theta_i_idx, phi_i_idx, theta_o_idx, phi_o_idx, mi.UInt32(1)], shape))
    dr.scatter_reduce(dr.ReduceOp.Add, s, outputs.z, ravel_index([theta_i_idx, phi_i_idx, theta_o_idx, phi_o_idx, mi.UInt32(2)], shape))

    # Create a 2D array to accumulate the count of omega_i for each bin
    # 创建一个 2D 数组 count_i 用于存储每个 theta_i 和 phi_i 对应的计数,即存储每个 bin 对应的计数值
    # 目的是统计每个 bin 中有多少个 omega_i
    count_i_shape = (theta_bins//2, phi_bins)
    count_i = dr.zeros(mi.Float, dr.prod(count_i_shape))

    # scatter_reduce 将每个 omega_i 对应的 bin 累加 1
    dr.scatter_reduce(dr.ReduceOp.Add, count_i, mi.Float(1), ravel_index([theta_i_idx, phi_i_idx], count_i_shape))

    # 调用 calculate_solid_angle_grid 函数计算每个 bin 的立体角。立体角是用于量化每个 bin 在球面上的面积的一个量度
    solid_angles, _, _ = calculate_solid_angle_grid(theta_bins, phi_bins)

    # 归一化直方图，使得每个 bin 的值与其计数和立体角相关联
    # 首先，将 s（存储输出值的数组）除以 count_i，得到每个 bin 的平均输出值。
    # 然后，再除以 solid_angles，以便考虑到不同 bin 对应的实际表面积。
    # 最后，使用 dr.select 将可能出现的 NaN 值替换为 0。
    histogram = mi.TensorXf(s, shape)
    histogram_count = mi.TensorXf(count_i, count_i_shape)
    histogram /= histogram_count[:, :, None, None, None] # 目的：这里的归一化是通过将 histogram 除以 histogram_count 来计算每个 bin 的 平均输出值。
    # 上面这行代码为什么这样做：在每个 bin 中，可能有多个 omega_i 对应到该 bin，这些 omega_i 的值会被累加在 s 中，因此这里通过将 s 除以 histogram_count，得到的是每个 bin 中的平均值
    histogram /= solid_angles[None, None, :, :, None]
    histogram = dr.select(dr.isnan(histogram), 0.0, histogram)

    # 函数返回两个值：
    # histogram：是一个4D数组，其中包含了归一化的输出值。它表示的是每个角度bin中与输入数据相关的**“值”**。通过归一化，histogram考虑了以下几个因素：
    # 每个bin中的出现次数（由histogram_count给出）
    # 每个bin对应的立体角面积（通过solid_angles进行计算）

    # histogram_count：是一个计数图（2D数组），它表示每个bin中出现的次数。
    # 它记录了每个bin中被累积的数量，即每个角度区间（theta_i和phi_i）中对应的数量。
    # 这个值的意义可以理解为直方图中每个区域的“计数”，它是数据分布的一个指标

    # histogram_count是一个简单的计数，告诉你每个bin中有多少数据点
    # histogram 是一个加权计数，它将每个bin的数据量与bin的实际面积、空间权重结合起来，给出更加真实的数值，
    # 通常用于表示在特定空间位置上的数据密度或者某种物理量
    return histogram, histogram_count

def cart2sph(v):
    v = dr.norm(v)
    theta = dr.acos(v.z)
    phi = dr.atan2(v.y, v.x)
    return theta, phi


# 主要作用是通过追踪场景中的光线，计算并收集用于光传输模拟（比如反射、透射、多重散射）的 RDM数据
# scene：场景对象，表示当前的 3D 场景。
# num_samples：样本数量，用来采样光线的数量。默认值为 1024*1024*16，即 16,777,216 条光线。
# theta_bins 和 phi_bins：分别是极角（theta）和方位角（phi）分箱的数量（即分辨率）。使用整数除法 // 来计算它们的值，这意味着 theta_bins=45 和 phi_bins=90。
# max_depth：最大光线追踪深度。通常设为较大值，比如 4096，用于控制光线最大反射或折射的次数。
def collect_rdm(scene, num_samples=1024*1024*16, theta_bins=180//4, phi_bins=360//4, max_depth=4096):
    # 生成一个形状为 (num_samples, 2) 的张量，其中每一行都是一个 2D 正态分布的随机值，表示随机的方向向量（x, y），生成 num_samples 个光线样本
    o = torch.randn(num_samples, 2, device=device)
    # 计算 o 每行的 L2 范数（向量的模），并将 o 归一化，使得每个方向的向量的长度为 1。
    o /= torch.norm(o, dim=-1, keepdim=True)
    # 将 2D 向量 o 转换为 3D 空间中的点（z 分量为 0），表示光源位置
    o = mi.Point3f(0.0, o[:, 0], o[:, 1])
    # 根据方向 o 计算光线的方向。d 是光线的方向向量，它是 -o，表示从 o 向外发射。
    d = mi.Vector3f(-o)
    # 创建一批光线对象，起点是 o，方向是 d。num_samples条光线
    ray = mi.Ray3f(o, d)

    # 路径追踪
    # wi：入射方向。
    # wo：出射方向。
    # throughput：透射率，表示光在每次反射或折射时的强度衰减。
    # depth：光线的深度，指示当前的反射或折射次数。
    # frame：用于转换方向的框架（坐标系）。
    # si：光线与场景中的表面交点信息。
    wi, wo, throughput, depth, frame, si = trace_paths(scene, ray, max_depth=max_depth)

    # 通过 dr.select 函数，若光线的深度小于 2，则将透射率设为 0.0。这意味着不考虑深度小于 2 的光线。
    # DrJiT 的核心函数（如 dr.select、dr.dot 等）会自动利用 并行计算
    throughput = dr.select(depth < 2, 0.0, throughput)

    # 该断言检查是否所有的光线都有交点。si.shape == None 表示该光线没有与场景相交。如果某些光线没有相交，函数会抛出异常。
    assert dr.count(dr.eq(si.shape, None)) == num_samples, f"Not all rays escaped: {dr.count(dr.eq(si.shape, None))}/{num_samples}"

    # 将深度减去 2，这样后续的计算就会偏移，忽略了两次初始的交点
    depth -= 2

    # 三个布尔类型变量的数组,便于并行计算
    # select_t：选择透射（穿透）光线。
    # select_r：选择反射光线（只有当反射角度与法线方向一致时）。
    # select_m：选择多重散射光线。
    select_t = dr.eq(depth, 0)
    select_r = dr.eq(depth, 1) & (dr.dot(wo, frame.n) > 0.0)
    select_m = ~select_t & ~select_r

    # 数组
    # 使用 dr.select 函数，根据这些条件来更新 throughput，对于其他不符合条件的光线，将其透射率设为 0。
    throughput_t = dr.select(select_t, throughput, 0.0)
    throughput_r = dr.select(select_r, throughput, 0.0)
    throughput_m = dr.select(select_m, throughput, 0.0)

    # compute_histogram_4d 用于计算透射、反射和多重散射的 RDM。它会根据 wi、wo（入射和出射方向）以及 throughput（透射率）计算 4D 直方图，
    # 并返回 RDM 数据和每个 bin 的计数
    rdm_t, count_t = compute_histogram_4d(wi, wo, throughput_t, theta_bins=theta_bins, phi_bins=phi_bins)
    rdm_r, count_r = compute_histogram_4d(wi, wo, throughput_r, theta_bins=theta_bins, phi_bins=phi_bins)
    rdm_m, count_m = compute_histogram_4d(wi, wo, throughput_m, theta_bins=theta_bins, phi_bins=phi_bins)

    return rdm_t, rdm_r, rdm_m, count_t, count_r, count_m

# // 是整数除法运算符，5 // 2 的结果是 2
def compute_rdm(parameters, theta_bins=180//4, phi_bins=360//4, max_depth=4096, num_batches=1024, batch_size=1024*8):
    # 创建一个 Mitsuba 场景，场景包括：
    # 一个由 create_single_yarn(parameters) 生成的纤维（yarn）。
    # 一个线性曲线（wrap），可能是纤维的包裹结构。
    # 两个磁盘（disk1 和 disk2），用于定义光源或探测器。
    scene = mi.load_dict({
        'type': 'scene',
        'curves': create_single_yarn(parameters),
        'wrap': {
            'type':'linearcurve',
            'filename': 'curves/ply.txt',
            'brdf_cylinder':{
                'type': 'null',
            }   
        },
        'disk1':{
            'type': 'disk',
            'id': 'disk1',
            'to_world': sT.look_at(origin=[5.0, 0.0, 0.0], target=[10.0, 0.0, 0.0], up=[0.0, 0.0, 1.0]).scale(0.5),
            'brdf_disk1':{
                'type': 'null',
            }
        },
        'disk2':{
            'type': 'disk',
            'id': 'disk2',
            'to_world': sT.look_at(origin=[-5.0, 0.0, 0.0], target=[-10.0, 0.0, 0.0], up=[0.0, 0.0, 1.0]).scale(0.5),
            'brdf_disk2':{
                'type': 'null',
            }
        },
    })

    # 批量计算 RDM，循环 num_batches 次，每次调用 collect_rdm 函数计算一个批次的 RDM 数据
    for i in range(num_batches):
        # bin 是指将连续的 theta（极角）和 phi（方位角）角度范围划分为离散的区间（即“分箱”或“分桶”），每个区间称为一个 bin，分辨率 指的是将角度范围划分的精细程度，即 bin 的数量
        rdm_t_, rdm_r_, rdm_m_, count_t_, count_r_, count_m_ = collect_rdm(scene, batch_size, theta_bins, phi_bins, max_depth)

        try:
            rdm_t += rdm_t_
            rdm_r += rdm_r_
            rdm_m += rdm_m_
            count_t += count_t_
            count_r += count_r_
            count_m += count_m_
        except:
            print('Collecting RDM')
            rdm_t = rdm_t_
            rdm_r = rdm_r_
            rdm_m = rdm_m_
            count_t = count_t_
            count_r = count_r_
            count_m = count_m_

        print(f'Batch {i+1}/{num_batches}')

    # 归一化 RDM 数据，得到平均值，分别是透射、反射、多重散射
    rdm_t /= num_batches
    rdm_r /= num_batches
    rdm_m /= num_batches
    
    # Calculate the centers of the theta and phi bins
    # 计算 theta 和 phi 的 每个bin 中心坐标。,计算了与不同 bin 相关的角度中心值
    theta_i_centers = np.linspace(0, np.pi / 2, theta_bins//2, endpoint=False) + (np.pi / 2 / (theta_bins//2)) / 2
    phi_i_centers = np.linspace(-np.pi, np.pi, phi_bins, endpoint=False) + (2 * np.pi / phi_bins) / 2
    theta_o_centers = np.linspace(0, np.pi, theta_bins, endpoint=False) + (np.pi / theta_bins) / 2
    phi_o_centers = np.linspace(-np.pi, np.pi, phi_bins, endpoint=False) + (2 * np.pi / phi_bins) / 2

    # 使用 np.meshgrid 生成 4D 网格坐标
    theta_i, phi_i, theta_o, phi_o = np.meshgrid(theta_i_centers, phi_i_centers, theta_o_centers, phi_o_centers, indexing='ij')
    # 将坐标堆叠成一个 4D 张量 x，表示每个 bin 的中心位置
    x = np.stack([theta_i, phi_i, theta_o, phi_o], axis=-1)
    x = mi.TensorXf(x)

    # 调用 calculate_solid_angle_grid计算每个 bin 的立体角
    solid_angles = mi.TensorXf(calculate_solid_angle_grid(theta_bins, phi_bins)[0])

    # 计算得到的 透射、反射、多重散射的RDM 数据、相应的数量、bin 中心坐标和立体角
    return rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, solid_angles