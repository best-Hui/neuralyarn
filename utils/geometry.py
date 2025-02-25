from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)

import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import bsdf.khungurn as khungurn

# 将三维曲线数据写入指定的文件中
def streamcurves(filepath, curves, radius, verbose=True):
    # filepath: 目标文件的路径，曲线数据将被写入此文件。
    # curves: 传入的曲线数据，是一个包含多个曲线的列表。每条曲线是由多个三维点（即包含 3 个坐标值的数组）组成的。
    # radius: 用于写入每个曲线点的附加值（可能是曲线的半径）。
    # verbose: 布尔值，表示是否打印详细信息，默认值为 True。
    if verbose:
        print(f'Writing to {filepath}')

    # 打开目标文件，并确保文件在写入完成后正确关闭
    with open(filepath, 'w') as file:
        # 遍历输入的每条曲线。每条曲线是一个三维坐标点的列表（或者说是一个二维数组）
        for curve in curves:
            # 对于每个曲线中的每个点（for p in curve:），检查该点是否包含无效值（如 NaN）。
            for p in curve:
                if np.any(np.isnan(p)):
                    # 如果点有效（没有 NaN），则将该点的坐标和半径写入文件。写入的格式是：x y z radius，其中 x, y, z 是点的坐标，radius 是附加值
                    print('WARN streamcurves - Invalid point', p)
                    continue
                
                file.write(f'{p[0]} {p[1]} {p[2]} {radius}\n')
            file.write('\n')

# 生成在一个圆形截面内分布的随机点。这些点被用作某些物理模型（例如纤维或线圈的模型）的起始点
# 函数的目的是在一个给定的圆形区域内放置一定数量的点，并确保这些点满足一定的分布规则。
def cross_section_circle(macro_radius, count, density):
    # macro_radius：宏观半径，表示生成的圆的半径。
    # count：要生成的点的数量。
    # density：点的密度，用于计算每个点的微观半径。

    # micro_radius 是每个点的半径。这个值是根据密度和宏观半径以及点的数量计算出来的。它决定了点与点之间的间距，以避免重叠。
    micro_radius = np.sqrt((density * macro_radius**2) / count)
    
    # Generate random 2d samples between 0 & 1
    # Try N * 10 times before giving up
    # 生成 count * 10 个随机的二维样本，范围在 [0, 1) 之间。这是为了提高生成点的机会，避免生成的点不符合要求
    sample_x = np.random.rand(count * 10)
    sample_y = np.random.rand(count * 10)

    # Find radius of allowed points
    # radius 是生成点时允许的最大距离。它是宏观半径减去每个点的微观半径，确保每个点之间有足够的空间。
    radius = macro_radius - micro_radius

    # Map 2d random samples to interval
    # 将随机样本映射到 [-radius, radius] 区间内。这样生成的样本点将分布在以原点为中心的一个方形区域内。
    sample_x = sample_x * (radius * 2) - radius
    sample_y = sample_y * (radius * 2) - radius

    # Initialise points
    # Inf is used so that calculating distances would be easier
    # 初始化 x_points 和 y_points 数组，用 np.inf 填充。这是为了方便后续计算点之间的距离，并通过比较 np.inf 判断是否已经找到有效的点
    x_points = np.full(count, np.inf)
    y_points = np.full(count, np.inf)
    idx = 0


    for x, y in zip(sample_x, sample_y):
        # 通过计算 micro_distance（与已有点的最小距离）和 macro_distance（到圆心的距离），判断当前点是否有效：
        # micro_distance 用来检测当前点是否与已有点重叠（避免点之间的重叠）。
        # macro_distance 用来确保点在允许的圆形区域内。
        micro_distance = np.min(np.sqrt((x_points - x) ** 2 + (y_points - y) ** 2))    # Distances to each fiber center
        macro_distance = np.sqrt(x ** 2 + y ** 2)                                      # Distance to the yarn center

        # is_self_intersect: 如果当前点与已有点的距离小于 2 * micro_radius，则认为它与其他点相交。
        # is_in_circle: 如果当前点到圆心的距离小于 radius，则认为它在允许的圆形区域内。
        is_self_intersect = micro_distance < 2 * micro_radius                          # Is intersecting with other fibers
        is_in_circle = macro_distance < radius                                         # Is in the yarn circle

        #  Add point if it is valid
        # 如果当前点有效（满足不相交且在圆内），将其加入到 x_points 和 y_points 数组，并增加索引 idx
        if is_in_circle and not is_self_intersect:
            x_points[idx] = x
            y_points[idx] = y
            idx += 1

        # Complete
        # 当已添加的有效点数达到 count 时，退出循环。
        if idx == count:
            break

    # Remove all inf values
    # 删除数组中的 np.inf 值（表示未被赋值的点）。
    x_points = x_points[np.invert(np.isinf(x_points))]
    y_points = y_points[np.invert(np.isinf(y_points))]

    # 将有效的 x_points 和 y_points 合并成一个 offsets 数组，并检查生成的点数量是否与 count 一致。如果不一致，抛出错误。
    offsets = np.vstack([x_points, y_points]).T

    if offsets.shape[0] != count:
        raise RuntimeError(f"Error, unable to generate all curves ({offsets.shape[0]}/{count}).")

    return np.vstack([x_points, y_points]), micro_radius

# helix 函数生成一个由若干横截面构成的螺旋曲线。
# 每个横截面是通过旋转和偏移操作对每个点对（p1 和 p2）进行处理后得到的。
# 螺旋的每一段都在旋转的同时按一定的规律逐步改变方向和位置，形成螺旋的形状。
# 最终生成的是一个三维的螺旋曲线，每个点的坐标都被保存在 curves 数组中。
def helix(points, radius, offset, twist_factor, up_vector=np.array([0, 0, 1])):
    # points: 输入的点序列，通常表示曲线的形状。每两个相邻的点构成一条曲线段。
    # radius: 螺旋的半径，用于计算旋转角度和位移。
    # offset: 该参数控制螺旋的横截面偏移，通常是一个二维数组（(2, n)），用于描述偏移量的分布。
    # twist_factor: 控制螺旋旋转速率的因子，决定每个小段的旋转角度。
    # up_vector: 上方向向量，默认值是 [0, 0, 1]，定义了螺旋的“向上”方向。


    # To store the coordinates of the cross section
    # cross_section_xs, cross_section_ys, cross_section_zs: 用于存储螺旋横截面的 x, y, z 坐标。
    cross_section_xs = []
    cross_section_ys = []
    cross_section_zs = []

    #  初始化旋转角度为 0。它将随曲线的每一小段逐步增加，形成螺旋的旋转效果
    twist_angle = 0

    # 遍历输入的 points 中的相邻点对，p1 和 p2 构成一条线段。points[:-1] 是从第一个点到倒数第二个点，points[1:] 是从第二个点到最后一个点。
    # For each point in the curve
    for i, (p1, p2) in enumerate(zip(points[:-1], points[1:])):
        cross_section = np.empty((3, offset.shape[0])) 

        # The displacement vector is p1 -> p2
        # 计算当前线段的位移向量 displacement，以及该线段的长度 distance。
        displacement = p2 - p1
        distance = np.linalg.norm(displacement)

        # 如果 distance 过小（接近零），说明该线段非常短，可能是由于浮动误差导致的。这时就直接将上一段的横截面坐标加入，并跳过当前的处理。
        if np.isclose(distance, 0):
            cross_section_xs.append(cross_section_xs[-1])
            cross_section_ys.append(cross_section_ys[-1])
            cross_section_zs.append(cross_section_zs[-1])
            continue

        # Get tangent, binormal, and normal
        # 计算线段的切向量 tangent，它是从 p1 指向 p2 的单位向量。
        # 使用 up_vector 和 tangent 的叉积计算副法向量 binormal，然后标准化。
        # 使用 tangent 和 binormal 的叉积计算法向量 normal，然后标准化。
        # 这些向量共同确定了螺旋的平面和旋转方向。
        tangent = displacement
        tangent /= np.linalg.norm(tangent)

        binormal = np.cross(up_vector, tangent)
        binormal /= np.linalg.norm(binormal)

        normal = np.cross(tangent, binormal)
        normal /= np.linalg.norm(normal)

        # Apply rotations, t = n / (x * d)
        # 根据当前线段的长度 distance 和 twist_factor，计算当前线段的旋转角度 twist_angle。
        # 使用 scipy.spatial.transform.Rotation 来生成一个旋转矩阵，并应用到 binormal 和 normal 上，得到旋转后的副法向量和法向量。
        twist_angle += np.pi * twist_factor * distance / radius
        rotation = Rot.from_rotvec(tangent * twist_angle)
        binormal = rotation.apply(binormal)
        normal = rotation.apply(normal)

        # Add the offsets to p1 to create the crossection
        # 通过将偏移量 offset 应用到旋转后的副法向量 binormal 和法向量 normal，计算横截面坐标。
        # 这些横截面点被加到 cross_section_xs, cross_section_ys, cross_section_zs 中。
        cross_section = p1 + offset[0][np.newaxis].T * binormal + offset[1][np.newaxis].T * normal
        cross_section = cross_section.T
        cross_section_xs.append(cross_section[0])
        cross_section_ys.append(cross_section[1])
        cross_section_zs.append(cross_section[2])

    # Combine x, y, z
    # 将所有的横截面点集合转换为 NumPy 数组，并将它们组合成一个三维数组 curves，包含每个横截面在 x, y, z 方向上的坐标。
    cross_section_xs = np.asarray(cross_section_xs).T
    cross_section_ys = np.asarray(cross_section_ys).T
    cross_section_zs = np.asarray(cross_section_zs).T
    curves = np.dstack([
        cross_section_xs,
        cross_section_ys,
        cross_section_zs
    ])

    return curves


# 创建了一个带有物理材质属性的螺旋形纱线：
# 首先生成了基础曲线，并利用 helix 函数为每条曲线生成纤维的螺旋形态；
# 接着，基于这些纤维数据，设置材质参数，并使用 Mitsuba 渲染器加载纱线的数据结构。
# 最终，返回包含所有参数和控制点的曲线对象。
def create_single_yarn(parameters, resolution=1001, ply_radius=0.5, start=-5.0, end=5.0, verbose=True):
    # parameters: 一个字典，包含各种控制参数，如纱线的数量、反射和透射的RGB值、长度宽度、密度等。
    # resolution: 解析度，默认为1001，控制纱线的分辨率，影响每根纱线的细分度。
    # ply_radius: 纱线的半径，默认为0.5。
    # start, end: 定义生成曲线的起始和结束位置，默认从 -5.0 到 5.0。
    # verbose: 布尔值，决定是否打印详细信息，默认 True。


    # 使用 np.linspace 创建一个从 start 到 end 等间隔的数组，用于表示沿着 x 轴的坐标。
    # np.zeros(resolution) 创建了两个长度为 resolution 的数组，分别用于 y 和 z 坐标（默认为0）
    # 使用 np.stack 将这三维坐标堆叠起来，形成一个 resolution x 3 的数组，表示基础的直线曲线。
    # 通过 [np.newaxis] 扩展维度，使得 ply_curves 的形状为 (1, resolution, 3)，这将作为后续生成纤维曲线的输入。
    ply_curves = np.stack([
        np.linspace(start, end, resolution),
        np.zeros(resolution),
        np.zeros(resolution)
    ], axis=1)[np.newaxis]

    # Generate fiber curves
    # 调用 cross_section_circle 函数生成纤维的偏移量和半径，其中 ply_radius 是每根纱线的半径，parameters['N'] 是纱线数量，parameters['Rho'] 是密度。
    # 使用 helix 函数基于 ply_curves 生成每根纱线的螺旋形曲线（fiber_curves）。
    # fiber_curves 的结果是一个三维数组，表示所有纤维的曲线。
    fiber_offset, fiber_radius = cross_section_circle(ply_radius, int(parameters['N']), parameters['Rho'])
    fiber_curves = [helix(ply, ply_radius, fiber_offset, parameters['Alpha']) for ply in ply_curves]
    fiber_curves = np.asarray(fiber_curves)

    # 通过 fiber_curves.shape 解包其维度，得到 a, b, c, d。这里，a 和 b 表示不同的纱线层（ply），c 和 d 分别表示每个点的坐标和维度。
    # 使用 reshape 将 fiber_curves 重塑为二维数组，展平所有层的曲线，确保其形状为 (a * b, c, d)，这有利于后续处理。
    a, b, c, d = fiber_curves.shape
    fiber_curves = fiber_curves.reshape([a * b, c, d])


    # 通过 mi.load_dict 加载纱线的物理属性。这些属性包括：
    # type:设定材质类型为linearcurve
    # filename: 纱线数据文件路径。
    # mybsdf: 自定义的 BSDF（双向散射分布函数），采用 khungurn 材质类型，设置其反射率 reflectance 和透射率 transmittance，以及相关的宽度参数 r_longwidth, tt_longwidth, 和 tt_aziwidth，这些都是纱线材质的物理属性。
    # parameters 中的各种参数（如反射率、透射率等）从输入字典中传递。
    curves = mi.load_dict({
        'type': 'linearcurve',
        'filename': 'curves/ply.txt',
        "mybsdf": {
            "type": "khungurn",
            "reflectance": {
                "type": "rgb",
                "value": [parameters['R_r'], parameters['R_g'], parameters['R_b']]
            },
            "transmittance": {
                "type": "rgb",
                "value": [parameters['TT_r'], parameters['TT_g'], parameters['TT_b']]
            },
            "r_longwidth": parameters['R_long'],
            "tt_longwidth": parameters['TT_long'],
            "tt_aziwidth": parameters['TT_azim'],
        }
    })

    # 使用 mi.traverse 方法遍历 curves 对象，返回与之相关的所有参数
    params = mi.traverse(curves)

    # 使用 NumPy 创建一个从 0 到 N * (resolution - 1) 的索引序列，然后将其转换为 PyTorch 张量，并移动到指定的设备（device，这里为cuda）。
    # ends 用于标记最后的索引，它会过滤掉每个曲线的最后一段，避免重复。
    # 最终的 segment_indices 是去除了最后段的索引，用于控制曲线的分段。
    segment_indices = torch.tensor(np.arange(0, parameters['N'] * (resolution - 1))).to(device)
    ends = ((segment_indices + 1) % (resolution - 1)) == 0
    segment_indices = segment_indices[~ends]


    # 计算控制点的数量 control_point_count，并将 fiber_curves 和每个点的半径 fiber_radius 合并成一个控制点数组 control_points。
    # segment_indices 被更新为整数类型，代表每个段的索引。
    # 使用 params.update() 更新所有参数。
    params['control_point_count'] = np.prod(fiber_curves.shape[:2])
    params['control_points'] = np.concatenate([fiber_curves, np.full(fiber_curves.shape[:2], fiber_radius)[:, :, None]], axis=-1).ravel()
    params['segment_indices'] = mi.UInt32(segment_indices.to(torch.uint32))
    params.update()

    # 返回最终的曲线数据对象 curves，其中包含了生成的纱线以及相关的材质和物理参数
    return curves





