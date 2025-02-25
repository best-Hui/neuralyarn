from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)


# 自定义的双向散射分布函数（BSDF），用于 Mitsuba 3 渲染器，继承 BSDF
# 纤维
class Khungurn(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.m_reflectance = props['reflectance']
        self.m_transmittance = props['transmittance']

        # Read properties from `props`
        self.r_longwidth = dr.deg2rad(props['r_longwidth'])
        self.tt_longwidth = dr.deg2rad(props['tt_longwidth'])
        self.tt_aziwidth = dr.deg2rad(props['tt_aziwidth'])
                
        # Set the BSDF flags
        scattering_flags   = mi.BSDFFlags.Glossy   | mi.BSDFFlags.FrontSide | mi.BSDFFlags.Anisotropic
        self.m_components  = [scattering_flags]
        self.m_flags = scattering_flags

    # general and coordinates helper functions
    # 采用 Schlick 近似 计算菲涅尔反射率。
    def fresnel(self, C_R, theta_i):
        return C_R + (1.0 - C_R) * dr.power(1.0 - dr.abs(dr.cos(theta_i)), 5)

    # 将三维方向 vector 转换成球面坐标 theta（仰角）和 phi（方位角）
    def to_spherical(self, vector):
        theta = dr.safe_asin(vector[0])
        phi = dr.atan2(vector[2], vector[1])
        return theta, phi

    # 将球面坐标转换回 笛卡尔坐标
    def to_cartesian(self, theta, phi):
        return mi.Vector3f(dr.sin(theta), dr.cos(theta) * dr.cos(phi), dr.cos(theta) * dr.sin(phi))

    # 进行安全除法，避免分母为 0 时发生错误。
    def safe_divide(self, numerator, denominator):
        return dr.select(dr.eq(denominator, 0.0), 0.0, numerator / denominator)

    # 计算浮点数的 模运算 (a % b)，但调整负数情况，使得返回值始终为非负数（与 Python % 操作符不同）。
    # remainder = a−b×⌊a / b⌋
    # 如果remainder小于0，则加上b以保证返回的值为非负数。
    def f_mod(self, a, b):
        remainder = a - b * dr.floor(a / b)
        negative_mask = remainder < 0
        remainder[negative_mask] += dr.abs(b)
        return remainder

    # 归一化角度，使其范围在 (-π, π] 之间
    def regularize(self, angle):
        angle = self.f_mod(angle, 2 * dr.pi)

        more_than = angle > +dr.pi
        less_than = angle < -dr.pi

        angle[more_than] = angle[more_than] - dr.two_pi
        angle[less_than] = angle[less_than] + dr.two_pi
        
        return angle

    # important sampling pdf helper functions#

    # 计算均匀分布的概率密度函数（PDF）。
    def uniform_pdf(self, x, a, b):
        return 1.0 / (b - a)

    # 计算标准正态分布（均值 0，标准差 1）的累积分布函数（CDF）。
    # 累计分布函数（Cumulative Distribution Function, CDF）是一个用来描述随机变量在某个特定值或更小值上所累积的概率的函数
    # 累计分布函数（CDF）是概率密度函数（PDF）在某个点之前的积分
    # 这里是使用误差函数帮助计算的,因为正态分布函数的累积积分没有精确解,一般使用数值解,误差函数的相关公式自行搜索
    # 标准正态分布的 CDF 可以通过误差函数表示,根据公式来的,自行搜索
    def std_gaussian_cdf(self, x):
        return 0.5 * (1.0 + dr.erf(x / dr.sqrt(2.0)))
    
    def gaussian_cdf(self, x, mu, sigma):
        return self.std_gaussian_cdf(x)

    # 计算高斯分布在区间 [a, b] 内的积分
    def gaussian_integral(self, mu, sigma, a, b):
        return self.gaussian_cdf(b, mu, sigma) - self.gaussian_cdf(a, mu, sigma)

    # 计算高斯分布的概率密度函数（PDF）
    # 完全是根据正态分布的pdf公式来的,自行搜索
    def gaussian_pdf(self, x, mu, sigma):
        t1 = 1.0 / (dr.sqrt(2.0 * dr.pi * dr.power(sigma, 2)))
        t2 = dr.exp(-dr.power(x - mu, 2)/(2.0 * dr.power(sigma, 2)))
        return t1 * t2

    # 截断正态分布pdf
    # 是一种对正态分布进行限制（截断）后的分布，通常用于表示那些在特定区间内被限制的随机变量。
    # 也就是说，截断正态分布是在正态分布的基础上，去除了某些值或限制了值的范围
    def uct_gaussian_pdf(self, x, mu, sigma, a, b):
        t1 = self.gaussian_pdf(x, mu, sigma)
        t2 = (1.0 - self.gaussian_integral(mu, sigma, a, b)) / (b - a)
        return t1 + t2
    
    # important sampling sample helper functions

    # 将 [0,1] 之间的随机变量 sample 变换为区间 [a, b] 之间的随机变量
    def uniform_sample(self, sample, a, b):
        return sample * (b - a) + a

    # 用误差函数（erf）反推一个标准正态分布（均值 0，标准差 1）的采样值
    def std_gaussian_sample(self, sample):
        return dr.sqrt(2.0) * dr.erf(2.0 * sample - 1.0)

    # 生成一个均值为 mu，标准差为 sigma 的正态分布样本
    def gaussian_sample(self, sample, mu, sigma):
        return self.std_gaussian_sample(sample) * sigma + mu

    # 实现了一种受限高斯采样(utc)
    def uct_gaussian_sample(self, sample, mu, sigma, a, b):
        s_g = self.gaussian_sample(sample, mu, sigma)
        s_u = self.uniform_sample(mi.PCG32(dr.width(sample), sample*12998409.3423098).next_float32(), a, b)
        inside = (a < s_g) & (s_g < b)
        result = dr.select(inside, s_g, s_u)
        # 如果 s_g 在区间内，就返回高斯样本；如果不在区间内，就返回均匀分布样本
        return result
    
    #eval function helper functions
    # 计算某种带 cos² 权重的正态分布积分，可能用于归一化常数的计算
    def G_integral(self, theta, mean, std):
        std2 = dr.power(std, 2)

        # 多项式近似计算
        # p 是多项式系数，可能用于逼近某个函数（如 cos²(x)）。
        # b[j] 递推计算系数。
        p = [1.0001, 0.0, -0.999745, 0.0, 0.3322, 0.0, -0.04301, 0.0, 0.002439]
        b = [0.0] * 8


        b[7] = p[8]
        b[6] = p[7] + mean * b[7]
        for j in range(5, -1, -1):
            b[j] = p[j + 1] + mean * b[j + 1] + (j + 2) * std2 * b[j + 2]
        
        B = 0
        for k in range(8):
            B += b[k] * dr.power(theta, k)

        A = p[0] + mean * b[0] + std2 * b[1]

        # 第一项：误差函数 erf() 计算从 -∞ 到 theta 的高斯积分。
        # 第二项：多项式 B 乘 高斯 PDF，可能用于修正计算。
        result = (A/2) * dr.erf((theta - mean) / (dr.sqrt(2.0) * std)) - std2 * B * self.gaussian_pdf(theta, mean, std)
        return result

    # 表示一个从 -π/2 到 π/2 的累积积分量。
    # 可能用于计算某个归一化因子。
    def G(self, mean, std):
        result = self.G_integral(dr.pi / 2.0, mean, std) - self.G_integral(-dr.pi/2.0, mean, std)
        return result

    # self.gaussian_pdf(theta, mean, std) 计算标准正态分布的 PDF。
    # G(mean, std) 可能是归一化因子，确保这个分布的总积分为 1。
    # safe_divide 处理分母可能为 0 的情况。
    def cos2_normalized_gaussian(self, theta, mean, std):
        result = self.safe_divide(self.gaussian_pdf(theta, mean, std), self.G(mean, std))
        return result

    # 计算 theta_o 的某种角度分布，可能表示某种反射 BRDF 计算。
    # self.r_longwidth 可能是 std（标准差），控制方向分布的宽度。
    # 论文给出的纤维着色公式
    def M_R(self, theta_i, theta_o):
        result = self.cos2_normalized_gaussian(theta_o, -theta_i, self.r_longwidth)
        return result

    # 论文给出的纤维着色公式
    def N_R(self):
        result = 1.0 / (2.0 * dr.pi)
        return result

    # 逻辑与 M_R 相似，但 self.tt_longwidth 控制分布的宽度，可能用于透射 BRDF/BSDF
    # 论文给出的纤维着色公式
    def M_TT(self, theta_i, theta_o):
        result = self.cos2_normalized_gaussian(theta_o, -theta_i, self.tt_longwidth)
        return result

    # phi_d = phi_o - (phi_i + π) 计算方位角的相对差值，并用 regularize() 规范化到 [-π, π] 之间。
    # self.uct_gaussian_pdf() 计算截断正态分布的 PDF，用于模拟某种角度的方位分布。
    def N_TT(self, phi_i, phi_o):
        phi_d = self.regularize(phi_o - self.regularize(phi_i + dr.pi))
        result = self.uct_gaussian_pdf(phi_d, 0.0, self.tt_aziwidth, -dr.pi, dr.pi)
        return result

    # sample function with importance sampling
    # 实现了基于重要性采样（Importance Sampling）的BSDF采样函数，用于计算反射和透射方向的采样，并估算对应的概率密度函数（PDF）
    def sample(self, ctx, si, sample1, sample2, active):
        # 这里使用 mi.PCG32 生成一个伪随机数采样器。
        # BSDFSample3f() 是 BSDF 采样结构，存储采样出的方向、概率密度等信息。
        sampler = mi.PCG32(1, sample1 * 20938 + sample2[0] * 90654 + sample2[1] * 43092)
        bs = mi.BSDFSample3f()

        # Compute Fresnel terms计算菲涅尔项
        # theta_i, phi_i = self.to_spherical(si.wi)：
        # 将入射方向 wi 转换为球坐标表示，提取 θ_i 和 φ_i。
        # C_R 和 C_TT：
        # C_R 是反射颜色（反射率）。
        # C_TT 是透射颜色（透射率）。
        # att_R = self.fresnel(C_R, theta_i)：
        # 计算菲涅尔反射率，att_R 表示入射角 θ_i 下的反射强度。
        # att_TT = (1 - att_R) * C_TT：
        # 计算透射项，即 att_TT = (1 - 反射率) × 透射率。
        theta_i, phi_i = self.to_spherical(si.wi)
        C_R  = self.m_reflectance.eval(si, active)
        C_TT = self.m_transmittance.eval(si, active)
        att_R = self.fresnel(C_R, theta_i)
        att_TT = (1 - att_R) * C_TT
        att    = att_R + att_TT

        # Compute ratio of R to TT accross 3 channels计算 RGB 颜色通道的比例
        # 计算 RGB 颜色通道的相对权重：
        # ratio_r, ratio_g, ratio_b 归一化到 [0,1]。
        # ratio_R 是反射的比率，ratio_TT = 1 - ratio_R 代表透射的比率。
        ratio_r = att[0]
        ratio_g = att[1]
        ratio_b = att[2]
        ratio_sum = ratio_r + ratio_g + ratio_b
        ratio_r = ratio_r / ratio_sum
        ratio_g = ratio_g / ratio_sum
        ratio_b = 1.0 - ratio_r - ratio_g
        ratio_R  = att_R / (att_R + att_TT)
        ratio_TT = 1.0 - ratio_R



        # Pick between reflection and transmission
        # 采样反射或透射
        # sample1 是一个随机数，用于概率性选择反射（R）或透射（TT）：
        # mi.luminance(ratio_R) 计算 ratio_R 的亮度。
        # 如果 sample1 < ratio_R，选择反射。
        # 否则，选择透射。
        selected_R = (sample1 < mi.luminance(ratio_R))

        # uct_gaussian_sample(sample2[0], -theta_i, self.r_longwidth, -π/2, π/2)：
        # 从截断正态分布中采样 theta_o：
        # 反射情况：使用 self.r_longwidth 控制反射角分布。
        # 透射情况：使用 self.tt_longwidth 控制透射角分布。
        # dr.select 是一个条件选择函数，通常用于根据布尔条件选择不同的值。它的基本用法是：
        # 如果 selected_R 为 True，则选择 A；
        # 如果 selected_R 为 False，则选择 B。
        theta_o = dr.select(selected_R, self.uct_gaussian_sample(sample2[0], -theta_i, self.r_longwidth,  -0.5*dr.pi, 0.5*dr.pi), self.uct_gaussian_sample(sample2[0], -theta_i, self.tt_longwidth, -0.5*dr.pi, 0.5*dr.pi))

        # 采样方位角
        # 反射情况：
        # phi_d 在 [-π, π] 之间均匀采样。
        # 透射情况：
        # phi_d 采用截断高斯分布 self.uct_gaussian_sample() 进行采样。
        phi_d = dr.select(selected_R, self.uniform_sample(sample2[1], -dr.pi, dr.pi), self.uct_gaussian_sample(sample2[1], 0.0, self.tt_aziwidth, -dr.pi, dr.pi))
        phi_o = (phi_i + dr.pi) + phi_d


        # 计算概率密度（PDF）
        # p_R 和 p_TT 是加权混合概率。
        # pdf_R 和 pdf_TT 分别是反射和透射情况的 PDF：
        # 反射的 θ_o 用截断高斯分布采样，φ_o 均匀分布。
        # 透射的 θ_o 和 φ_o 都使用截断高斯分布。
        p_R  = ratio_r * ratio_R[0]  + ratio_g * ratio_R[1]  + ratio_b * ratio_R[2]
        p_TT = ratio_r * ratio_TT[0] + ratio_g * ratio_TT[1] + ratio_b * ratio_TT[2]
        pdf_R  = self.uct_gaussian_pdf(theta_o, -theta_i, self.r_longwidth, -0.5*dr.pi, 0.5*dr.pi) * self.uniform_pdf(phi_d, -dr.pi, dr.pi)
        pdf_TT = self.uct_gaussian_pdf(theta_o, -theta_i, self.tt_longwidth, -0.5*dr.pi, 0.5*dr.pi) * self.uct_gaussian_pdf(phi_d, 0.0, self.tt_aziwidth, -dr.pi, dr.pi)
        _pdf = p_R * pdf_R + p_TT * pdf_TT
        

        # Fill up the BSDFSample struct
        # 填充 BSDFSample 结构
        # 归一化 PDF：
        # _pdf / dr.cos(theta_o) 进行角度修正（Lambert’s cosine law）。
        # 方向变换：
        # self.to_cartesian(theta_o, phi_o) 将球坐标转换回笛卡尔坐标。
        # eta = 1.0，表明无折射（或相对折射率为1）。
        bs.pdf = _pdf / dr.cos(theta_o)
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type      = mi.UInt32(mi.BSDFFlags.Glossy)
        bs.wo = self.to_cartesian(theta_o, phi_o)
        bs.eta = 1.0

        # 计算重要性采样的权重
        # 计算 BSDF 评估值 eval()，再乘以 cos(theta_o) 修正权重
        _eval = self.eval(ctx, si, bs.wo, active)
        weight = (_eval / _pdf) * dr.cos(theta_o)

        return bs, weight & (active & (bs.pdf > 0.0))


    # eval function
    # BSDF（双向散射分布函数）的评估函数 eval()，用于计算给定入射方向 wi 和出射方向 wo 下的 BSDF 值
    # 实际上是纱线的反射分类的bsdf
    def eval(self, ctx, si, wo, active):
        # 计算入射角和出射角
        theta_i, phi_i = self.to_spherical(si.wi)
        theta_o, phi_o = self.to_spherical(wo)

        # 计算反射率和透射率
        # C_R 是反射颜色（反射率）。
        # C_TT 是透射颜色（透射率）。
        # 这两个值可能与材质的 RGB 颜色或纹理相关。
        C_R  = self.m_reflectance.eval(si, active)
        C_TT = self.m_transmittance.eval(si, active)

        #  计算菲涅尔项，表示在 θ_i 角度下，多少光会被反射
        F_R = self.fresnel(C_R, theta_i)

        # 计算反射项
        # 菲涅尔反射率 F_R 影响反射强度。
        # M_R(theta_i, theta_o)：
        # 计算反射方向 θ_o 的方向性分布，通常是**正态分布（Gaussian）**或类似的函数。
        # N_R()：
        # 计算反射的法向化系数，通常是 1 / (2π)，表示在整个球面上的均匀分布。
        # 论文的散射函数公式
        S_R = F_R * self.M_R(theta_i, theta_o) * self.N_R()

        # 计算透射项
        # (1.0 - F_R):
        # 表示光的透射部分（即 1 - 反射率）。
        # C_TT:
        # 透射颜色。
        # M_TT(theta_i, theta_o):
        # 计算透射角度分布，通常是高斯分布或基于微表面模型的分布。
        # N_TT(phi_i, phi_o):
        # 计算透射的方位角影响，用 uct_gaussian_pdf() 计算 φ 方向的概率密度。
        # 光线即使与纤维相交,也可以发生透射.因为纤维是类玻璃管状,因此纤维的bsdf不仅包含反射项,还有透射项
        S_TT = (1.0 - F_R) * C_TT * self.M_TT(theta_i, theta_o) * self.N_TT(phi_i, phi_o)

        # 计算最终的 BSDF 值
        # S_R + S_TT：
        # 反射项 S_R 和透射项 S_TT 之和。
        # cos(theta_o)：
        # 乘以余弦因子（Lambert’s cosine law），确保遵守能量守恒。
        value = (S_R + S_TT) * dr.cos(theta_o)


        return value & active

    # pdf function
    # BSDF（双向散射分布函数）的概率密度函数（PDF）计算，用于评估一个给定的出射方向 wo 的采样概率。
    # 该 PDF 在重要性采样（importance sampling）中起着关键作用，影响路径追踪算法的收敛效率。
    def pdf(self, ctx, si, wo, active):
        # 计算入射角和出射角
        theta_i, phi_i = self.to_spherical(si.wi)
        theta_o, phi_o = self.to_spherical(wo)
        #  计算相对方位角，用于透射方向 PDF 的计算。
        phi_d = self.regularize(phi_o - self.regularize(phi_i + dr.pi))


        # 计算反射率和透射率
        # C_R 和 C_TT 分别是材质的反射率和透射率。
        # att_R 是菲涅尔反射项，表示在 θ_i 角度下，多少光会被反射。
        # att_TT 是透射项，表示剩余部分被透射。
        C_R  = self.m_reflectance.eval(si, active)
        C_TT = self.m_transmittance.eval(si, active)
        att_R = self.fresnel(C_R, theta_i)
        att_TT = (1.0 - att_R) * C_TT
        att    = att_R + att_TT

        # Compute ratio of R to TT accross 3 channels
        # 计算不同颜色通道上的反射和透射比例
        # att 是 RGB 颜色通道上的总光量，计算其每个通道的占比：
        # ratio_r: 红色通道的比重。
        # ratio_g: 绿色通道的比重。
        # ratio_b: 蓝色通道的比重（从 1 - ratio_r - ratio_g 得出）。
        ratio_r = att[0]
        ratio_g = att[1]
        ratio_b = att[2]
        ratio_sum = ratio_r + ratio_g + ratio_b

        ratio_r = ratio_r / ratio_sum
        ratio_g = ratio_g / ratio_sum
        ratio_b = 1.0 - ratio_r - ratio_g


        # 计算反射和透射的总权重
        ratio_R  = att_R / (att_R + att_TT)
        ratio_TT = 1.0 - ratio_R
        # p_R: 加权计算最终反射比例。
        # p_TT: 加权计算最终透射比例。
        p_R  = ratio_r * ratio_R[0]  + ratio_g * ratio_R[1]  + ratio_b * ratio_R[2]
        p_TT = ratio_r * ratio_TT[0] + ratio_g * ratio_TT[1] + ratio_b * ratio_TT[2]

        # 计算反射和透射的 PDF
        pdf_R  = self.uct_gaussian_pdf(theta_o, -theta_i, self.r_longwidth, -0.5 * dr.pi, 0.5 * dr.pi) * self.uniform_pdf(phi_d, -dr.pi, dr.pi)
        pdf_TT = self.uct_gaussian_pdf(theta_o, -theta_i, self.tt_longwidth, -0.5 * dr.pi, 0.5 * dr.pi) * self.uct_gaussian_pdf(phi_d, 0.0, self.tt_aziwidth, -dr.pi, dr.pi)

        # 计算最终的 PDF
        # 除以 cos(theta_o)，用于将球面 PDF 变换到平面 PDF（多用于路径追踪算法）。
        _pdf = (p_R * pdf_R + p_TT * pdf_TT) / dr.cos(theta_o)
        return dr.select(active, _pdf, 0.0)
    
    def to_string(self):
        return ('Khungurn')

# 向 Mitsuba 渲染器注册了一个名为 khungurn 的 BSDF 类型。
# 这意味着，当你在场景中使用 khungurn 作为材质类型时，Mitsuba 会识别这个名称，
# 并通过 lambda props: Khungurn(props) 这个函数来实例化并返回 Khungurn 类的一个实例
mi.register_bsdf("khungurn", lambda props: Khungurn(props))