from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)

from mitsuba import Transform4f as T

import numpy as np

import bsdf.khungurn as khungurn

# 纱线
class NeuralYarn(mi.BSDF):
    def __init__(self, props: mi.Properties):
        mi.BSDF.__init__(self, props)
        self.khungurn = props['khungurn']
        self.alpha = props['alpha']

        self.kappa_R = props['kappa_R']
        self.beta_M = props['beta_M']
        self.gamma_M = props['gamma_M']
        self.kappa_M = props['kappa_M']

        self.theta = dr.atan(dr.pi * self.alpha)

        self.model_m = None
        self.model_t = None

        self.m_components.clear()
        for i in range(self.khungurn.component_count()):
            self.m_components.append(self.khungurn.flags(i))
        self.m_flags = self.khungurn.flags()

    # -----------------------------------------------------------------
    # Utilty Functions
    # -----------------------------------------------------------------
    def to_spherical(self, vector):
        theta = dr.safe_asin(vector[0])
        phi = dr.atan2(vector[2], vector[1])
        return theta, phi

    def to_cartesian(self, theta, phi):
        return mi.Vector3f(dr.sin(theta), dr.cos(theta) * dr.cos(phi), dr.cos(theta) * dr.sin(phi))

    # general and coordinates helper functions
    def fresnel(self, C_R, theta_i):
        return C_R + (1.0 - C_R) * dr.power(1.0 - dr.abs(dr.cos(theta_i)), 5)
    
    def to_spherical(self, vector):
        theta = dr.safe_asin(vector[0])
        phi = dr.atan2(vector[2], vector[1])
        return theta, phi

    def to_cartesian(self, theta, phi):
        return mi.Vector3f(dr.sin(theta), dr.cos(theta) * dr.cos(phi), dr.cos(theta) * dr.sin(phi))

    def safe_divide(self, numerator, denominator):
        return dr.select(dr.eq(denominator, 0.0), 0.0, numerator / denominator)
    
    def f_mod(self, a, b):
        remainder = a - b * dr.floor(a / b)
        negative_mask = remainder < 0
        remainder[negative_mask] += dr.abs(b)
        return remainder
    
    def regularize(self, angle):
        angle = self.f_mod(angle, 2 * dr.pi)

        more_than = angle > +dr.pi
        less_than = angle < -dr.pi

        angle[more_than] = angle[more_than] - dr.two_pi
        angle[less_than] = angle[less_than] + dr.two_pi
        
        return angle

    # important sampling pdf helper functions#
    def uniform_pdf(self, x, a, b):
        return 1.0 / (b - a)
    
    def std_gaussian_cdf(self, x):
        return 0.5 * (1.0 + dr.erf(x / dr.sqrt(2.0)))
    
    def gaussian_cdf(self, x, mu, sigma):
        return self.std_gaussian_cdf(x)
    
    def gaussian_integral(self, mu, sigma, a, b):
        return self.gaussian_cdf(b, mu, sigma) - self.gaussian_cdf(a, mu, sigma)

    def gaussian_pdf(self, x, mu, sigma):
        t1 = 1.0 / (dr.sqrt(2.0 * dr.pi * dr.power(sigma, 2)))
        t2 = dr.exp(-dr.power(x - mu, 2)/(2.0 * dr.power(sigma, 2)))
        return t1 * t2
    
    def uct_gaussian_pdf(self, x, mu, sigma, a, b):
        t1 = self.gaussian_pdf(x, mu, sigma)
        t2 = (1.0 - self.gaussian_integral(mu, sigma, a, b)) / (b - a)
        return t1 + t2
    
    # important sampling sample helper functions
    def uniform_sample(self, sample, a, b):
        return sample * (b - a) + a
    
    def std_gaussian_sample(self, sample):
        return dr.sqrt(2.0) * dr.erf(2.0 * sample - 1.0)
    
    def gaussian_sample(self, sample, mu, sigma):
        return self.std_gaussian_sample(sample) * sigma + mu
    
    def uct_gaussian_sample(self, sample, mu, sigma, a, b):
        s_g = self.gaussian_sample(sample, mu, sigma)
        s_u = self.uniform_sample(mi.PCG32(dr.width(sample), sample*12998409.3423098).next_float32(), a, b)
        inside = (a < s_g) & (s_g < b)
        result = dr.select(inside, s_g, s_u)
        return result
    
    #eval function helper functions
    def G_integral(self, theta, mean, std):
        std2 = dr.power(std, 2)

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

        result = (A/2) * dr.erf((theta - mean) / (dr.sqrt(2.0) * std)) - std2 * B * self.gaussian_pdf(theta, mean, std)
        return result

    def G(self, mean, std):
        result = self.G_integral(dr.pi / 2.0, mean, std) - self.G_integral(-dr.pi/2.0, mean, std)
        return result
    
    def cos2_normalized_gaussian(self, theta, mean, std):
        result = self.safe_divide(self.gaussian_pdf(theta, mean, std), self.G(mean, std))
        return result
    
    # -----------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------
    # 对 3D 向量 vector 进行绕 Z 轴旋转，旋转角度为 angle
    def twist_frame(self, vector, angle):
        return mi.Vector3f(
            vector.x * dr.cos(angle) - vector.y * dr.sin(angle),
            vector.x * dr.sin(angle) + vector.y * dr.cos(angle),
            vector.z)

    # 根据不同的概率分布函数（PDF）和累积分布函数（CDF）来从不同的“lobes”（假设是光照模型中的多个部分或散射项）中采样，
    # 并根据给定的采样值生成一个新的方向向量 wo
    # 供def sample调用
    def sample_wo(self, wi, sample1, sample2):
        # 旋转入射向量 wi，使用 theta 参数。旋转变换的目的是对坐标系进行变换，通常用于光照模型中改变光线的方向
        _wi = self.twist_frame(wi, self.theta)

        # 计算球面坐标
        theta_i, phi_i = self.to_spherical(wi)

        # 发生透射的概率
        p_T = self.eval_model_t(wi)

        # PDF of lobes
        # 这里定义了四个 PDF 值，分别对应不同的散射成分
        p1 = p_T # 透射
        p2 = (1.0 - p_T) * self.kappa_R # 反射
        p3 = (1.0 - p_T) * (1.0 - self.kappa_R) * self.kappa_M # 多重散射
        p4 = (1.0 - p_T) * (1.0 - self.kappa_R) * (1.0 - self.kappa_M) # 可能是能量损耗(表面的吸收)

        # CDF
        # 计算 CDF 值，表示不同的概率区间，用于在后续的采样中选择对应的散射类型。
        # 透射的概率 p_T（即 p1）在整个流程中占有特殊位置，但它并没有直接影响 CDF 的累加过程。p_T 仅决定了透射和其他散射行为的分配比例。透射并不参与 CDF 累积，而是通过 select_t 来选择是否进行透射。
        # 反射的概率是 (1 - p_T) * self.kappa_R（即 p2）。反射的采样在 CDF 中从 c0 开始，累积到 c1。这就是为什么 c1 = c0 + p2，即反射的区间被直接加入到 CDF 中。
        # c2 是 反射概率的再次累积。你看到 c2 = c1 + p2，这里是为了增加两个反射过程的累计概率。可能是考虑到在某些模型中，反射行为可以有不同的强度或类型，反射被拆分成多个步骤，因此反射的概率被重复加上。
        # 多重散射的概率 p3 累加到 c3，这时 CDF 完全覆盖了反射和多重散射的所有概率。
        # 最后，c4 被设定为 1.0，表示整个过程的累积概率的上限。
        c0 = 0.0
        # c1 = c0 + p2 # 反射
        c1 = c0 + p1 # 透射,我做了改动,原来是c1 = c0 + p2
        c2 = c1 + p2 # 反射
        c3 = c2 + p3 # 多重散射
        c4 = 1.0

        # Sample with lobe CDF
        # 目的是选择采样哪个光的行为
        select_t = (c0 < sample1) & (sample1 < c1)
        select_r = (c1 < sample1) & (sample1 < c2)
        select_m = (c2 < sample1) & (sample1 < c3)
        # 重要性采样中的采样波瓣(lobe)k,采样波瓣 k 不能简单直接等同于表面被吸收的能量，
        # 与能量损耗或未被其他主要散射分量（透射、反射、多重散射）所涵盖的能量情况相关
        select_k = (c3 < sample1) & (sample1 < c4)

        # Sample T
        # wo_t 是 wi 的反向方向，代表透射
        wo_t = -wi

        # Sample R
        # sample_r 使用高斯分布来对 theta_o 和 phi_o 进行采样，生成新的方向 wo_r
        def sample_r(wi):
            theta_i, phi_i = self.to_spherical(wi)
            theta_o = self.uct_gaussian_sample(sample2.x, -theta_i, self.khungurn.r_longwidth, -0.5*dr.pi, 0.5*dr.pi)
            phi_o = self.uniform_sample(sample2.y, -dr.pi, dr.pi) + phi_i
            wo = self.to_cartesian(theta_o, phi_o)
            return wo
    
        _wo = sample_r(_wi)
        wo_r = self.twist_frame(_wo, -self.theta) 
        
        # Sample M
        # sample_m 类似地从高斯分布中采样，并生成新的方向 wo_m
        def sample_m(wi):
            theta_o = self.uct_gaussian_sample(sample2.x, -theta_i, self.beta_M, -0.5*dr.pi, 0.5*dr.pi)
            phi_o = self.uct_gaussian_sample(sample2[1], 0.0, self.gamma_M, dr.pi, dr.pi) + phi_i
            wo = self.to_cartesian(theta_o, phi_o)
            return wo
        
        _wo = sample_m(_wi)
        wo_m = self.twist_frame(_wo, -self.theta)

        # Sample K
        # wo_k 从均匀分布的球面上采样
        wo_k = mi.warp.square_to_uniform_sphere(sample2)

        # 最终，dr.select 根据之前的选择（select_t, select_r, select_m, select_k）从不同的散射类型中选择出对应的方向 wo。
        wo = mi.Vector3f(0, 0, 1)
        wo = dr.select(select_t, wo_t, wo)
        wo = dr.select(select_r, wo_r, wo)
        wo = dr.select(select_m, wo_m, wo)
        wo = dr.select(select_k, wo_k, wo)

        return wo

    #  供def pdf(self, ctx, si, wo, active)调用
    def sample_wo_pdf(self, wi, wo):
        p_T = self.eval_model_t(wi)

        wi = self.twist_frame(wi, self.theta)
        wo = self.twist_frame(wo, self.theta)
    
        theta_i, phi_i = self.to_spherical(wi)
        theta_o, phi_o = self.to_spherical(wo)
        # 计算入射方向和输出方向的方位角差异，并使用 regularize 函数将其规范化，确保差值在合理范围内。
        phi_d = self.regularize(phi_o - phi_i)

        # pdf_R 是针对 R 类型散射（可能是某种反射模型）计算的 PDF。
        # 首先通过高斯分布 uct_gaussian_pdf 计算 theta_o 的 PDF，再通过 uniform_pdf 计算 phi_o 的均匀分布 PDF。最后，除以 dr.cos(theta_o) 进行归一化。
        # 论文公式
        pdf_R = self.uct_gaussian_pdf(theta_o, -theta_i, self.khungurn.r_longwidth, -0.5*dr.pi, +0.5*dr.pi) \
            * self.uniform_pdf(phi_o, -dr.pi, dr.pi) / dr.cos(theta_o)

        # pdf_m1 是基于 M 类型散射的 PDF，通过高斯分布计算 theta_o 和 phi_d 的概率密度
        pdf_m1 = self.uct_gaussian_pdf(theta_o, -theta_i, self.beta_M, -0.5*dr.pi, +0.5*dr.pi) \
            * self.uct_gaussian_pdf(phi_d, 0.0, self.gamma_M, -dr.pi, dr.pi) / dr.cos(theta_o)

        # pdf_m2 是一个常数，表示均匀分布的概率密度，适用于一种均匀散射模型
        pdf_m2 = 1.0 / (4.0 * dr.pi)

        # pdf_M 是混合模型的 PDF，其中 self.kappa_M 是权重因子，控制了 pdf_m1 和 pdf_m2 的混合比例。
        # 论文公式
        pdf_M = self.kappa_M * pdf_m1 + (1.0 - self.kappa_M) * pdf_m2

        # p_R 和 p_M 分别是 R 和 M 散射的权重，它们依赖于模型参数 p_T 和预先定义的常数 kappa_R 和 kappa_M。
        p_R = (1.0 - p_T) * self.kappa_R
        p_M = (1.0 - p_T) * (1.0 - self.kappa_R)

        # 最终的 PDF 是通过将 R 类型散射的 PDF 和 M 类型散射的 PDF 按照权重加权求和得到的
        # 如果 wi 和 wo 的方向相反（即 wi == -wo），则返回一个 PDF 值为 1。
        pdf = p_R*pdf_R + p_M*pdf_M
        pdf = dr.select(dr.all(dr.eq(wi, -wo)), 1.0, pdf)

        return pdf

    # 用于从 BSDF（双向散射分布函数）模型中采样，并返回一个样本和该样本的值。
    # 它通过采样 wo（反射或散射的方向）来进行光线追踪模拟，通常应用于光照模型中的反射、折射等现象
    # 渲染管线中,mitsuba会自动调用这个函数,sample是自定义材质必须实现的方法
    def sample(self, ctx, si, sample1, sample2, active):
        # BSDFSample3f 是一个对象，通常用于存储一个 BSDF 样本的相关信息，比如散射方向（wo）、采样的概率密度函数（pdf）和其他散射参数
        bs = mi.BSDFSample3f()
        # bs.wo = mi.warp.square_to_uniform_sphere(sample2)
        # 这里调用了 sample_wo 函数来生成反射方向 wo。
        bs.wo = self.sample_wo(si.wi, sample1, sample2)

        # self.pdf(ctx, si, bs.wo, active)调用sample_wo_pdf,计算wi wo的pdf
        bs.pdf = self.pdf(ctx, si, bs.wo, active)

        # sampled_component 和 sampled_type 存储了此次采样的相关信息。
        # 这里，sampled_type 设置为 mi.BSDFFlags.DiffuseReflection，表示此次采样是针对漫反射反射的采样（Diffuse Reflection）
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type      = mi.UInt32(+mi.BSDFFlags.DiffuseReflection)

        # eta 表示折射率，通常用于模拟折射和反射过程。
        # 在这个例子中，它被设置为 1.0，表示没有折射（在某些情况下，这可能代表光的传播没有折射现象发生）
        bs.eta = mi.Float(1.0)

        # eval 函数用于评估给定反射方向 wo 下的 BSDF 值。
        # 这里，eval 返回的值通常是反射或散射的光强度，单位是每个单位立体角的辐射度。通过将其除以采样的 PDF 值 bs.pdf，得到该方向的正确贡献值
        # 重要性采样,bsdf除以pdf
        value = self.eval(ctx, si, bs.wo, active) / bs.pdf
        return (bs, value)

    # 是根据给定的入射方向 wi 和反射方向 wo 评估一个 BSDF（双向散射分布函数）值
    # 供def sample调用
    def eval(self, ctx, si, wo, active):
        # 计算变换角度 m_angle
        m_angle = dr.atan(dr.pi * self.alpha)

        # 将 si.wi（入射光的方向）存储在 wi 中
        wi = mi.Vector3f(si.wi)

        # 使用旋转矩阵对入射方向 wi 进行旋转。旋转角度为 m_angle，这通常是为了模拟某些特定的表面粗糙度或变化。旋转后的入射方向保存在 wi_ 中
        wi_ = mi.Vector3f(wi.x * dr.cos(m_angle) - wi.y * dr.sin(m_angle),
                          wi.x * dr.sin(m_angle) + wi.y * dr.cos(m_angle),
                          wi.z)
        # 旋转输出方向 wo
        wo_ = mi.Vector3f(wo.x * dr.cos(m_angle) - wo.y * dr.sin(m_angle),
                          wo.x * dr.sin(m_angle) + wo.y * dr.cos(m_angle),
                          wo.z)

        # 将旋转后的入射方向 wi_ 赋给 si.wi，可能是为了将表面信息（例如法向量和反射模型）更新为旋转后的方向。
        si.wi = wi_

        # 这里通过 dr.select 判断反射方向 wo 是否与表面的法向量 si.n 大于零，如果是，则调用 khungurn.eval 评估反射模型的贡献 s_r，
        # 否则 s_r 为 0。
        # s_r 进一步乘以 wo.z，表示仅在 wo 的 z 方向（通常为法向方向）上才有贡献。这可能是为了保证光线的能量守恒
        # 纱线bsdf的反射项等于纤维的bsdf,包含纤维的透射和反射分量,见论文说明
        s_r = dr.select(dr.dot(wo, si.n) > 0.0, self.khungurn.eval(ctx, si, wo_, active), 0.0)
        s_r = s_r * wo.z
        # 恢复原始入射方向 wi
        si.wi = wi

        # 计算散射模型 m 对于给定入射方向 si.wi 和反射方向 wo 的贡献
        # 论文提到的S散射函数,实际上是bsdf值
        s_m = self.eval_model_m(si.wi, wo)
        
        return dr.maximum(s_r + s_m, 0.0)
    
    def pdf(self, ctx, si, wo, active):
        return self.sample_wo_pdf(si.wi, wo)

    # 输入wi 进行透射的预测
    def eval_model_t(self, wi):
        nn_in = dr.zeros(mi.TensorXf, shape=[3, dr.width(wi)])
        nn_in[0] = wi.x
        nn_in[1] = wi.y
        nn_in[2] = wi.z

        nn_out = self.model_t.forward(nn_in)

        value = nn_out.array

        return value

    # 输入wi和wo进行多重散射的预测
    def eval_model_m(self, wi, wo):
        nn_in = dr.zeros(mi.TensorXf, shape=[6, dr.width(wi)])
        nn_in[0] = wi.x
        nn_in[1] = wi.y
        nn_in[2] = wi.z
        nn_in[3] = wo.x
        nn_in[4] = wo.y
        nn_in[5] = wo.z

        nn_out = self.model_m.forward(nn_in)

        value = dr.unravel(mi.Color3f, dr.ravel(nn_out), order='C')

        return value

    # 用于获取与当前表面、材质或模型相关的某些特征值
    def eval_attribute(self, name, si, active):
        return self.khungurn.eval_attribute(name, si, active)

    # 该函数的主体是一个空函数，通常表示该方法可能需要被覆盖或扩展以实现特定的功能。它接受一个回调函数 callback，但是没有实现任何具体的操作
    def traverse(self, callback):
        pass

    # 这个函数的作用是响应模型参数的变化。它接受一组 keys，可能表示已更改的参数，并可以在此基础上执行一些更新操作
    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return ('NeuralYarn')


    # 这是一个工厂函数，负责创建一个 neuralyarn 对象
    def create(parameters, model_m, model_t, kappa_R, beta_M, gamma_M, kappa_M):
        # 纤维
        khungurn = mi.load_dict({
            "type": "khungurn",
            "reflectance": {
                "type": "rgb",
                "value": list(parameters[['R_r', 'R_g', 'R_b']]),
            },
            "transmittance": {
                "type": "rgb",
                "value": [0.0, 0.0, 0.0],
            },
            "r_longwidth": parameters['R_long'],
            "tt_longwidth": parameters['TT_long'],
            "tt_aziwidth": parameters['TT_azim'],
        })
        # 纱线
        neuralyarn = mi.load_dict({
        "type": "neuralyarn",
            "khungurn": khungurn,
            "alpha": parameters['Alpha'],
            "kappa_R": float(kappa_R),
            "beta_M": float(beta_M),
            "gamma_M": float(gamma_M),
            "kappa_M": float(kappa_M),
        })
        neuralyarn.model_m = model_m
        neuralyarn.model_t = model_t

        return neuralyarn

# 将 neuralyarn 注册为一种 BSDF 类型，允许在渲染引擎中使用它。
# 通过 mi.register_bsdf 注册后，渲染引擎可以在材质创建过程中识别并使用 neuralyarn
mi.register_bsdf("neuralyarn", lambda props: NeuralYarn(props))