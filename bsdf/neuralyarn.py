from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)

from mitsuba import Transform4f as T

import numpy as np

import bsdf.khungurn as khungurn

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
    def twist_frame(self, vector, angle):
        return mi.Vector3f(
            vector.x * dr.cos(angle) - vector.y * dr.sin(angle),
            vector.x * dr.sin(angle) + vector.y * dr.cos(angle),
            vector.z)
    
    def sample_wo(self, wi, sample1, sample2):
        _wi = self.twist_frame(wi, self.theta)

        theta_i, phi_i = self.to_spherical(wi)

        p_T = self.eval_model_t(wi)

        # PDF of lobes
        p1 = p_T
        p2 = (1.0 - p_T) * self.kappa_R
        p3 = (1.0 - p_T) * (1.0 - self.kappa_R) * self.kappa_M
        p4 = (1.0 - p_T) * (1.0 - self.kappa_R) * (1.0 - self.kappa_M)

        # CDF
        c0 = 0.0
        c1 = c0 + p2
        c2 = c1 + p2
        c3 = c2 + p3
        c4 = 1.0

        # Sample with lobe CDF
        select_t = (c0 < sample1) & (sample1 < c1)
        select_r = (c1 < sample1) & (sample1 < c2)
        select_m = (c2 < sample1) & (sample1 < c3)
        select_k = (c3 < sample1) & (sample1 < c4)

        # Sample T
        wo_t = -wi

        # Sample R
        def sample_r(wi):
            theta_i, phi_i = self.to_spherical(wi)
            theta_o = self.uct_gaussian_sample(sample2.x, -theta_i, self.khungurn.r_longwidth, -0.5*dr.pi, 0.5*dr.pi)
            phi_o = self.uniform_sample(sample2.y, -dr.pi, dr.pi) + phi_i
            wo = self.to_cartesian(theta_o, phi_o)
            return wo
    
        _wo = sample_r(_wi)
        wo_r = self.twist_frame(_wo, -self.theta) 
        
        # Sample M
        def sample_m(wi):
            theta_o = self.uct_gaussian_sample(sample2.x, -theta_i, self.beta_M, -0.5*dr.pi, 0.5*dr.pi)
            phi_o = self.uct_gaussian_sample(sample2[1], 0.0, self.gamma_M, dr.pi, dr.pi) + phi_i
            wo = self.to_cartesian(theta_o, phi_o)
            return wo
        
        _wo = sample_m(_wi)
        wo_m = self.twist_frame(_wo, -self.theta)

        # Sample K
        wo_k = mi.warp.square_to_uniform_sphere(sample2)

        wo = mi.Vector3f(0, 0, 1)
        wo = dr.select(select_t, wo_t, wo)
        wo = dr.select(select_r, wo_r, wo)
        wo = dr.select(select_m, wo_m, wo)
        wo = dr.select(select_k, wo_k, wo)

        return wo

    def sample_wo_pdf(self, wi, wo):
        p_T = self.eval_model_t(wi)

        wi = self.twist_frame(wi, self.theta)
        wo = self.twist_frame(wo, self.theta)
    
        theta_i, phi_i = self.to_spherical(wi)
        theta_o, phi_o = self.to_spherical(wo)
        phi_d = self.regularize(phi_o - phi_i)

        pdf_R = self.uct_gaussian_pdf(theta_o, -theta_i, self.khungurn.r_longwidth, -0.5*dr.pi, +0.5*dr.pi) \
            * self.uniform_pdf(phi_o, -dr.pi, dr.pi) / dr.cos(theta_o)

        pdf_m1 = self.uct_gaussian_pdf(theta_o, -theta_i, self.beta_M, -0.5*dr.pi, +0.5*dr.pi) \
            * self.uct_gaussian_pdf(phi_d, 0.0, self.gamma_M, -dr.pi, dr.pi) / dr.cos(theta_o)

        pdf_m2 = 1.0 / (4.0 * dr.pi)

        pdf_M = self.kappa_M * pdf_m1 + (1.0 - self.kappa_M) * pdf_m2

        p_R = (1.0 - p_T) * self.kappa_R
        p_M = (1.0 - p_T) * (1.0 - self.kappa_R)

        pdf = p_R*pdf_R + p_M*pdf_M
        pdf = dr.select(dr.all(dr.eq(wi, -wo)), 1.0, pdf)

        return pdf

    def sample(self, ctx, si, sample1, sample2, active):
        bs = mi.BSDFSample3f()
        # bs.wo = mi.warp.square_to_uniform_sphere(sample2)
        bs.wo = self.sample_wo(si.wi, sample1, sample2)
        bs.pdf = self.pdf(ctx, si, bs.wo, active)
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type      = mi.UInt32(+mi.BSDFFlags.DiffuseReflection)
        bs.eta = mi.Float(1.0)

        value = self.eval(ctx, si, bs.wo, active) / bs.pdf
        return (bs, value)
    
    def eval(self, ctx, si, wo, active):
        m_angle = dr.atan(dr.pi * self.alpha)

        wi = mi.Vector3f(si.wi)

        wi_ = mi.Vector3f(wi.x * dr.cos(m_angle) - wi.y * dr.sin(m_angle),
                          wi.x * dr.sin(m_angle) + wi.y * dr.cos(m_angle),
                          wi.z)
        
        wo_ = mi.Vector3f(wo.x * dr.cos(m_angle) - wo.y * dr.sin(m_angle),
                          wo.x * dr.sin(m_angle) + wo.y * dr.cos(m_angle),
                          wo.z)

        si.wi = wi_
        s_r = dr.select(dr.dot(wo, si.n) > 0.0, self.khungurn.eval(ctx, si, wo_, active), 0.0)
        s_r = s_r * wo.z
        si.wi = wi

        s_m = self.eval_model_m(si.wi, wo)
        
        return dr.maximum(s_r + s_m, 0.0)
    
    def pdf(self, ctx, si, wo, active):
        return self.sample_wo_pdf(si.wi, wo)
    
    def eval_model_t(self, wi):
        nn_in = dr.zeros(mi.TensorXf, shape=[3, dr.width(wi)])
        nn_in[0] = wi.x
        nn_in[1] = wi.y
        nn_in[2] = wi.z

        nn_out = self.model_t.forward(nn_in)

        value = nn_out.array

        return value
    
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
    
    def eval_attribute(self, name, si, active):
        return self.khungurn.eval_attribute(name, si, active)
    
    def traverse(self, callback):
        pass

    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return ('NeuralYarn')
    
    def create(parameters, model_m, model_t, kappa_R, beta_M, gamma_M, kappa_M):
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
    
mi.register_bsdf("neuralyarn", lambda props: NeuralYarn(props))