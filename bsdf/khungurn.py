from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)

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

    def M_R(self, theta_i, theta_o):
        result = self.cos2_normalized_gaussian(theta_o, -theta_i, self.r_longwidth)
        return result
    
    def N_R(self):
        result = 1.0 / (2.0 * dr.pi)
        return result
    
    def M_TT(self, theta_i, theta_o):
        result = self.cos2_normalized_gaussian(theta_o, -theta_i, self.tt_longwidth)
        return result
    
    def N_TT(self, phi_i, phi_o):
        phi_d = self.regularize(phi_o - self.regularize(phi_i + dr.pi))
        result = self.uct_gaussian_pdf(phi_d, 0.0, self.tt_aziwidth, -dr.pi, dr.pi)
        return result

    # sample function with importance sampling
    def sample(self, ctx, si, sample1, sample2, active):
        sampler = mi.PCG32(1, sample1 * 20938 + sample2[0] * 90654 + sample2[1] * 43092)
        bs = mi.BSDFSample3f()

        # Compute Fresnel terms
        theta_i, phi_i = self.to_spherical(si.wi)
        C_R  = self.m_reflectance.eval(si, active)
        C_TT = self.m_transmittance.eval(si, active)
        att_R = self.fresnel(C_R, theta_i)
        att_TT = (1 - att_R) * C_TT
        att    = att_R + att_TT

        # Compute ratio of R to TT accross 3 channels
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
        selected_R = (sample1 < mi.luminance(ratio_R))

        theta_o = dr.select(selected_R, self.uct_gaussian_sample(sample2[0], -theta_i, self.r_longwidth,  -0.5*dr.pi, 0.5*dr.pi), self.uct_gaussian_sample(sample2[0], -theta_i, self.tt_longwidth, -0.5*dr.pi, 0.5*dr.pi))

        phi_d = dr.select(selected_R, self.uniform_sample(sample2[1], -dr.pi, dr.pi), self.uct_gaussian_sample(sample2[1], 0.0, self.tt_aziwidth, -dr.pi, dr.pi))

        phi_o = (phi_i + dr.pi) + phi_d

        p_R  = ratio_r * ratio_R[0]  + ratio_g * ratio_R[1]  + ratio_b * ratio_R[2]
        p_TT = ratio_r * ratio_TT[0] + ratio_g * ratio_TT[1] + ratio_b * ratio_TT[2]

        pdf_R  = self.uct_gaussian_pdf(theta_o, -theta_i, self.r_longwidth, -0.5*dr.pi, 0.5*dr.pi) * self.uniform_pdf(phi_d, -dr.pi, dr.pi)

        pdf_TT = self.uct_gaussian_pdf(theta_o, -theta_i, self.tt_longwidth, -0.5*dr.pi, 0.5*dr.pi) * self.uct_gaussian_pdf(phi_d, 0.0, self.tt_aziwidth, -dr.pi, dr.pi)

        _pdf = p_R * pdf_R + p_TT * pdf_TT
        

        # Fill up the BSDFSample struct
        bs.pdf = _pdf / dr.cos(theta_o)
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type      = mi.UInt32(mi.BSDFFlags.Glossy)
        bs.wo = self.to_cartesian(theta_o, phi_o)
        bs.eta = 1.0
    
        _eval = self.eval(ctx, si, bs.wo, active)
        weight = (_eval / _pdf) * dr.cos(theta_o)

        return bs, weight & (active & (bs.pdf > 0.0))


    # eval function
    def eval(self, ctx, si, wo, active):
        theta_i, phi_i = self.to_spherical(si.wi)
        theta_o, phi_o = self.to_spherical(wo)

        C_R  = self.m_reflectance.eval(si, active)
        C_TT = self.m_transmittance.eval(si, active)
        
        F_R = self.fresnel(C_R, theta_i)
        S_R = F_R * self.M_R(theta_i, theta_o) * self.N_R()
        S_TT = (1.0 - F_R) * C_TT * self.M_TT(theta_i, theta_o) * self.N_TT(phi_i, phi_o)

        value = (S_R + S_TT) * dr.cos(theta_o)

        return value & active

    # pdf function
    def pdf(self, ctx, si, wo, active):
        theta_i, phi_i = self.to_spherical(si.wi)
        theta_o, phi_o = self.to_spherical(wo)
        phi_d = self.regularize(phi_o - self.regularize(phi_i + dr.pi))

        C_R  = self.m_reflectance.eval(si, active)
        C_TT = self.m_transmittance.eval(si, active)
        att_R = self.fresnel(C_R, theta_i)
        att_TT = (1.0 - att_R) * C_TT
        att    = att_R + att_TT

        # Compute ratio of R to TT accross 3 channels
        ratio_r = att[0]
        ratio_g = att[1]
        ratio_b = att[2]
        ratio_sum = ratio_r + ratio_g + ratio_b

        ratio_r = ratio_r / ratio_sum
        ratio_g = ratio_g / ratio_sum
        ratio_b = 1.0 - ratio_r - ratio_g

        ratio_R  = att_R / (att_R + att_TT)
        ratio_TT = 1.0 - ratio_R

        p_R  = ratio_r * ratio_R[0]  + ratio_g * ratio_R[1]  + ratio_b * ratio_R[2]
        p_TT = ratio_r * ratio_TT[0] + ratio_g * ratio_TT[1] + ratio_b * ratio_TT[2]

        pdf_R  = self.uct_gaussian_pdf(theta_o, -theta_i, self.r_longwidth, -0.5 * dr.pi, 0.5 * dr.pi) * self.uniform_pdf(phi_d, -dr.pi, dr.pi)
        pdf_TT = self.uct_gaussian_pdf(theta_o, -theta_i, self.tt_longwidth, -0.5 * dr.pi, 0.5 * dr.pi) * self.uct_gaussian_pdf(phi_d, 0.0, self.tt_aziwidth, -dr.pi, dr.pi)

        _pdf = (p_R * pdf_R + p_TT * pdf_TT) / dr.cos(theta_o)
        return dr.select(active, _pdf, 0.0)
    
    def to_string(self):
        return ('Khungurn')
    
mi.register_bsdf("khungurn", lambda props: Khungurn(props))