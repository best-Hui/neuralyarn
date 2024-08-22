import numpy as np
import scipy as sp

def to_marschner(x, y, z):
    theta = np.arcsin(x)
    phi = np.arctan2(y, z)
    return theta, phi

def from_marschner(tht, phi):
    x = np.sin(tht)
    y = np.cos(tht) * np.sin(phi)
    z = np.cos(tht) * np.cos(phi)
    return x, y, z

def from_spherical(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def to_spherical(x, y, z):
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    return theta, phi

def regularize_angle(theta):
    theta = np.fmod(theta, 2 * np.pi)
    more_than = theta > np.pi
    less_than = theta < -np.pi
    theta[more_than] = theta[more_than] - (2 * np.pi)
    theta[less_than] = theta[less_than] + (2 * np.pi)
    return theta

# Adapted from J. Olivares et al 2018 J. Phys.: Conf. Ser. 1043 012003
# https://iopscience.iop.org/article/10.1088/1742-6596/1043/1/012003/pdf
def i0(x):
    x2 = x * x
    t1 = np.cosh(x) / np.power((1 + np.power(1/2, 2) * x2), 1/4)
    t2 = (1 + 0.24273 * x2) / (1 + 0.43023 * x2)
    return t1 * t2

# https://en.wikipedia.org/wiki/Von_Mises_distribution
def vonmises(x, mu, kappa):
    t1 = np.exp(kappa * np.cos(x - mu))
    t2 = 2 * np.pi * i0(kappa)
    return t1 / t2

def std_gaussian_cdf(z):
    return 0.5 * (1 + sp.special.erf(z / np.sqrt(2)))

def gaussian_cdf(x, mu, sigma):
    return std_gaussian_cdf((x - mu) / sigma)

def gaussian_integral(mu, sigma, a, b):
    return gaussian_cdf(b, mu, sigma) - gaussian_cdf(a, mu, sigma)

def gaussian_pdf(x, mu, sigma):
    t1 = 1 / (np.sqrt(2 * np.pi * np.power(sigma, 2)))
    t2 = np.exp(-np.power(x - mu, 2)/(2 * np.power(sigma, 2)))
    return t1 * t2

# Uniform Corrected Trunctated Gaussian PDF
def uct_gaussian_pdf(x, mu, sigma, a, b):
    t1 = gaussian_pdf(x, mu, sigma)
    t2 = (1 - gaussian_integral(mu, sigma, a, b)) / (b - a)
    return t1 + t2

def std_gaussian_sample(sample):
    return np.sqrt(2) * sp.special.erfinv(2 * sample - 1)

def gaussian_sample(sample, mu, sigma):
    return std_gaussian_sample(sample) * sigma + mu

def uniform_sample(sample, a, b):
    return sample * (b - a) + a

def uct_gaussian_sample(sample, mu, sigma, a, b):
    s_g = gaussian_sample(sample, mu, sigma)
    s_u = uniform_sample(np.random.rand(len(sample)), a, b)
    
    outside = (s_g < a) | (b < s_g)
    result = np.where(outside, s_u, s_g)

    return result

def I(x, mu, sigma):
    sigma2 = sigma**2

    p = np.empty(9)
    b = np.empty(8)

    p[8] = 0.002439
    p[7] = 0.0
    p[6] = -0.04301
    p[5] = 0.0
    p[4] = 0.3322
    p[3] = 0.0
    p[2] = -0.999745
    p[1] = 0.0
    p[0] = 1.0001

    b[7] = p[8]
    b[6] = p[7] + mu * b[7]
    b[5] = p[6] + mu * b[6] + 7 * sigma2 * b[7]
    b[4] = p[5] + mu * b[5] + 6 * sigma2 * b[6]
    b[3] = p[4] + mu * b[4] + 5 * sigma2 * b[5]
    b[2] = p[3] + mu * b[3] + 4 * sigma2 * b[4]
    b[1] = p[2] + mu * b[2] + 3 * sigma2 * b[3]
    b[0] = p[1] + mu * b[1] + 2 * sigma2 * b[2]

    A = p[0] + mu * b[0] + sigma2 * b[1]

    B = 0
    for i in range(len(b)):
        B += b[i] * x**i

    t1 = (A / 2) * sp.special.erf((x - mu)/(np.sqrt(2) * sigma))
    t2 = sigma2 * B * gaussian_pdf(x, mu, sigma)
    return t1 - t2

def G(mu, sigma):
    return I(np.pi/2, mu, sigma) - I(-np.pi/2, mu, sigma)

def cos2_normalized_gaussian(x, mu, sigma):
    return gaussian_pdf(x, mu, sigma) / G(mu, sigma)