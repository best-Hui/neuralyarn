from utils.np import to_marschner, uct_gaussian_pdf, from_spherical

import numpy as np
import scipy as sp


def compute_kappa_R(rdm_t, rdm_r, rdm_m, sa):
    energy_t = np.sum(rdm_t.mean(axis=-1) * sa[None, None, :, :])
    energy_r = np.sum(rdm_r.mean(axis=-1) * sa[None, None, :, :])
    energy_m = np.sum(rdm_m.mean(axis=-1) * sa[None, None, :, :])

    kappa_R = energy_r / (energy_t + energy_r + energy_m)
    return kappa_R

def prepare_data(S, x, sa, parameters):
    S /= np.sum(S * sa, axis=(2, 3))[:, :, None, None]

    THT_I, PHI_I, THT_O, PHI_O = x.transpose(4, 0, 1, 2, 3)

    # Convert to cartesian
    WI = np.stack(from_spherical(THT_I, PHI_I), axis=-1)
    WO = np.stack(from_spherical(THT_O, PHI_O), axis=-1)

    # Create fiber frame
    angle = -np.pi * parameters['Alpha']
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), +np.cos(angle), 0.0],
        [0.0,           0.0,            1.0]
    ])
    frame_x = rotation @ [1.0, 0.0, 0.0]
    frame_y = rotation @ [0.0, 1.0, 0.0]
    frame_z = rotation @ [0.0, 0.0, 1.0]

    # Convert to fiber frame
    WI[:, :, :, :, 0] = np.dot(WI, frame_x)
    WI[:, :, :, :, 1] = np.dot(WI, frame_y)
    WI[:, :, :, :, 2] = np.dot(WI, frame_z)
    WO[:, :, :, :, 0] = np.dot(WO, frame_x)
    WO[:, :, :, :, 1] = np.dot(WO, frame_y)
    WO[:, :, :, :, 2] = np.dot(WO, frame_z)

    # Convert to Marschner
    THT_I, PHI_I = to_marschner(*np.transpose(WI, [4, 0, 1, 2, 3]))
    THT_O, PHI_O = to_marschner(*np.transpose(WO, [4, 0, 1, 2, 3]))

    return (THT_I, PHI_I, THT_O, PHI_O), S

def pdf(X, p, beta, gamma):
    tht_i, phi_i, tht_o, phi_o = X

    M = uct_gaussian_pdf(tht_o, -tht_i, beta, -np.pi/2, +np.pi/2)
    N = uct_gaussian_pdf(phi_o, 0.0, gamma, -np.pi, +np.pi)
    return p * M * N / np.cos(tht_o) + (1-p) * (1/(4*np.pi))

def fit_pdf_m(input, label):
    input = [data.ravel() for data in input]
    label = label.ravel()

    popt, pcov = sp.optimize.curve_fit(pdf, input, label, p0=[0.5, 1, 1], bounds=[(0, 0, 0), (1, 5, 10)])

    return popt

def fit_pdf(rdm_t, rdm_r, rdm_m, x, sa, parameters):
    kappa_M, beta_M, gamma_M = fit_pdf_m(*prepare_data(rdm_m.mean(axis=-1), x, sa, parameters))
    kappa_R = compute_kappa_R(rdm_t, rdm_r, rdm_m, sa)

    return kappa_R, beta_M, gamma_M, kappa_M