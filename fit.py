from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)

from mitsuba import Transform4f as T
from mitsuba import ScalarTransform4f as sT

from config.parameters import get_fiber_parameters

from bsdf.neuralyarn import NeuralYarn

from network.model import Model_M, Model_T
from network.wrapper import MiModelWrapper

from utils.pdf import fit_pdf
from utils.rdm import compute_rdm
from utils.torch import cart2sph
from utils.np import to_marschner, uct_gaussian_pdf, from_spherical
from utils.geometry import create_single_yarn

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import argparse
import os

parser = argparse.ArgumentParser(description="Fitting the RDM")

parser.add_argument("--checkpoint_name", help="Checkpoint name to store outputs")
parser.add_argument("--yarn_name", help="Name of the yarn defined in config/parameters.py")
parser.add_argument("--epochs_m", type=int, help="Number of epochs to train for model_m", default=25000)
parser.add_argument("--epochs_t", type=int,  help="Number of epochs to train for model_t", default=25000)
parser.add_argument("--lr_m", type=float,  help="Learning rate for model_m", default=0.005)
parser.add_argument("--lr_t", type=float, help="Learning rate for model_t", default=0.002)

args = parser.parse_args()

output_dir = os.path.join('checkpoints/', args.checkpoint_name)

# Load rdm data
print('Loading RDM data from', output_dir)
data = np.load(os.path.join(output_dir, 'rdm.npz'))

rdm_t = data['rdm_t']
rdm_r = data['rdm_r']
rdm_m = data['rdm_m']
count_t = data['count_t']
count_r = data['count_r']
count_m = data['count_m']
x = data['x']
sa = data['sa']

print('Training Model M')
model_m = Model_M().to(device)

n_epochs = args.epochs_m
print_every = 100

optimizer = torch.optim.Adam(model_m.parameters(), lr=args.lr_m)

nn_in = torch.tensor(x).reshape(-1, 4).float().to(device)
nn_in = torch.cat([cart2sph(*nn_in[:, [0, 1]].T), cart2sph(*nn_in[:, [2, 3]].T)], dim=-1).float().to(device)
nn_true = torch.tensor(rdm_m).reshape(-1, 3).float().to(device)

criterion = nn.MSELoss()

history = []

for epoch in range(n_epochs):
    # Zero the gradients
    optimizer.zero_grad()
    
    nn_pred = model_m(nn_in)
    
    loss = criterion(nn_pred, nn_true)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % print_every == 0:
        history.append(loss.item())
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.7f}, Log Loss: {np.log(loss.item()):.7f}')

model_m.eval()

print('Training Model T')
model_t = Model_T().to(device)

n_epochs = args.epochs_t
print_every = 100

optimizer = torch.optim.Adam(model_t.parameters(), lr=args.lr_t)

nn_in = torch.tensor(x)[:, :, 0, 0, :2].reshape(-1, 2).float().to(device)
nn_in = cart2sph(*nn_in.T).float().to(device)
nn_true = count_t / (count_t + count_r + count_m)
nn_true = torch.tensor(nn_true).reshape(-1, 1).to(device)

criterion = nn.MSELoss()

for epoch in range(n_epochs):
    # Zero the gradients
    optimizer.zero_grad()
    
    nn_pred = model_t(nn_in)
    
    loss = criterion(nn_pred, nn_true)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % print_every == 0:
        history.append(loss.item())
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.7f}, Log Loss: {np.log(loss.item()):.7f}')

model_t.eval()

print('Fitting PDF')
parameters = get_fiber_parameters(args.yarn_name)
kappa_R, beta_M, gamma_M, kappa_M = fit_pdf(rdm_t, rdm_r, rdm_m, x, sa, parameters)
print('Parameters:', kappa_R, beta_M, gamma_M, kappa_M)

print('Saving to', output_dir)
torch.save(model_m.state_dict(), os.path.join(output_dir, 'model_m.pth'))
torch.save(model_t.state_dict(), os.path.join(output_dir, 'model_t.pth'))
np.savez(os.path.join(output_dir, 'pdf.npz'), kappa_R=kappa_R, beta_M=beta_M, gamma_M=gamma_M, kappa_M=kappa_M)
