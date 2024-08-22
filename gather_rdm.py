from config.parameters import get_fiber_parameters
from utils.rdm import compute_rdm

import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description="Gather RDM for a yarn.")

parser.add_argument("--checkpoint_name", help="Checkpoint name to store outputs")
parser.add_argument("--yarn_name", help="Name of the yarn defined in config/parameters.py")
parser.add_argument("--batch_size", type=int, help='Samples per batch for RDM collection', default=1024*8)
parser.add_argument("--num_batches", type=int, help='Number of batches for RDM collection', default=1024)
parser.add_argument("--theta_bins", type=int, help='Theta resolution of the RDM', default=18)
parser.add_argument("--phi_bins", type=int, help='Phi resolution of the RDM', default=36)

args = parser.parse_args()

output_dir = os.path.join('./checkpoints', args.checkpoint_name)

os.makedirs(output_dir, exist_ok=True)

parameters = get_fiber_parameters(args.yarn_name)

rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa \
    = compute_rdm(parameters, theta_bins=args.theta_bins, phi_bins=args.theta_bins, num_batches=args.num_batches, batch_size=args.batch_size)

rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa = [x.numpy() for x in [rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa]]

print('Saving to', output_dir)

np.savez(os.path.join(output_dir, 'rdm.npz'), rdm_t=rdm_t, rdm_r=rdm_r, rdm_m=rdm_m, 
         count_t=count_t, count_r=count_r, count_m=count_m, 
         x=x, sa=sa)

print('Done')

