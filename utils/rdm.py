from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)

import torch

import numpy as np
from mitsuba import ScalarTransform4f as sT

from utils.geometry import create_single_yarn

def calculate_solid_angle_grid(n, m):
    # Define the ranges for theta (polar) and phi (azimuthal) angles
    theta_edges = np.linspace(0, np.pi, n + 1)
    phi_edges = np.linspace(0, 2 * np.pi, m + 1)
    
    # Calculate bin centers
    theta_bins = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_bins = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    
    # Calculate the solid angle for each bin
    solid_angle_grid = np.zeros((n, m))
    
    for i in range(n):
        delta_cos_theta = np.cos(theta_edges[i]) - np.cos(theta_edges[i+1])
        for j in range(m):
            delta_phi = phi_edges[j+1] - phi_edges[j]
            solid_angle_grid[i, j] = delta_cos_theta * delta_phi
    
    return solid_angle_grid, theta_bins, phi_bins

def trace_paths(scene, rays, max_depth=100):
    n_samples = dr.shape(rays.o)[1]
    sampler = mi.load_dict({'type' : 'independent'})
    sampler.seed(np.random.randint(np.iinfo(np.int32).max), wavefront_size=n_samples)

    si = scene.ray_intersect(rays)
    valid = si.is_valid()

    # Randomly sample initial wi
    wi = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())

    frame_i = mi.Frame3f(si.sh_frame)
    si.wi = wi

    throughput = mi.Spectrum(1.0)
    active = mi.Bool(valid)
    depth = mi.UInt32(0)
    i = mi.UInt32(0)

    loop = mi.Loop(name="", state=lambda:(i, sampler, rays, si, throughput, active, depth))

    while loop(active & (i < max_depth)):
        depth[active] += mi.UInt32(1)

        # Sample new direction
        ctx = mi.BSDFContext()
        bs, bsdf_val = si.bsdf().sample(ctx, si, sampler.next_1d(), sampler.next_2d(), True)

        # Update throughput and rays for next bounce
        throughput[active] *= bsdf_val
        rays[active] = si.spawn_ray(si.to_world(bs.wo))

        # Find next intersection
        si = scene.ray_intersect(rays, active)
        active &= si.is_valid()

        # Increase loop iteration counter
        i += 1

    wo = frame_i.to_local(rays.d)
    
    return wi, wo, throughput, depth, frame_i, si

# Flatten N-dimensional indices to a flat array
def ravel_index(index, shape):
    ravel_idx = dr.zeros(mi.UInt32, dr.shape(index[0])[0])
    step = 1
    for i in range(len(shape) - 1, 0 - 1, -1):
        ravel_idx += index[i] * step
        step *= shape[i]

    return ravel_idx

def compute_histogram_4d(omega_i, omega_o, outputs, theta_bins=180, phi_bins=360):

    theta_i = dr.acos(omega_i.z)
    phi_i = dr.atan2(omega_i.y, omega_i.x)
    theta_o = dr.acos(omega_o.z)
    phi_o = dr.atan2(omega_o.y, omega_o.x)

    # Digitize the thetas and phis to find the bin indices
    theta_i_idx = mi.UInt32(theta_i / (dr.pi / 2) * (theta_bins//2))
    phi_i_idx = mi.UInt32((phi_i + dr.pi) / dr.two_pi * phi_bins)
    theta_o_idx = mi.UInt32(theta_o / dr.pi * theta_bins)
    phi_o_idx = mi.UInt32((phi_o + dr.pi) / dr.two_pi * phi_bins)
    
    # Filter out out-of-bound indices (i.e., -1 or indices equal to the number of bins)
    valid_mask = mi.Bool(
        (theta_i_idx >= 0) & (theta_i_idx < theta_bins//2) & \
        (phi_i_idx >= 0) & (phi_i_idx < phi_bins) & \
        (theta_o_idx >= 0) & (theta_o_idx < theta_bins) & \
        (phi_o_idx >= 0) & (phi_o_idx < phi_bins))

    # Apply the mask to indices and outputs
    theta_i_idx = theta_i_idx[valid_mask]
    phi_i_idx = phi_i_idx[valid_mask]
    theta_o_idx = theta_o_idx[valid_mask]
    phi_o_idx = phi_o_idx[valid_mask]  
    outputs = outputs[valid_mask]

    # Create a 4D array to accumulate the sum of outputs for each bin
    shape = (theta_bins//2, phi_bins, theta_bins, phi_bins, 3)
    s = dr.zeros(mi.Float, dr.prod(shape))

    # Accumulate outputs into the bins
    dr.scatter_reduce(dr.ReduceOp.Add, s, outputs.x, ravel_index([theta_i_idx, phi_i_idx, theta_o_idx, phi_o_idx, mi.UInt32(0)], shape))
    dr.scatter_reduce(dr.ReduceOp.Add, s, outputs.y, ravel_index([theta_i_idx, phi_i_idx, theta_o_idx, phi_o_idx, mi.UInt32(1)], shape))
    dr.scatter_reduce(dr.ReduceOp.Add, s, outputs.z, ravel_index([theta_i_idx, phi_i_idx, theta_o_idx, phi_o_idx, mi.UInt32(2)], shape))

    # Create a 2D array to accumulate the count of omega_i for each bin
    count_i_shape = (theta_bins//2, phi_bins)
    count_i = dr.zeros(mi.Float, dr.prod(count_i_shape))

    dr.scatter_reduce(dr.ReduceOp.Add, count_i, mi.Float(1), ravel_index([theta_i_idx, phi_i_idx], count_i_shape))

    solid_angles, _, _ = calculate_solid_angle_grid(theta_bins, phi_bins)

    histogram = mi.TensorXf(s, shape)
    histogram_count = mi.TensorXf(count_i, count_i_shape)
    histogram /= histogram_count[:, :, None, None, None]
    histogram /= solid_angles[None, None, :, :, None]
    histogram = dr.select(dr.isnan(histogram), 0.0, histogram)

    return histogram, histogram_count

def cart2sph(v):
    v = dr.norm(v)
    theta = dr.acos(v.z)
    phi = dr.atan2(v.y, v.x)
    return theta, phi

def collect_rdm(scene, num_samples=1024*1024*16, theta_bins=180//4, phi_bins=360//4, max_depth=4096):
    o = torch.randn(num_samples, 2, device=device)
    o /= torch.norm(o, dim=-1, keepdim=True)

    o = mi.Point3f(0.0, o[:, 0], o[:, 1])
    d = mi.Vector3f(-o)

    ray = mi.Ray3f(o, d)

    wi, wo, throughput, depth, frame, si = trace_paths(scene, ray, max_depth=max_depth)

    throughput = dr.select(depth < 2, 0.0, throughput)

    assert dr.count(dr.eq(si.shape, None)) == num_samples, f"Not all rays escaped: {dr.count(dr.eq(si.shape, None))}/{num_samples}"

    depth -= 2
    
    select_t = dr.eq(depth, 0)
    select_r = dr.eq(depth, 1) & (dr.dot(wo, frame.n) > 0.0)
    select_m = ~select_t & ~select_r

    throughput_t = dr.select(select_t, throughput, 0.0)
    throughput_r = dr.select(select_r, throughput, 0.0)
    throughput_m = dr.select(select_m, throughput, 0.0)

    rdm_t, count_t = compute_histogram_4d(wi, wo, throughput_t, theta_bins=theta_bins, phi_bins=phi_bins)
    rdm_r, count_r = compute_histogram_4d(wi, wo, throughput_r, theta_bins=theta_bins, phi_bins=phi_bins)
    rdm_m, count_m = compute_histogram_4d(wi, wo, throughput_m, theta_bins=theta_bins, phi_bins=phi_bins)

    return rdm_t, rdm_r, rdm_m, count_t, count_r, count_m

def compute_rdm(parameters, theta_bins=180//4, phi_bins=360//4, max_depth=4096, num_batches=1024, batch_size=1024*8):

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

    for i in range(num_batches):
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

    rdm_t /= num_batches
    rdm_r /= num_batches
    rdm_m /= num_batches
    
    # Calculate the centers of the theta and phi bins
    theta_i_centers = np.linspace(0, np.pi / 2, theta_bins//2, endpoint=False) + (np.pi / 2 / (theta_bins//2)) / 2
    phi_i_centers = np.linspace(-np.pi, np.pi, phi_bins, endpoint=False) + (2 * np.pi / phi_bins) / 2
    theta_o_centers = np.linspace(0, np.pi, theta_bins, endpoint=False) + (np.pi / theta_bins) / 2
    phi_o_centers = np.linspace(-np.pi, np.pi, phi_bins, endpoint=False) + (2 * np.pi / phi_bins) / 2

    theta_i, phi_i, theta_o, phi_o = np.meshgrid(theta_i_centers, phi_i_centers, theta_o_centers, phi_o_centers, indexing='ij')
    x = np.stack([theta_i, phi_i, theta_o, phi_o], axis=-1)
    x = mi.TensorXf(x)

    solid_angles = mi.TensorXf(calculate_solid_angle_grid(theta_bins, phi_bins)[0])

    return rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, solid_angles