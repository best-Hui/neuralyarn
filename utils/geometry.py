from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)

import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import bsdf.khungurn as khungurn

def streamcurves(filepath, curves, radius, verbose=True):
    if verbose:
        print(f'Writing to {filepath}')

    with open(filepath, 'w') as file:
        for curve in curves:
            for p in curve:
                if np.any(np.isnan(p)):
                    print('WARN streamcurves - Invalid point', p)
                    continue
                
                file.write(f'{p[0]} {p[1]} {p[2]} {radius}\n')
            file.write('\n')

def cross_section_circle(macro_radius, count, density):
    micro_radius = np.sqrt((density * macro_radius**2) / count)
    
    # Generate random 2d samples between 0 & 1
    # Try N * 10 times before giving up
    sample_x = np.random.rand(count * 10)
    sample_y = np.random.rand(count * 10)

    # Find radius of allowed points
    radius = macro_radius - micro_radius

    # Map 2d random samples to interval
    sample_x = sample_x * (radius * 2) - radius
    sample_y = sample_y * (radius * 2) - radius

    # Initialise points
    # Inf is used so that calculating distances would be easier
    x_points = np.full(count, np.inf)
    y_points = np.full(count, np.inf)
    idx = 0

    for x, y in zip(sample_x, sample_y):
        micro_distance = np.min(np.sqrt((x_points - x) ** 2 + (y_points - y) ** 2))    # Distances to each fiber center
        macro_distance = np.sqrt(x ** 2 + y ** 2)                                      # Distance to the yarn center

        is_self_intersect = micro_distance < 2 * micro_radius                          # Is intersecting with other fibers
        is_in_circle = macro_distance < radius                                         # Is in the yarn circle

        #  Add point if it is valid
        if is_in_circle and not is_self_intersect:
            x_points[idx] = x
            y_points[idx] = y
            idx += 1

        # Complete
        if idx == count:
            break

    # Remove all inf values
    x_points = x_points[np.invert(np.isinf(x_points))]
    y_points = y_points[np.invert(np.isinf(y_points))]

    offsets = np.vstack([x_points, y_points]).T

    if offsets.shape[0] != count:
        raise RuntimeError(f"Error, unable to generate all curves ({offsets.shape[0]}/{count}).")

    return np.vstack([x_points, y_points]), micro_radius

def helix(points, radius, offset, twist_factor, up_vector=np.array([0, 0, 1])):

    # To store the coordinates of the cross section
    cross_section_xs = []
    cross_section_ys = []
    cross_section_zs = []

    twist_angle = 0

    # For each point in the curve
    for i, (p1, p2) in enumerate(zip(points[:-1], points[1:])):
        cross_section = np.empty((3, offset.shape[0])) 

        # The displacement vector is p1 -> p2
        displacement = p2 - p1
        distance = np.linalg.norm(displacement)

        if np.isclose(distance, 0):
            cross_section_xs.append(cross_section_xs[-1])
            cross_section_ys.append(cross_section_ys[-1])
            cross_section_zs.append(cross_section_zs[-1])
            continue

        # Get tangent, binormal, and normal
        tangent = displacement
        tangent /= np.linalg.norm(tangent)

        binormal = np.cross(up_vector, tangent)
        binormal /= np.linalg.norm(binormal)

        normal = np.cross(tangent, binormal)
        normal /= np.linalg.norm(normal)

        # Apply rotations, t = n / (x * d)
        twist_angle += np.pi * twist_factor * distance / radius
        rotation = Rot.from_rotvec(tangent * twist_angle)
        binormal = rotation.apply(binormal)
        normal = rotation.apply(normal)

        # Add the offsets to p1 to create the crossection
        cross_section = p1 + offset[0][np.newaxis].T * binormal + offset[1][np.newaxis].T * normal
        
        # Store it
        cross_section = cross_section.T
        cross_section_xs.append(cross_section[0])
        cross_section_ys.append(cross_section[1])
        cross_section_zs.append(cross_section[2])

    # Combine x, y, z
    cross_section_xs = np.asarray(cross_section_xs).T
    cross_section_ys = np.asarray(cross_section_ys).T
    cross_section_zs = np.asarray(cross_section_zs).T
    curves = np.dstack([
        cross_section_xs,
        cross_section_ys,
        cross_section_zs
    ])

    return curves

def create_single_yarn(parameters, resolution=1001, ply_radius=0.5, start=-5.0, end=5.0, verbose=True):
    ply_curves = np.stack([
        np.linspace(start, end, resolution),
        np.zeros(resolution),
        np.zeros(resolution)
    ], axis=1)[np.newaxis]

    # Generate fiber curves
    fiber_offset, fiber_radius = cross_section_circle(ply_radius, int(parameters['N']), parameters['Rho'])
    fiber_curves = [helix(ply, ply_radius, fiber_offset, parameters['Alpha']) for ply in ply_curves]
    fiber_curves = np.asarray(fiber_curves)
    a, b, c, d = fiber_curves.shape
    fiber_curves = fiber_curves.reshape([a * b, c, d])

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

    params = mi.traverse(curves)

    segment_indices = torch.tensor(np.arange(0, parameters['N'] * (resolution - 1))).to(device)
    ends = ((segment_indices + 1) % (resolution - 1)) == 0
    segment_indices = segment_indices[~ends]

    params['control_point_count'] = np.prod(fiber_curves.shape[:2])
    params['control_points'] = np.concatenate([fiber_curves, np.full(fiber_curves.shape[:2], fiber_radius)[:, :, None]], axis=-1).ravel()
    params['segment_indices'] = mi.UInt32(segment_indices.to(torch.uint32))
    params.update()

    return curves





