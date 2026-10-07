# contains all gaussian math, rendering, etc.

import torch
import math
import numpy as np
from PIL import Image

from densification import density_control

def covariance_2d(scale, theta):
    '''
    scale: (N, 2) positive,  theta: (N,) radians
    covariance (Sigma) = R S S^T R^T  -> (N, 2, 2)
    '''
    R = rotation_2d(theta) # (N, 2, 2)
    D = torch.diag_embed(scale ** 2) # (N, 2, 2)
    return R @ D @ R.transpose(-1, -2)


def gaussian_weight(xy, mu, Sigma):
    '''
    xy: (P, 2) pixel coords,  mu: center (N, 2),  Sigma: covariance (N, 2, 2)
    w[p, n] = exp(-0.5 (xy_p - mu_n)^T Sigma_n^-1 (xy_p - mu_n))
    '''
    delta = xy[None, :, :] - mu[:, None, :] 

    eye = torch.eye(2, device=Sigma.device, dtype=Sigma.dtype)
    inv_Sigma = torch.linalg.inv(Sigma + 1e-6 * eye)

    transformed = delta @ inv_Sigma
    distance_squared = (transformed * delta).sum(dim=-1)

    return torch.exp(-0.5 * distance_squared).transpose(0, 1)


def rotation_2d(theta):
    '''
    create rotation matrix R to determine direction of dims 
    '''
    c = torch.cos(theta)
    s = torch.sin(theta)

    return torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s,  c], dim=-1),
    ], dim=-2)

def pixel_grid(H, W, device, dtype):
    '''
    make grid of (x, y) coords for pixels 
    '''
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([x, y], dim=-1).reshape(-1, 2)


# def render(mu, Sigma, color, opacity, order, H, W):
#     '''
#     P2: alpha compositing: determine what color a pixel at (x, y) is
#     '''
#     xy = pixel_grid(H, W, mu.device, mu.dtype) # (H*W, 2)
#     w = gaussian_weight(xy, mu, Sigma)  # (P, N)  from P1

#     # alpha of pixel for each gaussian 
#     alpha = opacity[None, :] * w # (P, N)

#     C = torch.zeros(H * W, 3, device=mu.device, dtype=mu.dtype) # color
#     T = torch.ones(H * W, device=mu.device, dtype=mu.dtype) # visibility left

#     # go thru gaussians from front to back
#     for i in order:
#         a = alpha[:, i]
#         C = C + (T * a)[:, None] * color[i]
#         T = T * (1 - a)

#     return C.reshape(H, W, 3) # final rgb color

def render(mu, Sigma, color, opacity, order, H, W):
    xy = pixel_grid(H, W, mu.device, mu.dtype)
    w = gaussian_weight(xy, mu, Sigma)  # (P, N)

    # Arrange columns from nearest to farthest.
    alpha = w[:, order] * opacity[order][None, :]

    # For Gaussian i, visibility is the product of (1 - alpha)
    # for all Gaussians BEFORE i.
    prefix = torch.cat([
        torch.ones(H * W, 1, device=mu.device, dtype=mu.dtype),
        1 - alpha[:, :-1],
    ], dim=1)

    T = torch.cumprod(prefix, dim=1)
    # Sum each Gaussian's RGB contribution at every pixel.
    C = (T * alpha) @ color[order]
    return C.reshape(H, W, 3)

def quaternion_to_rotation(q):
    '''
    q: (N, 4) as (w, x, y, z)
    normalize q, then build R(q) above -> (N, 3, 3)
    '''
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    w, x, y, z = q.unbind(dim=-1)

    R = torch.stack([
        torch.stack([
            1 - 2 * (y*y + z*z),
            2 * (x*y - w*z),
            2 * (x*z + w*y),
        ], dim=-1),

        torch.stack([
            2 * (x*y + w*z),
            1 - 2 * (x*x + z*z),
            2 * (y*z - w*x),
        ], dim=-1),

        torch.stack([
            2 * (x*z - w*y),
            2 * (y*z + w*x),
            1 - 2 * (x*x + y*y),
        ], dim=-1),
    ], dim=-2)
    return R


def covariance_3d(scale, quat):
    '''
    scale: (N, 3) positive,  quat: (N, 4)
    R = quaternion_to_rotation(quat); return R S S^T R^T  -> (N, 3, 3)
    '''
    R = quaternion_to_rotation(quat)
    D = torch.diag_embed(scale ** 2)
    return R @ D @ R.transpose(-1, -2)


def project_gaussian(mu3, Sigma3, R_wc, t, K):
    '''
    project 3D mean and covariance into camera --> get 2D mean/covar to pass into 2d rasterizer (render()) 
    '''
    # mu3: (N, 3) world means,  Sigma3: (N, 3, 3) world covariances
    mu_cam = mu3 @ R_wc.T + t
    x, y, depth = mu_cam.unbind(dim=-1)
    z = depth.clamp_min(1e-3) # no 0 division

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    mu2 = torch.stack([ # perspective-project mu_cam with K (cam's intrinsic matrix)
        fx * x / z + cx,
        fy * y / z + cy,
    ], dim=-1)

    zero = torch.zeros_like(z)

    # jacobian of mu_cam
    J = torch.stack([
        torch.stack([fx / z, zero, -fx * x / z**2], dim=-1),
        torch.stack([zero, fy / z, -fy * y / z**2], dim=-1),
    ], dim=-2)

    Sigma_cam = R_wc @ Sigma3 @ R_wc.T # (N, 3, 3)
    Sigma2 = J @ Sigma_cam @ J.transpose(-1, -2)

    return mu2, Sigma2, depth