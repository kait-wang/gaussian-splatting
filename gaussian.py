# contains all gaussian math, rendering, etc.

import torch
import math
from pathlib import Path
import numpy as np
from PIL import Image


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

def render(mu, Sigma, color, opacity, order, H, W):
    '''
    alpha compositing: determine what color a pixel at (x, y) is
    '''
    xy = pixel_grid(H, W, mu.device, mu.dtype)
    w = gaussian_weight(xy, mu, Sigma) 

    # alpha of pixel for each gaussian 
    alpha = opacity[None, :] * w # (P, N)

    C = torch.zeros(H * W, 3, device=mu.device, dtype=mu.dtype) # color
    T = torch.ones(H * W, device=mu.device, dtype=mu.dtype) # visibility left

    # go thru gaussians from front to back
    for i in order:
        a = alpha[:, i]
        C = C + (T * a)[:, None] * color[i]
        T = T * (1 - a)

    return C.reshape(H, W, 3)