from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


def pose_sinusoidal_encoding(
    x: torch.Tensor,
    num_freqs: int = 10,
    scale: float = 1e-4
) -> torch.Tensor:
    """
    SODA-style sinusoidal encoding for rays (or any coordinates).

    Args:
      x:          (..., D) tensor of coordinates; D can be 2, 3, 6, etc.
      num_freqs:  number of frequency bands (L) for the encoding
      scale:      's' hyperparameter, e.g. 0.0001 as in Table 12 of the paper

    Returns:
      A tensor of shape (..., 2 * D * num_freqs), containing:
        [
         sin(1 * 2π * s * x), cos(1 * 2π * s * x),
         sin(2 * 2π * s * x), cos(2 * 2π * s * x),
         ...
         up to 2^(num_freqs - 1)
        ]
    """
    # 1) Remember the original shape so we can restore it later
    orig_shape = x.shape
    D = orig_shape[-1]  # last dimension is the coordinate dimension

    # 2) Flatten everything except the last dimension
    x_flat = x.view(-1, D)  # shape (N, D)
    n = x_flat.shape[0]     # number of flattened elements

    # 3) Create frequency multipliers [1, 2, 4, ..., 2^(num_freqs-1)]
    freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32, device=x.device)

    # 4) Scale the input by (2π * s) and then multiply by each frequency
    scaled_x = (2.0 * torch.pi * scale) * x_flat.unsqueeze(-1)  # (N, D, 1)
    scaled_x = scaled_x * freqs.view(1, 1, num_freqs)          # (N, D, num_freqs)

    # 5) Apply sin and cos
    sin_x = torch.sin(scaled_x)  # (N, D, num_freqs)
    cos_x = torch.cos(scaled_x)  # (N, D, num_freqs)

    # 6) Concatenate sin and cos => shape (N, D, 2*num_freqs)
    enc = torch.cat([sin_x, cos_x], dim=-1)

    # 7) Flatten out the D dimension => shape (N, 2*D*num_freqs)
    enc = enc.view(n, -1)

    # 8) Reshape back to (..., 2*D*num_freqs)
    out_shape = list(orig_shape[:-1]) + [2 * D * num_freqs]
    enc = enc.view(*out_shape)

    return enc