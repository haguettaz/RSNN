"""
This module contains functions for computing NUV composite priors to enforce constraints on posterior means.
"""

from typing import Tuple

import numpy as np


def box_prior(
    mx: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, gamma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NUV composite box prior to enforce x_min <= mx <= x_max elementwise.

    Args:
        mx (np.ndarray[float]): The posterior means.
        x_min (np.ndarray[float]): The smallest admissible values.
        x_max (np.ndarray[float]): The largest admissible values.
        gamma (float, optional): The constraint' hardness parameter, the smaller the softer. Defaults to 1.0.

    Returns:
        Tuple[np.ndarray[float], np.ndarray[float]]: The box prior means and variances.
    """

    if np.any(x_min > x_max):
        raise ValueError("x_min cannot be larger than x_max")

    mask_finite_xmin = np.isfinite(x_min)
    mask_finite_xmax = np.isfinite(x_max)

    mfx = np.empty_like(mx)
    Vfx = np.empty_like(mx)

    # Without constraints
    mask = (~mask_finite_xmin) & (~mask_finite_xmax)
    mfx[mask] = mx[mask]
    Vfx[mask] = 1e9

    # Left half-space, i.e., x <= x_max
    mask = (~mask_finite_xmin) & mask_finite_xmax
    sigma2x_max = np.abs(mx[mask] - x_max[mask])
    mfx[mask] = x_max[mask] - sigma2x_max
    Vfx[mask] = sigma2x_max / gamma

    # Right half-space, i.e., x >= x_min
    mask = mask_finite_xmin & (~mask_finite_xmax)
    sigma2x_min = np.abs(mx[mask] - x_min[mask])
    mfx[mask] = x_min[mask] + sigma2x_min
    Vfx[mask] = sigma2x_min / gamma

    # Laplace, i.e., x = x_min = x_max
    mask = mask_finite_xmin & mask_finite_xmax & (x_min == x_max)
    sigma2 = np.abs(mx[mask] - x_min[mask])
    mfx[mask] = x_min[mask]
    Vfx[mask] = (sigma2) / gamma

    # Box, i.e., x_min <= x <= x_max
    mask = mask_finite_xmin & mask_finite_xmax & (x_min < x_max)
    sigma2x_min = np.abs(mx[mask] - x_min[mask])
    sigma2x_max = np.abs(mx[mask] - x_max[mask])
    mfx[mask] = (x_min[mask] * sigma2x_max + x_max[mask] * sigma2x_min) / (sigma2x_min + sigma2x_max)
    Vfx[mask] = (sigma2x_min * sigma2x_max) / (sigma2x_min + sigma2x_max) / gamma

    return mfx, Vfx


def binary_prior(mx:np.ndarray, Vx:np.ndarray, x_min:np.ndarray, x_max:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    NUV composite binarizing prior to enforce 2-discrete mx in between x_min and x_max.

    Args:
        mx (np.ndarray[float]): posterior means.
        Vx (np.ndarray[float]): posterior variances.
        x_min (np.ndarray[float]): smallest admissible values.
        x_max (np.ndarray[float]): largest admissible values.

    Returns:
        Tuple[np.ndarray[float], np.ndarray[float]]: The binary prior means and variances.
    """
    if np.any(x_min > x_max):
        raise ValueError("x_min cannot be larger than x_max")

    Vx_min = Vx + (mx - x_min) ** 2
    Vx_max = Vx + (mx - x_max) ** 2

    mfx = (x_min * Vx_max + x_max * Vx_min) / (Vx_min + Vx_max)
    Vfx = (Vx_min * Vx_max) / (Vx_min + Vx_max)

    return mfx, Vfx
