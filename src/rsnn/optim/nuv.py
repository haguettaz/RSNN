"""
This module contains functions for computing NUV composite priors to enforce constraints on posterior means.
"""

from typing import Tuple

import numpy as np

from .gmp import equality_block

# def box_prior(mx: np.ndarray, xmin: np.ndarray, xmax: np.ndarray, beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     NUV composite box prior to enforce xmin <= mx <= xmax elementwise.

#     Args:
#         mx (np.ndarray[float]): The posterior means.
#         xmin (np.ndarray[float]): The smallest admissible values.
#         xmax (np.ndarray[float]): The largest admissible values.
#         beta (float, optional): The constraint' hardness parameter, the smaller the softer. Defaults to 1.0.

#     Returns:
#         Tuple[np.ndarray[float], np.ndarray[float]]: The box prior means and variances.
#     """

#     if np.any(xmin > xmax):
#         raise ValueError("xmin cannot be larger than xmax")

#     mask_finite_xmin = np.isfinite(xmin)
#     mask_finite_xmax = np.isfinite(xmax)

#     mfx = np.empty_like(mx)
#     Vfx = np.empty_like(mx)

#     # Without constraints
#     mask = (~mask_finite_xmin) & (~mask_finite_xmax)
#     mfx[mask] = mx[mask]
#     Vfx[mask] = 1e9

#     # Laplace, i.e., x = xmin = xmax
#     mask = mask_finite_xmin & mask_finite_xmax & (xmin == xmax)
#     sigma2 = np.abs(mx[mask] - xmin[mask])
#     mfx[mask] = xmin[mask]
#     Vfx[mask] = (sigma2) / beta

#     # Box, i.e., xmin <= x <= xmax
#     mask = mask_finite_xmin & mask_finite_xmax & (xmin < xmax)
#     sigma2x_min = np.abs(mx[mask] - xmin[mask])
#     sigma2x_max = np.abs(mx[mask] - xmax[mask])
#     mfx[mask] = (xmin[mask] * sigma2x_max + xmax[mask] * sigma2x_min) / (sigma2x_min + sigma2x_max)
#     Vfx[mask] = (sigma2x_min * sigma2x_max) / (sigma2x_min + sigma2x_max) / beta

#     return mfx, Vfx


def half_space_prior(mx, xmin=None, xmax=None, beta=1.0) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        mx (np.ndarray): _description_
        xmin (np.ndarray): _description_
        xmax (np.ndarray): _description_
        beta (float, optional): constraint' hardness parameter. Its magnitude reflects its strength, while its sign its nature, either attractive (positive) or repulsive (negative). Defaults to 1.0.

    Raises:
        ValueError: _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    if xmax is None:  # x >= xmin
        sigma2 = np.abs(mx - xmin)
        return xmin + sigma2, sigma2 / beta

    elif xmin is None:  # x <= xmax
        sigma2 = np.abs(mx - xmax)
        return xmax - sigma2, sigma2 / beta

    raise ValueError("Either xmin or xmax must be specified")


def box_prior(mx: np.ndarray, xmin: np.ndarray, xmax: np.ndarray, beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        mx (np.ndarray): _description_
        xmin (np.ndarray): _description_
        xmax (np.ndarray): _description_
        beta (float, optional): constraint' hardness parameter. Its magnitude reflects its strength, while its sign its nature, either attractive (positive) or repulsive (negative). Defaults to 1.0.

    Raises:
        ValueError: _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    if np.any(xmin >= xmax):
        raise ValueError("xmin must be strictly smaller than xmax")

    sigma2x_min = np.abs(mx - xmin)
    sigma2x_max = np.abs(mx - xmax)

    return (xmin * sigma2x_max + xmax * sigma2x_min) / (sigma2x_min + sigma2x_max), (sigma2x_min * sigma2x_max) / (
        sigma2x_min + sigma2x_max
    ) / beta


def laplace_prior(mx: np.ndarray, x: np.ndarray, beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        mx (np.ndarray): _description_
        x (np.ndarray): _description_
        beta (float, optional): constraint' hardness parameter. Its magnitude reflects its strength, while its sign its nature, either attractive (positive) or repulsive (negative). Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    return x, np.abs(mx - x) / beta


def plain_prior(
    mx: np.ndarray, Vx: np.ndarray, x: np.ndarray, beta: float = 1.0, r2: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        mx (np.ndarray): _description_
        Vx (np.ndarray): _description_
        x (np.ndarray): _description_
        beta (float, optional): constraint' hardness parameter. Its magnitude reflects its strength, while its sign its nature, either attractive (positive) or repulsive (negative). Defaults to 1.0.
        r2 (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    return x, np.maximum((Vx + (mx - x) ** 2), r2) / beta


def pnorm_prior(
    mx: np.ndarray,
    Vx: np.ndarray,
    x: np.ndarray,
    p: float = 2.0,
    beta: float = 1.0,
    r2: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    return x, np.maximum(Vx + np.abs(mx - x) ** (2 - p) / (p), r2) / beta


def binary_prior(
    mx: np.ndarray, Vx: np.ndarray, xmin: np.ndarray, xmax: np.ndarray, beta: float = 1.0, rep: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NUV composite binarizing prior to enforce 2-discrete mx in between xmin and xmax.

    Args:
        mx (np.ndarray[float]): posterior means.
        Vx (np.ndarray[float]): posterior variances.
        xmin (np.ndarray[float]): smallest admissible values.
        xmax (np.ndarray[float]): largest admissible values.
        beta (float, optional): constraint' hardness parameter. Its magnitude reflects its strength, while its sign its nature, either attractive (positive) or repulsive (negative). Defaults to 1.0.
        rep (bool, optional): whether to use the repulsive prior. Defaults to False.

    Returns:
        Tuple[np.ndarray[float], np.ndarray[float]]: The binary prior means and variances.
    """
    if np.any(xmin > xmax):
        raise ValueError("xmin cannot be larger than xmax")

    # box constraint on [xmin, xmax] and one repulsive plain nuv prior in the middle
    if rep:
        mxbox, Vxbox = box_prior(mx, xmin, xmax)
        return equality_block(
            mxbox, Vxbox, np.full_like(mx, (xmin + xmax) / 2), -(Vx + (mx - (xmin + xmax) / 2) ** 2) / beta
        )

    # two (attractive) plain nuv priors
    return equality_block(xmin, (Vx + (mx - xmin) ** 2) / beta, xmax, (Vx + (mx - xmax) ** 2) / beta)
