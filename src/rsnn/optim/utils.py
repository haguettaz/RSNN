"""
This module contains utility functions for performing various operations on arrays and matrices.
"""

import numpy as np


def box_error(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    """
    Calculate the box constraint error for a set of posterior means.

    Args:
        x (np.ndarray[float]): the posterior means.
        x_min (np.ndarray[float]): the smallest admissible values.
        x_max (np.ndarray[float]): the largest admissible values.

    Returns:
        np.ndarray[float]: the box constraint error.
    """
    err = np.zeros(1)

    mask_finite_xmin = np.isfinite(x_min)
    mask_finite_xmax = np.isfinite(x_max)

    # Left half-space, i.e., x <= x_max
    mask = (~mask_finite_xmin) & mask_finite_xmax
    if np.any(mask):
        # print("left half-space", np.max(np.abs(x_max[mask] - x[mask]) - (x_max[mask] - x[mask])))
        err = np.maximum(err, np.max(np.abs(x_max[mask] - x[mask]) - (x_max[mask] - x[mask])))

    # Right half-space, i.e., x >= x_min
    mask = (~mask_finite_xmax) & mask_finite_xmin
    if np.any(mask):
        # print("right half-space", np.max(np.abs(x[mask] - x_min[mask]) - (x[mask] - x_min[mask])))
        err = np.maximum(err, np.max(np.abs(x[mask] - x_min[mask]) - (x[mask] - x_min[mask])))

    # Box, i.e., x_min <= x <= x_max
    mask = mask_finite_xmin & mask_finite_xmax
    if np.any(mask):
        x_, x_min_, x_max_ = x[mask], x_min[mask], x_max[mask]
        # print("box", np.max(np.abs(x[mask] - x_min[mask]) + np.abs(x[mask] - x_max[mask]) - (x_max[mask] - x_min[mask])))
        err = np.maximum(err, np.max(np.abs(x_ - x_min_) + np.abs(x_ - x_max_) - (x_max_ - x_min_)))  # type: ignore

    return err


def bin_error(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> float:
    """
    Calculate the binarization constraint error for a set of posterior means.

    Args:
        x (np.ndarray[float]): the posterior means.
        x_min (np.ndarray[float]): the smallest admissible values.
        x_max (np.ndarray[float]): the largest admissible values.

    Returns:
        np.ndarray[float]: the binarization constraint error.
    """
    return np.max(np.minimum(x - x_min, x_max - x))
