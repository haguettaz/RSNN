import numpy as np


def box_error(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> float:
    """
    Box constraint error.

    Args:
        x (np.ndarray): the posterior means.
        x_min (np.ndarray): the smallest admissible values.
        x_max (np.ndarray): the largest admissible values.

    Returns:
        (float): the error.
    """
    err = np.zeros(1)

    mask_finite_xmin = np.isfinite(x_min)
    mask_finite_xmax = np.isfinite(x_max)

    # left half-space
    mask = (~mask_finite_xmin) & mask_finite_xmax
    if np.any(mask):
        # print("left half-space", np.max(np.abs(x_max[mask] - x[mask]) - (x_max[mask] - x[mask])))
        err = np.maximum(err, np.max(np.abs(x_max[mask] - x[mask]) - (x_max[mask] - x[mask])))

    # right half-space
    mask = (~mask_finite_xmax) & mask_finite_xmin
    if np.any(mask):
        # print("right half-space", np.max(np.abs(x[mask] - x_min[mask]) - (x[mask] - x_min[mask])))
        err = np.maximum(err, np.max(np.abs(x[mask] - x_min[mask]) - (x[mask] - x_min[mask])))

    # box
    mask = mask_finite_xmin & mask_finite_xmax
    if np.any(mask):
        # print("box", np.max(np.abs(x[mask] - x_min[mask]) + np.abs(x[mask] - x_max[mask]) - (x_max[mask] - x_min[mask])))
        err = np.maximum(err, np.max(np.abs(x[mask] - x_min[mask]) + np.abs(x[mask] - x_max[mask]) - (x_max[mask] - x_min[mask])))

    return err

def bin_error(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> float:
    """
    Binarization constraint error.

    Args:
        x (np.ndarray): the posteriors means.
        x_min (np.ndarray): the smallest admissible values.
        x_max (np.ndarray): the largest admissible values.

    Returns:
        (float): the error.
    """
    return np.max(np.minimum(x-x_min, x_max-x))