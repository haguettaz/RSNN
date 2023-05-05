import numpy as np


def box_error(x: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> float:
    err = np.zeros(1)

    mask_finite_xmin = np.isfinite(xmin)
    mask_finite_xmax = np.isfinite(xmax)

    # left half-space
    mask = (~mask_finite_xmin) & mask_finite_xmax
    if np.any(mask):
        # print("left half-space", np.max(np.abs(xmax[mask] - x[mask]) - (xmax[mask] - x[mask])))
        err = np.maximum(err, np.max(np.abs(xmax[mask] - x[mask]) - (xmax[mask] - x[mask])))

    # right half-space
    mask = (~mask_finite_xmax) & mask_finite_xmin
    if np.any(mask):
        # print("right half-space", np.max(np.abs(x[mask] - xmin[mask]) - (x[mask] - xmin[mask])))
        err = np.maximum(err, np.max(np.abs(x[mask] - xmin[mask]) - (x[mask] - xmin[mask])))

    # box
    mask = mask_finite_xmin & mask_finite_xmax
    if np.any(mask):
        # print("box", np.max(np.abs(x[mask] - xmin[mask]) + np.abs(x[mask] - xmax[mask]) - (xmax[mask] - xmin[mask])))
        err = np.maximum(err, np.max(np.abs(x[mask] - xmin[mask]) + np.abs(x[mask] - xmax[mask]) - (xmax[mask] - xmin[mask])))

    return err

def bin_error(x: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> float:
    return np.max(np.minimum(x-xmin, xmax-x))