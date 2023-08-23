"""
This module contains utility functions for performing various operations on arrays and matrices.
"""

from typing import Tuple

import numpy as np


def half_space_cost(mx, xmin=None, xmax=None, beta=1.0) -> np.ndarray:
    """_summary_

    Args:
        mx (np.ndarray): _description_
        xmin (np.ndarray): _description_
        xmax (np.ndarray): _description_
        beta (float, optional): constraint' hardness parameter. Its magnitude reflects its strength, while its sign its nature, either attractive (positive) or repulsive (negative). Defaults to 1.0. 

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """
    if xmax is None: # x >= xmin
        return beta * (np.abs(mx - xmin) - (mx - xmin))
    
    elif xmin is None: # x <= xmax
        return beta * (np.abs(mx - xmax) - (xmax - mx))
    
    raise ValueError("Either xmin or xmax must be specified")    

def box_cost(mx:np.ndarray, xmin:np.ndarray, xmax:np.ndarray, beta:float=1.0) -> np.ndarray:
    """_summary_

    Args:
        mx (np.ndarray): _description_
        xmin (np.ndarray): _description_
        xmax (np.ndarray): _description_
        beta (float, optional): constraint' hardness parameter. Its magnitude reflects its strength, while its sign its nature, either attractive (positive) or repulsive (negative). Defaults to 1.0. 

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """
    if np.any(xmin >= xmax):
        raise ValueError("xmin must be strictly smaller than xmax")

    return beta * (np.abs(mx - xmin) + np.abs(mx - xmax) - (xmax - xmin))

def laplace_cost(mx:np.ndarray, x:np.ndarray, beta:float=1.0) -> np.ndarray:
    """_summary_

    Args:
        mx (np.ndarray): _description_
        x (np.ndarray): _description_
        beta (float, optional): constraint' hardness parameter. Its magnitude reflects its strength, while its sign its nature, either attractive (positive) or repulsive (negative). Defaults to 1.0. 

    Returns:
        _type_: _description_
    """
    return beta * np.abs(mx - x)

def plain_cost(mx:np.ndarray, x:np.ndarray, beta:float=1.0) -> np.ndarray:
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

    return beta * np.log(np.abs(mx - x))

def pnorm_cost(mx: np.ndarray, x: np.ndarray, p: float = 2.0, beta:float=1.0) -> np.ndarray:
    """_summary_

    Args:
        mx (np.ndarray): _description_
        x (np.ndarray): _description_
        p (float, optional): _description_. Defaults to 2.0.
        beta (float, optional): _description_. Defaults to 1.0.

    Returns:
        np.ndarray: _description_
    """
    return beta * np.abs(mx - x) ** p

def box_error(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    """
    Calculate the box constraint error for a set of posterior means.

    Args:
        x (np.ndarray[float]): The posterior means.
        x_min (np.ndarray[float]): The smallest admissible values.
        x_max (np.ndarray[float]): The largest admissible values.

    Returns:
        np.ndarray[float]: The box constraint error.
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
        x (np.ndarray[float]): The posterior means.
        x_min (np.ndarray[float]): The smallest admissible values.
        x_max (np.ndarray[float]): The largest admissible values.

    Returns:
        np.ndarray[float]: The binarization constraint error.
    """
    return np.max(np.minimum(x - x_min, x_max - x))
