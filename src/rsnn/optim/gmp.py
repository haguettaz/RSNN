from typing import Tuple

import numpy as np


def observation_block_forward(
    A: np.ndarray, mx_forward: np.ndarray, Vx_forward: np.ndarray, my_backward: np.ndarray, Vy_backward: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward Gaussian message passing through an observation block (Table V in Loeliger2016) with one scalar observation.

    Args:
        A (np.ndarray[float]): state to observation matrix.
        mx_forward (np.ndarray[float]): forward state mean vector.
        Vx_forward (np.ndarray[float]): forward state covariance matrix.
        my_backward (np.ndarray[float]): backward observation mean.
        Vy_backward (np.ndarray[float]): backward observation variance.

    Returns:
        Tuple[np.ndarray[float], np.ndarray[float]]: forward state mean vector and convariance matrix following the observation block.
    """
    Vx_forward_At = Vx_forward @ A

    g = 1 / (Vy_backward + np.inner(A, Vx_forward_At))

    mz_forward = mx_forward + g * (my_backward - np.inner(A, mx_forward)) * Vx_forward_At
    Vz_forward = Vx_forward - g * np.outer(Vx_forward_At, Vx_forward_At)

    return mz_forward, Vz_forward


def equality_block(mx: np.ndarray, Vx: np.ndarray, my: np.ndarray, Vy: np.ndarray):
    return (mx * Vy + my * Vx) / (Vx + Vy), (Vx * Vy) / (Vx + Vy)
