import numpy as np


def all_close_to_one_of(array, values, rtol=1e-5, atol=1e-8):
    """
    Check if all elements of an array are close to one of the values.

    Args:
        array (np.ndarray): the constrained arrays.
        values (np.ndarray): the allowed values.
        rtol (float, optional): the relative tolerance. Defaults to 1e-5.
        atol (float, optional): the absolute tolerance. Defaults to 1e-8.

    Returns:
        (bool): the check result.
    """
    return np.all(np.any(np.isclose(array[..., np.newaxis], values, rtol=rtol, atol=atol), axis=-1))

def obs_block(mxf, Vxf, myb, Vyb, C):
    """
    Gaussian message passing through a (scalar) observation block.

    Args:
        mxf (np.ndarray): the forward input mean vector with shape (K).
        Vxf (np.ndarray): the forward input covariance matrix with shape (K, K).
        myb (np.ndarray): the backward observation mean with shape (1).
        Vyb (np.ndarray): the backward observation variance with shape (1).
        C (np.ndarray): the observation matrix with shape (K).

    Returns:
        (np.ndarray): the forward output mean vector with shape (K).
        (np.ndarray): the forward output covariance matrix with shape (K, K).
    """
    CVxf = C @ Vxf
    g_inv = Vyb + np.inner(CVxf,C)
    return mxf + (myb - C @ mxf) / g_inv * CVxf, Vxf - np.outer(CVxf, CVxf / g_inv)
