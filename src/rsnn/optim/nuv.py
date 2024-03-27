import numpy as np


def box_nuv(mx, xb, gamma=1.0):
    """
    Update the NUV means and variances for a box constraints of the form |x| <= xb.

    Args:
        mx (np.ndarray): the posterior mean.
        xb (np.ndarray): the box constraints.
        gamma (float): the NUV slope parameter.

    Returns:
        (np.ndarray): the NUV means.
        (np.ndarray): the NUV variances.
    """
    Vxlf = np.abs(mx + xb)
    Vxrl = np.abs(mx - xb)
    return xb * (Vxlf - Vxrl) / (Vxlf + Vxrl), Vxlf * Vxrl / (Vxlf + Vxrl) / gamma


def right_half_space_nuv(mx, xb, gamma=1.0):
    """
    Update the NUV means and variances for a right half space constraints of the form x >= xb.

    Args:
        mx (np.ndarray): the posterior mean.
        xb (np.ndarray): the half-space constraints.
        gamma (float): the NUV slope parameter.

    Returns:
        (np.ndarray): the NUV means.
        (np.ndarray): the NUV variances.
    """
    return xb + np.abs(mx - xb), np.abs(mx - xb) / gamma


def left_half_space_nuv(mx, xb, gamma=1.0):
    """
    Update the NUV means and variances for a left half space constraints of the form x <= xb.

    Args:
        mx (np.ndarray): the posterior mean.
        xb (np.ndarray): the half-space constraints.
        gamma (float): the NUV slope parameter.

    Returns:
        (np.ndarray): the NUV means.
        (np.ndarray): the NUV variances.
    """
    return xb - np.abs(mx - xb), np.abs(mx - xb) / gamma


def binary_nuv(mx, Vx, xb):
    """
    Update the NUV means and variances for a binary constraints of the form x in {-xb, xb}.

    Args:
        mx (np.ndarray): the posterior mean.
        xb (np.ndarray): the binary constraints.

    Returns:
        (np.ndarray): the NUV means.
        (np.ndarray): the NUV variances.
    """
    Vxlf = Vx + np.square(mx + xb)
    Vxrf = Vx + np.square(mx - xb)

    return xb * (Vxlf - Vxrf) / (Vxlf + Vxrf), Vxlf * Vxrf / (Vxlf + Vxrf)
