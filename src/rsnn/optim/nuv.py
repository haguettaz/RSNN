import numpy as np


def box_prior(mx, xmin=None, xmax=None, gamma=1):
    """
    Returns prior messages corresponding to box constraints between xmin and xmax (inclusive).

    Args:
        mx (np.FloatTensor): posterior means.
        xmin (float, optional): smallest admissible value. Defaults to None.
        xmax (float, optional): largest admissible value. Defaults to None.
        gamma (float, optional): constraint' hardness parameter, the smaller the softer. Defaults to 1.

    Returns:
        mfx (np.FloatTensor): prior means.
        Vfx (np.FloatTensor): prior variances.
    """

    if np.any(xmin > xmax):
        raise ValueError(f"xmin cannot be larger than xmax")
    
    mask_finite_xmin = np.isfinite(xmin)
    mask_finite_xmax = np.isfinite(xmax)

    mfx = np.empty_like(mx)
    Vfx = np.empty_like(mx)

    # no constraints
    mask = (~mask_finite_xmin) & (~mask_finite_xmax)
    mfx[mask] = mx[mask]
    Vfx[mask] = 1e9

    # left half-space
    mask = (~mask_finite_xmin) & mask_finite_xmax
    sigma2x_max = np.abs(mx[mask] - xmax[mask])
    mfx[mask] = xmax[mask] - sigma2x_max
    Vfx[mask] = sigma2x_max / gamma

    # right half-space
    mask = mask_finite_xmin & (~mask_finite_xmax)
    sigma2x_min = np.abs(mx[mask] - xmin[mask])
    mfx[mask] = xmin[mask] + sigma2x_min
    Vfx[mask] = sigma2x_min / gamma

    # laplace
    mask = mask_finite_xmin & mask_finite_xmax & (xmin == xmax)
    sigma2 = np.abs(mx[mask] - xmin[mask])
    mfx[mask] = xmin[mask]
    Vfx[mask] = (sigma2) / gamma

    # box
    mask = mask_finite_xmin & mask_finite_xmax & (xmin < xmax)
    sigma2x_min = np.abs(mx[mask] - xmin[mask])
    sigma2x_max = np.abs(mx[mask] - xmax[mask])
    mfx[mask] = (xmin[mask] * sigma2x_max + xmax[mask] * sigma2x_min) / (sigma2x_min + sigma2x_max)
    Vfx[mask] = (sigma2x_min * sigma2x_max) / (sigma2x_min + sigma2x_max) / gamma

    return mfx, Vfx


def binary_prior(mx, Vx, xmin, xmax):
    """
    Returns prior messages corresponding to binary constraints between wmin and wmax.

    Args:
        mx (np.FloatTensor): posterior means with size (..., C).
        Vx (np.FloatTensor): posterior variances with size (..., C). Defaults to None.
        xmin (float): smallest admissible value. 
        xmax (float): largest admissible value. 

    Returns:
        mfx (np.FloatTensor): prior means with shape (..., C).
        Vfx (np.FloatTensor): prior variances with shape (..., C).
    """
    if np.any(xmin > xmax):
        raise ValueError(f"xmin cannot be larger than xmax")

    Vx_min = Vx + (mx - xmin)**2
    Vx_max = Vx + (mx - xmax)**2

    mfx = (xmin * Vx_max + xmax * Vx_min) / (Vx_min + Vx_max)
    Vfx = (Vx_min * Vx_max) / (Vx_min + Vx_max)

    return mfx, Vfx


def m_ary_prior(mxm, Vxm, xmin, xmax, xlvl):
    """
    Returns prior messages corresponding to M-level constraints between xmin and xmax (inclusive).

    Args:
        mx (np.FloatTensor): posterior means with size (..., M-1, L).
        Vx (np.FloatTensor): posterior variances with size (..., M-1, L).
        xmin (float): smallest admissible value.
        xmax (float): largest admissible value.
        M (int): number of levels.

    Returns:
        mfx (np.FloatTensor): prior means with shape (..., L).
        Vfx (np.FloatTensor): prior variances with shape (..., L).
    """
    if np.any(xmin > xmax):
        raise ValueError(f"xmin cannot be larger than xmax")

    if xlvl - 1 != mxm.shape[-1] != Vxm.shape[-1]:
        raise ValueError(f"xlvl, mxm, and Vxm must match")

    mfxm, Vfxm = binary_prior(mxm, Vxm, xmin/(xlvl-1), xmax/(xlvl-1))

    return mfxm, Vfxm
