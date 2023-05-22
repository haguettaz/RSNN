import numpy as np


def box_prior(mx:np.ndarray, x_min=np.ndarray, x_max=np.ndarray, gamma=1):
    """
    NUV composite box prior to enforce x_min <= mx <= x_max elementwise.

    Args:
        mx (np.ndarray): posterior means.
        x_min (np.ndarray): smallest admissible values.
        x_max (np.ndarray): largest admissible values.
        gamma (float): constraint' hardness parameter, the smaller the softer. Defaults to 1.

    Returns:
        mfx (np.ndarray): box prior means.
        Vfx (np.ndarray): box prior variances.
    """

    if np.any(x_min > x_max):
        raise ValueError(f"x_min cannot be larger than x_max")
    
    mask_finite_xmin = np.isfinite(x_min)
    mask_finite_xmax = np.isfinite(x_max)

    mfx = np.empty_like(mx)
    Vfx = np.empty_like(mx)

    # no constraints
    mask = (~mask_finite_xmin) & (~mask_finite_xmax)
    mfx[mask] = mx[mask]
    Vfx[mask] = 1e9

    # left half-space
    mask = (~mask_finite_xmin) & mask_finite_xmax
    sigma2x_max = np.abs(mx[mask] - x_max[mask])
    mfx[mask] = x_max[mask] - sigma2x_max
    Vfx[mask] = sigma2x_max / gamma

    # right half-space
    mask = mask_finite_xmin & (~mask_finite_xmax)
    sigma2x_min = np.abs(mx[mask] - x_min[mask])
    mfx[mask] = x_min[mask] + sigma2x_min
    Vfx[mask] = sigma2x_min / gamma

    # laplace
    mask = mask_finite_xmin & mask_finite_xmax & (x_min == x_max)
    sigma2 = np.abs(mx[mask] - x_min[mask])
    mfx[mask] = x_min[mask]
    Vfx[mask] = (sigma2) / gamma

    # box
    mask = mask_finite_xmin & mask_finite_xmax & (x_min < x_max)
    sigma2x_min = np.abs(mx[mask] - x_min[mask])
    sigma2x_max = np.abs(mx[mask] - x_max[mask])
    mfx[mask] = (x_min[mask] * sigma2x_max + x_max[mask] * sigma2x_min) / (sigma2x_min + sigma2x_max)
    Vfx[mask] = (sigma2x_min * sigma2x_max) / (sigma2x_min + sigma2x_max) / gamma

    return mfx, Vfx


def binary_prior(mx, Vx, x_min, x_max):
    """
    NUV composite binarizing prior to enforce 2-discrete mx in between x_min and x_max.

    Args:
        mx (np.ndarray): posterior means.
        Vx (np.ndarray): posterior variances.
        x_min (float): smallest admissible values. 
        x_max (float): largest admissible values. 

    Returns:
        mfx (np.ndarray): binarizing prior means.
        Vfx (np.ndarray): binarizing prior variances.
    """
    if np.any(x_min > x_max):
        raise ValueError(f"x_min cannot be larger than x_max")

    Vx_min = Vx + (mx - x_min)**2
    Vx_max = Vx + (mx - x_max)**2

    mfx = (x_min * Vx_max + x_max * Vx_min) / (Vx_min + Vx_max)
    Vfx = (Vx_min * Vx_max) / (Vx_min + Vx_max)

    return mfx, Vfx


# def m_ary_prior(mxm, Vxm, x_min, x_max, xlvl):
#     """
#     NUV composite binarizing prior to enforce xlvl-discrete mx in between x_min and x_max.

#     Args:
#         mxm (np.ndarray): posterior means for x_m, where x = x_1 + x_2 + ..., componentwise.
#         Vxm (np.ndarray): posterior variances for x_m, where x = x_1 + x_2 + ..., componentwise.
#         x_min (float): smallest admissible values. 
#         x_max (float): largest admissible values. 
#         xlvl (int): number of discretization levels.

#     Returns:
#         mfx (np.ndarray): discrete prior means.
#         Vfx (np.ndarray): discrete prior variances.
#     """
#     if np.any(x_min > x_max):
#         raise ValueError(f"x_min cannot be larger than x_max")

#     if xlvl - 1 != mxm.shape[-1] != Vxm.shape[-1]:
#         raise ValueError(f"xlvl, mxm, and Vxm must match")

#     mfxm, Vfxm = binary_prior(mxm, Vxm, x_min/(xlvl-1), x_max/(xlvl-1))

#     return mfxm, Vfxm
