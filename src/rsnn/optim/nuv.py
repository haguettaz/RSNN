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
        mx_f (np.FloatTensor): prior means.
        Vx_f (np.FloatTensor): prior variances.
    """
    mask_finite_xmin = np.isfinite(xmin)
    mask_finite_xmax = np.isfinite(xmax)

    mx_f = np.empty_like(mx)
    Vx_f = np.empty_like(mx)

    # no constraints
    mask = (~mask_finite_xmin) & (~mask_finite_xmax)
    mx_f[mask] = mx[mask]
    Vx_f[mask] = 1e9

    # left half-space
    mask = (~mask_finite_xmin) & mask_finite_xmax
    sigma2x_max = np.abs(mx[mask] - xmax[mask])
    mx_f[mask] = xmax[mask] - sigma2x_max
    Vx_f[mask] = sigma2x_max / gamma

    # right half-space
    mask = mask_finite_xmin & (~mask_finite_xmax)
    sigma2x_min = np.abs(mx[mask] - xmin[mask])
    mx_f[mask] = xmin[mask] + sigma2x_min
    Vx_f[mask] = sigma2x_min / gamma

    # laplace
    mask = mask_finite_xmin & mask_finite_xmax & (xmin == xmax)
    sigma2 = np.abs(mx[mask] - xmin[mask])
    mx_f[mask] = xmin[mask]
    Vx_f[mask] = (sigma2) / gamma

    # box
    mask = mask_finite_xmin & mask_finite_xmax & (xmin < xmax)
    sigma2x_min = np.abs(mx[mask] - xmin[mask])
    sigma2x_max = np.abs(mx[mask] - xmax[mask])
    mx_f[mask] = (xmin[mask] * sigma2x_max + xmax[mask] * sigma2x_min) / (sigma2x_min + sigma2x_max)
    Vx_f[mask] = (sigma2x_min * sigma2x_max) / (sigma2x_min + sigma2x_max) / gamma

    return mx_f, Vx_f


def binary_prior(mx, Vx=None, xmin=0, xmax=1, update="am"):
    """
    Returns prior messages corresponding to binary constraints between wmin and wmax.

    Args:
        mx (np.FloatTensor): posterior means with size (..., C).
        Vx (np.FloatTensor): posterior variances with size (..., C). Defaults to None.
        xmin (float, optional): smallest admissible value. Defaults to 0.
        xmax (float, optional): largest admissible value. Defaults to 1.
        update (str, optional): update rule, either "am" or "em". Defaults to "am".

    Returns:
        mx_f (np.FloatTensor): prior means with shape (..., C).
        Vx_f (np.FloatTensor): prior variances with shape (..., C).
    """
    if xmin > xmax:
        raise ValueError(f"xmin cannot be larger than xmax")

    if update not in {"am", "em"}:
        raise ValueError(f"update must be 'am' or 'em'")

    if Vx is None and update == "em":
        raise ValueError(f"posterior variances are required for EM update rule")

    if update == "am":
        Vx_min = (mx - xmin)**2 + 1e-9
        Vx_max = (mx - xmax)**2 + 1e-9
    else:
        Vx_min = Vx + (mx - xmin)**2
        Vx_max = Vx + (mx - xmax)**2

    mx_f = (xmin * Vx_max + xmax * Vx_min) / (Vx_min + Vx_max)
    Vx_f = (Vx_min * Vx_max) / (Vx_min + Vx_max)

    return mx_f, Vx_f


def m_ary_prior(mx, Vx, xmin, xmax, M, update="am"):
    """
    Returns prior messages corresponding to M-level constraints between wmin and wmax (inclusive).

    Args:
        mx (np.FloatTensor): posterior means with size (..., M-1, L).
        Vx (np.FloatTensor): posterior variances with size (..., M-1, L).
        xmin (float): smallest admissible value.
        xmax (float): largest admissible value.
        M (int): number of levels.
        update (str): update rule, either "am" or "em". Defaults to "am".

    Returns:
        mx_f (np.FloatTensor): prior means with shape (..., L).
        Vx_f (np.FloatTensor): prior variances with shape (..., L).
    """
    if xmin > xmax:
        raise ValueError(f"xmin cannot be larger than xmax")

    if update not in {"am", "em"}:
        raise ValueError(f"update must be 'am' or 'em'")

    if Vx is None and update == "em":
        raise ValueError(f"posterior variances are required for EM update rule")

    mx_f = np.empty_like(mx)
    Vx_f = np.empty_like(mx)

    for m in range(M - 1):
        mx_f[..., m, :], Vx_f[..., m, :] = binary_prior(
            mx[..., m, :], Vx[..., m, :] if update == "em" else None, xmin / (M - 1), xmax / (M - 1), update
        )

    return mx_f, Vx_f
