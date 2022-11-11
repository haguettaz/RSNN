import torch

def compute_box_prior(mx, xmin=None, xmax=None, gamma=1):
    """
    Returns prior messages corresponding to box constraints between xmin and xmax (inclusive).

    Args:
        mx (torch.FloatTensor): posterior means.
        xmin (float, optional): smallest admissible value. Defaults to None.
        xmax (float, optional): largest admissible value. Defaults to None.
        gamma (float, optional): constraint' hardness parameter, the smaller the softer. Defaults to 1.

    Returns:
        mx_f (torch.FloatTensor): prior means.
        Vx_f (torch.FloatTensor): prior variances.
    """
    if xmin is None and xmax is None:
        return mx, 1e12 * torch.ones_like(mx)

    if xmin is None: # half-space
        sigma2x_max = (mx - xmax).abs()
        mx_f = xmax - sigma2x_max
        Vx_f = sigma2x_max / gamma
        return mx_f, Vx_f

    if xmax is None: # half-space
        sigma2x_min = (mx - xmin).abs()
        mx_f = xmin + sigma2x_min
        Vx_f = sigma2x_min / gamma
        return mx_f, Vx_f

    if xmin == xmax: # laplace
        mx_f = xmin * torch.ones_like(mx)
        Vx_f = (mx - xmin).abs() / gamma
        return mx_f, Vx_f

    if xmin > xmax: 
        raise ValueError(f"xmin cannot be larger than xmax")

    sigma2x_min = (mx - xmin).abs()
    sigma2x_max = (mx - xmax).abs()
    mx_f = (xmin * sigma2x_max + xmax * sigma2x_min) / (sigma2x_min + sigma2x_max)
    Vx_f = (sigma2x_min * sigma2x_max) / (sigma2x_min + sigma2x_max) / gamma
    return mx_f, Vx_f


def compute_binary_prior(mx, Vx=None, xmin=0, xmax=1, update="am"):
    """
    Returns prior messages corresponding to binary constraints between wmin and wmax.

    Args:
        mx (torch.FloatTensor): posterior means with size (..., C).
        Vx (torch.FloatTensor): posterior variances with size (..., C). Defaults to None.
        xmin (float, optional): smallest admissible value. Defaults to 0.
        xmax (float, optional): largest admissible value. Defaults to 1.
        update (str, optional): update rule, either "am" or "em". Defaults to "am".

    Returns:
        mx_f (torch.FloatTensor): prior means with shape (..., C).
        Vx_f (torch.FloatTensor): prior variances with shape (..., C).
    """
    if xmin > xmax:
        raise ValueError(f"xmin cannot be larger than xmax")

    if update not in {"am", "em"}:
        raise ValueError(f"update must be 'am' or 'em'")

    if Vx is None and update == "em":
        raise ValueError(f"posterior variances are required for EM update rule")

    if update == "am":
        Vx_min = (mx - xmin).pow(2) + 1e-9
        Vx_max = (mx - xmax).pow(2) + 1e-9
    else:
        Vx_min = Vx + (mx - xmin).pow(2)
        Vx_max = Vx + (mx - xmax).pow(2)

    mx_f = (xmin * Vx_max + xmax * Vx_min) / (Vx_min + Vx_max)
    Vx_f = (Vx_min * Vx_max) / (Vx_min + Vx_max)

    return mx_f, Vx_f


def compute_m_ary_prior(mx, Vx, xmin, xmax, M, update="am"):
    """
    Returns prior messages corresponding to M-level constraints between wmin and wmax (inclusive).

    Args:
        mx (torch.FloatTensor): posterior means with size (..., M-1, L).
        Vx (torch.FloatTensor): posterior variances with size (..., M-1, L).
        xmin (float): smallest admissible value.
        xmax (float): largest admissible value.
        M (int): number of levels.
        update (str): update rule, either "am" or "em". Defaults to "am".

    Returns:
        mx_f (torch.FloatTensor): prior means with shape (..., L).
        Vx_f (torch.FloatTensor): prior variances with shape (..., L).
    """
    if xmin > xmax:
        raise ValueError(f"xmin cannot be larger than xmax")

    if update not in {"am", "em"}:
        raise ValueError(f"update must be 'am' or 'em'")

    if Vx is None and update == "em":
        raise ValueError(f"posterior variances are required for EM update rule")

    mx_f = torch.empty_like(mx)
    Vx_f = torch.empty_like(mx)

    for m in range(M - 1):
        mx_f[..., m, :], Vx_f[..., m, :] = compute_binary_prior(
            mx[..., m, :], Vx[..., m, :] if update == "em" else None, xmin / (M - 1), xmax / (M - 1), update
        )

    return mx_f, Vx_f
