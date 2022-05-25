import torch


def compute_posterior(mw_f, Vw_f, my_b, Vy_b, C):
    """
    Returns posterior means and variances according to Table V in Loeliger2016.

    Args:
        mw_f (torch.FloatTensor): forward weight means with size (L, K).
        Vw_f (torch.FloatTensor): forward weight variances with size (L, K).
        my_b (torch.FloatTensor): backward observation means with size (L, N, 2).
        Vy_b (torch.FloatTensor): backward observation variances with size (L, N, 2).
        C (torch.FloatTensor): observation matrix with size (L, N, 2, K).

    Returns:
        mw (torch.FloatTensor): posterior weight means with size (L, K).
        Vw (torch.FloatTensor): posterior weight variances with size (L, K).
        my (torch.FloatTensor): posterior observation means with size (L, N, 2).
    """

    L, N, _, K = C.size()

    cur_mw_f = mw_f.unsqueeze(-1)
    cur_Vw_f = torch.diag_embed(Vw_f)
    my_b = my_b.unsqueeze(-1)
    Vy_b = torch.diag_embed(Vy_b)

    for n in range(N):
        prev_mw_f = cur_mw_f
        prev_Vw_f = cur_Vw_f
        G_inv = Vy_b[:, n] + C[:, n] @ prev_Vw_f @ C.mT[:, n]
        cur_mw_f = prev_mw_f + prev_Vw_f @ C.mT[:, n] @ torch.linalg.solve(G_inv, my_b[:, n] - C[:, n] @ prev_mw_f)
        cur_Vw_f = prev_Vw_f - prev_Vw_f @ C.mT[:, n] @ torch.linalg.solve(G_inv, C[:, n] @ prev_Vw_f)

    my = (C @ cur_mw_f.view(L, 1, K, 1)).view(L, N, 2)
    mw = cur_mw_f.view(L, K)
    Vw = cur_Vw_f.diagonal(dim1=-2, dim2=-1)

    return mw, Vw, my


def compute_box_prior(mx, xmin=None, xmax=None, gamma=1):
    """
    Returns prior messages corresponding to box constraints between wmin and wmax (inclusive).

    Args:
        mx (torch.FloatTensor): posterior means with shape (..., C).
        xmin (float or torch.FloatTensor, optional): smallest admissible value. If torch.FloatTensor, it must be broadcastable with mx. Defaults to None.
        xmax (float or torch.FloatTensor, optional): largest admissible value. If torch.FloatTensor, it must be broadcastable with mx. Defaults to None.
        gamma (float, optional): _description_. Defaults to 1.

    Returns:
        mx_f (torch.FloatTensor): prior means with shape (..., C).
        Vx_f (torch.FloatTensor): prior variances with shape (..., C).
    """
    if xmin is None and xmax is None:
        raise ValueError(f"at least one of xmin, xmax must be finite")

    if xmin is None:
        sigma2x_max = (mx - xmax).abs()
        mx_f = xmax - sigma2x_max
        Vx_f = sigma2x_max / gamma
        return mx_f, Vx_f

    if xmax is None:
        sigma2x_min = (mx - xmin).abs()
        mx_f = xmin + sigma2x_min
        Vx_f = sigma2x_min / gamma
        return mx_f, Vx_f

    if torch.any(xmin > xmax):
        raise ValueError(f"xmin cannot be larger than xmax")

    if torch.all(xmin == xmax):  # equivalent to xmin Laplace prior
        mx_f = xmin * torch.ones_like(mx)
        Vx_f = (mx - xmin).abs() / gamma
        return mx_f, Vx_f

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
        xmin (float or torch.FloatTensor, optional): smallest admissible value. If torch.FloatTensor, it must be broadcastable with mx. Defaults to 0.
        xmax (float or torch.FloatTensor, optional): largest admissible value. If torch.FloatTensor, it must be broadcastable with mx. Defaults to 1.
        update (str, optional): update rule, either "am" or "em". Defaults to "am".

    Returns:
        mx_f (torch.FloatTensor): prior means with shape (..., C).
        Vx_f (torch.FloatTensor): prior variances with shape (..., C).
    """
    if torch.any(xmin > xmax):
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
        mx (torch.FloatTensor): posterior means with size (..., M-1, C).
        Vx (torch.FloatTensor): posterior variances with size (..., M-1, C).
        xmin (float or torch.FloatTensor): smallest admissible value. If torch.FloatTensor, it must be broadcastable with mx.
        xmax (float or torch.FloatTensor): largest admissible value. If torch.FloatTensor, it must be broadcastable with mx.
        M (int): number of levels.
        update (str): update rule, either "am" or "em". Defaults to "am".

    Returns:
        mx_f (torch.FloatTensor): prior means with shape (..., C).
        Vx_f (torch.FloatTensor): prior variances with shape (..., C).
    """
    if torch.any(xmin > xmax):
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
