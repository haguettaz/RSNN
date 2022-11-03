import torch


def compute_posterior(mw_f, Vw_f, my_b, Vy_b, C):
    """
    Returns posterior means and variances according to Table V in Loeliger2016.

    Args:
        mw_f (torch.FloatTensor): forward weight means with size (L, K).
        Vw_f (torch.FloatTensor): forward weight variances with size (L, K).
        my_b (torch.FloatTensor): backward observation means with size (N, L, D).
        Vy_b (torch.FloatTensor): backward observation variances with size (N, L, D).
        C (torch.FloatTensor): observation tensor with size (N, L, D, K).

    Returns:
        mw (torch.FloatTensor): posterior weight means with size (L, K).
        Vw (torch.FloatTensor): posterior weight variances with size (L, K).
        my (torch.FloatTensor): posterior observation means with size (N, L, D).
    """

    N, L, D, K = C.size()

    cur_mw_f = mw_f.unsqueeze(-1)
    cur_Vw_f = torch.diag_embed(Vw_f)
    my_b = my_b.unsqueeze(-1)
    Vy_b = torch.diag_embed(Vy_b)

    for n in range(N):
        prev_mw_f = cur_mw_f
        prev_Vw_f = cur_Vw_f
        G_inv = Vy_b[n] + C[n] @ prev_Vw_f @ C.mT[n]

        # to prevent singular matrices, we add 1e-9
        cur_mw_f = prev_mw_f + prev_Vw_f @ C.mT[n] @ torch.linalg.solve(G_inv, my_b[n] - C[n] @ prev_mw_f + 1e-9)
        cur_Vw_f = prev_Vw_f - prev_Vw_f @ C.mT[n] @ torch.linalg.solve(G_inv, C[n] @ prev_Vw_f + 1e-9)

    my = (C @ cur_mw_f.view(1, L, K, 1)).view(N, L, D)
    mw = cur_mw_f.view(L, K)
    Vw = cur_Vw_f.diagonal(dim1=-2, dim2=-1)

    return mw, Vw, my
