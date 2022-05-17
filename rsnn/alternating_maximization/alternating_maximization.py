import torch

from .utils import get_mask_around_firing, get_mask_at_firing, get_mask_refractory_period


def compute_prior_messages(mw, my, wlim, theta, eta, Tr, eps, dymin, firing_sequences):
    """_summary_

    Args:
        mw (torch.FloatTensor): posterior weight mean vectors with size (L, 1, K, 1).
        my (torch.FloatTensor): posterior observation mean vectors with size (L, N, 2, 1).
        wlim (tuple): min and max weights.
        theta (float): firing threshold.
        eta (float): max disturbance.
        Tr (int): refractory period.
        eps (int): _description_
        dymin (float): min observation increase around firing.
        firing_sequences (torch.BoolTensor): firing sequences tensor with size (L, N).

    Returns:
        Vw_f (torch.FloatTensor): forward weight covariance matrices with size (L, 1, K, K).
        mw_f (torch.FloatTensor): forward weight mean vectors with size (L, 1, K, 1).
        Vy_b (torch.FloatTensor): backward observation covariance matrices with size (L, N, 2, 2).
        my_b (torch.FloatTensor): backward observation mean vectors with size (L, N, 2, 1).
    """
    L, N = firing_sequences.size()
    K = mw.size(2)

    ## Weights
    wmin, wmax = wlim
    sigma_a_2 = 1e-1 * torch.abs(mw - wmin)
    sigma_b_2 = 1e-1 * torch.abs(mw - wmax)
    Vw_f = torch.diag_embed(((sigma_a_2 * sigma_b_2) / (sigma_a_2 + sigma_b_2)).view(L, 1, K))
    mw_f = (wmax * sigma_a_2 + wmin * sigma_b_2) / (sigma_a_2 + sigma_b_2)

    ## Action potentials
    mask_at_firing = get_mask_at_firing(firing_sequences)
    mask_refractory_period = get_mask_refractory_period(firing_sequences, Tr, eps)
    mask_around_firing = get_mask_around_firing(firing_sequences, eps)
    mask_before_firing = ~(mask_refractory_period | mask_around_firing)

    Vy_b = torch.diag_embed(1e9 * torch.ones(L, N, 2), dim1=2, dim2=3)
    my_b = torch.zeros(L, N, 2, 1)

    # before a firing, action potential is below theta - eta
    Vy_b[..., 0, 0][mask_before_firing] = 1e-1 * torch.abs(my[..., 0, 0][mask_before_firing] - eta)
    my_b[..., 0, 0][mask_before_firing] = -torch.abs(my[..., 0, 0][mask_before_firing] - eta) + eta

    # around a firing, slope of the action potential is above dymin
    Vy_b[..., 1, 1][mask_around_firing] = 1e-1 * torch.abs(my[..., 1, 0][mask_around_firing] - dymin)
    my_b[..., 1, 0][mask_around_firing] = torch.abs(my[..., 1, 0][mask_around_firing] - dymin) + dymin

    # at firing, action potential is theta
    Vy_b[..., 0, 0][mask_at_firing] = 1e-2 * torch.abs(my[..., 0, 0][mask_at_firing] - theta)
    my_b[..., 0, 0][mask_at_firing] = theta

    return Vw_f, mw_f, Vy_b, my_b


def compute_posterior_means(Vw_f, mw_f, Vy_b, my_b, C):
    """_summary_

    Args:
        Vw_f (torch.FloatTensor): forward weight covariance matrices with size (L, 1, K, K).
        mw_f (torch.FloatTensor): forward weight mean vectors with size (L, 1, K, 1).
        Vy_b (torch.FloatTensor): backward observation covariance matrices with size (L, N, 2, 2).
        my_b (torch.FloatTensor): backward observation mean vectors with size (L, N, 2, 1).
        C (torch.FloatTensor): observation matrix with size (L, N, 2, K).

    Returns:
        mw (torch.FloatTensor): posterior weight mean vectors with size (L, 1, K, 1).
        my (torch.FloatTensor): posterior observation mean vectors with size (L, N, 2, 1).
    """

    L, N, _, K = C.size()

    cur_Vw_f = Vw_f[:, 0]
    cur_mw_f = mw_f[:, 0]

    for n in range(N):
        prev_Vw_f = cur_Vw_f
        prev_mw_f = cur_mw_f
        G_inv = Vy_b[:, n] + C[:, n] @ prev_Vw_f @ C.mT[:, n]
        cur_Vw_f = prev_Vw_f - prev_Vw_f @ C.mT[:, n] @ torch.linalg.solve(G_inv, C[:, n] @ prev_Vw_f)
        cur_mw_f = prev_mw_f + prev_Vw_f @ C.mT[:, n] @ torch.linalg.solve(G_inv, my_b[:, n] - C[:, n] @ prev_mw_f)

    mw = cur_mw_f.view(L, 1, K, 1)
    my = C @ mw

    return mw, my
