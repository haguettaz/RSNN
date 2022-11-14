import torch


def fgmp_obs_blck(mx_f, Vx_f, my_b, Vy_b, A):
    """
    Forward Gaussian message passing through an observation block (Table V in Loeliger2016) with one scalar observation.

    Args:
        mx_f (torch.FloatTensor): X mean with one dimension of length K.
        Vx_f (torch.FloatTensor): X variance with two dimensions of length K.
        my_b (torch.FloatTensor): Y mean with zero dimension.
        Vy_b (torch.FloatTensor): Y variance with zero dimension.
        A (torch.FloatTensor): observation tensor one dimension of length K.

    Returns:
        (torch.FloatTensor): Z mean with one dimension of length K.
        (torch.FloatTensor): Z variance with two dimensions of length K.
    """
    if my_b.dim() > 0:
        raise ValueError("my_b must be a scalar.")

    Vx_f_At = Vx_f @ A
    g = 1 / (Vy_b + Vx_f_At @ A + 1e-12) # a scalar
    
    mz_f = mx_f + g * (my_b - A @ mx_f) * Vx_f_At
    Vz_f = Vx_f - g * torch.outer(Vx_f_At, Vx_f_At)

    return mz_f, Vz_f