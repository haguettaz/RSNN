import numpy as np


def fgmp_obs_blck(mx_forward, Vx_forward, my_backward, Vy_backward, A):
    """
    Forward Gaussian message passing through an observation block (Table V in Loeliger2016) with one scalar observation.

    Args:
        mx_forward (torch.FloatTensor): X mean with one dimension of length K.
        Vx_forward (torch.FloatTensor): X variance with two dimensions of length K.
        my_backward (torch.FloatTensor): Y mean with zero dimension.
        Vy_backward (torch.FloatTensor): Y variance with zero dimension.
        A (torch.FloatTensor): observation tensor one dimension of length K.

    Returns:
        (torch.FloatTensor): Z mean with one dimension of length K.
        (torch.FloatTensor): Z variance with two dimensions of length K.
    """
    assert my_backward.size == 1
    assert Vy_backward.size == 1

    Vx_forward_At = Vx_forward @ A
    
    g = 1 / (Vy_backward + np.inner(A, Vx_forward_At))
    
    mz_forward = mx_forward + g * (my_backward - np.inner(A, mx_forward)) * Vx_forward_At
    Vz_forward = Vx_forward - g * np.outer(Vx_forward_At, Vx_forward_At)

    return mz_forward, Vz_forward