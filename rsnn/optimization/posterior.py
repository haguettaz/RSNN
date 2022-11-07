from time import time

import torch

from ..utils.utils import inv_2x2


def compute_weight_posterior(mw_f, Vw_f, mz_b_theta, Vz_b_theta, C_theta, mz_b_a, Vz_b_a, C_a, mz_b_b, Vz_b_b, C_b):
    """
    Returns posterior weight means and variances according to Table V in Loeliger2016.

    Args:
        mw_f (_type_): prior weight mean with size (K, 1).
        Vw_f (_type_): prior weight variance with size (K, K).
        mz_b_f (_type_): prior potential mean for firing times with size (N_f, 2, 1).
        Vz_b_f (_type_): prior potential variance for firing times with size (N_f, 2, 2).
        C_f (_type_): observation tensor for firing times with size (N_f, 2, K).
        mz_b_a (_type_): prior potential mean for active times with size (N_a, 1, 1).
        Vz_b_a (_type_): prior potential variance for active times with size (N_a, 1, 1).
        C_a (torch.FloatTensor): observation tensor for active times with size (N_a, 1, K).
        mz_b_s (_type_): prior potential mean for silent times with size (N_s, 1, 1).
        Vz_b_s (_type_): prior potential variance for silent times with size (N_s, 1, 1).
        C_s (_type_): observation tensor for silent times with size (N_s, 1, K).

    Returns:
        _type_: _description_
    """
    N_theta, N_a, N_b = C_theta.size(0), C_a.size(0), C_b.size(0)

    prev_mw_f = mw_f.clone()
    prev_Vw_f = Vw_f.clone()

    # Equality Constraints at Firing Times
    for n in range(N_theta):
        # H = torch.linalg.solve(Vz_b_f[n] + C_f[n] @ prev_Vw_f @ C_f[n].T + 1e-9, C_f[n] @ prev_Vw_f).T
        # mw_f = prev_mw_f + H @ (mz_b_f[n] - C_f[n] @ prev_mw_f)
        # Vw_f = prev_Vw_f - H @ C_f[n] @ prev_Vw_f
        H = prev_Vw_f @ C_theta[n].T / (Vz_b_theta[n] + C_theta[n] @ prev_Vw_f @ C_theta[n].T)
        # H = G * prev_Vw_f @ C_f[n].T
        mw_f = prev_mw_f + H @ (mz_b_theta[n] - C_theta[n] @ prev_mw_f)
        Vw_f = prev_Vw_f - H @ C_theta[n] @ prev_Vw_f
        prev_mw_f = mw_f.clone()
        prev_Vw_f = Vw_f.clone()

    # Unequality Constraints at Active Times
    for n in range(N_a):
        H = prev_Vw_f @ C_a[n].T / (Vz_b_a[n] + C_a[n] @ prev_Vw_f @ C_a[n].T)
        mw_f = prev_mw_f + H @ (mz_b_a[n] - C_a[n] @ prev_mw_f)
        Vw_f = prev_Vw_f - H @ C_a[n] @ prev_Vw_f
        # G = 1 / (Vz_b_a[n] + C_a[n] @ prev_Vw_f @ C_a[n].T + 1e-9)
        # H = prev_Vw_f @ C_a[n].T * G
        # mw_f = prev_mw_f + H @ (mz_b_a[n] - C_a[n] @ prev_mw_f)
        # Vw_f = prev_Vw_f - H @ C_a[n] @ prev_Vw_f
        prev_mw_f = mw_f.clone()
        prev_Vw_f = Vw_f.clone()

    # Unequality Constraints at Silent Times
    for n in range(N_b):
        H = prev_Vw_f @ C_b[n].T / (Vz_b_b[n] + C_b[n] @ prev_Vw_f @ C_b[n].T)
        mw_f = prev_mw_f + H @ (mz_b_b[n] - C_b[n] @ prev_mw_f)
        Vw_f = prev_Vw_f - H @ C_b[n] @ prev_Vw_f
        # G = 1 / (Vz_b_s[n] + C_s[n] @ prev_Vw_f @ C_s[n].T + 1e-9)
        # H = prev_Vw_f @ C_s[n].T * G
        # mw_f = prev_mw_f + H @ (mz_b_s[n] - C_s[n] @ prev_mw_f)
        # Vw_f = prev_Vw_f - H @ C_s[n] @ prev_Vw_f
        prev_mw_f = mw_f.clone()
        prev_Vw_f = Vw_f.clone()

    return mw_f.flatten(), Vw_f.diag()
