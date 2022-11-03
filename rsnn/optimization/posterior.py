import torch


def compute_weight_posterior(mw_f, Vw_f, mz_b_f, Vz_b_f, C_f, mz_b_a, Vz_b_a, C_a, mz_b_s, Vz_b_s, C_s):
    """
    Returns posterior weight means and variances according to Table V in Loeliger2016.

    Args:
        mw_f (_type_): prior weight mean with size (K, 1).
        Vw_f (_type_): prior weight variance with size (K, K).
        mz_b_f (_type_): prior potential mean for firing times with size (K, 1).
        Vz_b_f (_type_): prior potential variance for firing times with size (K, K).
        C_f (_type_): observation tensor for firing times with size (N_f, 2, K).
        mz_b_a (_type_): prior potential mean for active times with size (K, 1).
        Vz_b_a (_type_): prior potential variance for active times with size (K, K).
        C_a (torch.FloatTensor): observation tensor for active times with size (N_a, 1, K).
        mz_b_s (_type_): prior potential mean for silent times with size (K, 1).
        Vz_b_s (_type_): prior potential variance for silent times with size (K, K).
        C_s (_type_): observation tensor for silent times with size (N_s, 1, K).

    Returns:
        _type_: _description_
    """
    N_f, N_a, N_s = C_f.size(0), C_a.size(0), C_s.size(0)

    prev_mw_f = mw_f.clone()
    prev_Vw_f = Vw_f.clone()

    # firing times = 1 constraint on the potential and 1 constraint on its derivative
    for n in range(N_f):
        H = torch.linalg.solve(Vz_b_f[n] + C_f[n] @ prev_Vw_f @ C_f[n].T + 1e-9, C_f[n] @ prev_Vw_f).T
        mw_f = prev_mw_f + H @ (mz_b_f[n] - C_f[n] @ prev_mw_f)
        Vw_f = prev_Vw_f - H @ C_f[n] @ prev_Vw_f
        prev_mw_f = mw_f.clone()
        prev_Vw_f = Vw_f.clone()

    # active periods = 1 constraint on the potential derivative
    for n in range(N_a):
        H = prev_Vw_f @ C_a[n].T / (Vz_b_a[n] + C_a[n] @ prev_Vw_f @ C_a[n].T + 1e-9)
        mw_f = prev_mw_f + H @ (mz_b_a[n] - C_a[n] @ prev_mw_f)
        Vw_f = prev_Vw_f - H @ C_a[n] @ prev_Vw_f
        prev_mw_f = mw_f.clone()
        prev_Vw_f = Vw_f.clone()

    # silent periods = 1 constraint on the potential
    for n in range(N_s):
        H = prev_Vw_f @ C_s[n].T / (Vz_b_s[n] + C_s[n] @ prev_Vw_f @ C_s[n].T + 1e-9)
        mw_f = prev_mw_f + H @ (mz_b_s[n] - C_s[n] @ prev_mw_f)
        Vw_f = prev_Vw_f - H @ C_s[n] @ prev_Vw_f
        prev_mw_f = mw_f.clone()
        prev_Vw_f = Vw_f.clone()

    return mw_f, Vw_f
