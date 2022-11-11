import torch

def compute_weight_posterior(mw_f, Vw_f, mz_b_f, Vz_b_f, mz_b_a, Vz_b_a, mz_b_s, Vz_b_s, C_f, C_a, C_s):
    """
    Returns posterior weight means and variances according to Table V in Loeliger2016.

    Args:
        mw_f (_type_): prior weight mean with size (K).
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
    prev_mw_f = mw_f.clone()
    prev_Vw_f = Vw_f.diag()

    N_f, N_a, N_s = C_f.size(0), C_a.size(0), C_s.size(0)

    # Equality Constraints at Firing Times
    for n in range(N_f):
        mw_f, Vw_f = fgmp_obs_blck(prev_mw_f, prev_Vw_f, mz_b_f[n], Vz_b_f[n], C_f[n])
        prev_mw_f, prev_Vw_f = mw_f.clone(), Vw_f.clone()

    # Unequality Constraints at Active Times
    for n in range(N_a):
        mw_f, Vw_f = fgmp_obs_blck(prev_mw_f, prev_Vw_f, mz_b_a[n], Vz_b_a[n], C_a[n])
        prev_mw_f, prev_Vw_f = mw_f.clone(), Vw_f.clone()

    # Unequality Constraints at Silent Times
    for n in range(N_s):
        mw_f, Vw_f = fgmp_obs_blck(prev_mw_f, prev_Vw_f, mz_b_s[n], Vz_b_s[n], C_s[n])
        prev_mw_f, prev_Vw_f = mw_f.clone(), Vw_f.clone()

    return mw_f, Vw_f.diag()

def fgmp_obs_blck(mx_f, Vx_f, my_b, Vy_b, A):
    """
    Forward Gaussian message passing through an observation block (Table V in Loeliger2016) with scalar observation.

    Args:
        mx_f (_type_): X mean with one dimension of length K.
        Vx_f (_type_): X variance with two dimensions of length K.
        my_b (_type_): Y mean with zero dimension.
        Vy_b (_type_): Y variance with zero dimension.
        A (_type_): observation tensor one dimension of length K.

    Returns:
        _type_: _description_
    """
    if my_b.dim() > 0:
        raise ValueError("my_b must be a scalar.")

    Vx_f_At = Vx_f @ A
    g = 1 / (Vy_b + Vx_f_At @ A + 1e-12) # a scalar
    
    mz_f = mx_f + g * (my_b - A @ mx_f) * Vx_f_At
    Vz_f = Vx_f - g * (Vx_f_At.view(-1, 1) @ Vx_f_At.view(1, -1))

    return mz_f, Vz_f