def compute_weight_posterior(mw_f, Vw_f, mz_b_f, Vz_b_f, mz_b_a, Vz_b_a, mz_b_s, Vz_b_s, C_f, C_a, C_s):
    """
    Compute the weight posterior means and variances my forward Gaussian message passing.

    Args:
        mw_f (torch.FloatTensor): weight prior mean with one dimension of length K.
        Vw_f (torch.FloatTensor): weight prior variances with one dimension of length K.
        mz_b_f (torch.FloatTensor): firing observations (potential) prior means with one dimension of length N_f.
        Vz_b_f (torch.FloatTensor): firing observations (potential) prior means with one dimension of length N_f.
        mz_b_a (torch.FloatTensor): active observations (potential derivative) prior means with one dimension of length N_a.
        Vz_b_a (torch.FloatTensor): active observations (potential derivative) prior variances with one dimension of length N_a.
        mz_b_s (torch.FloatTensor): silent observations (potential) prior means with one dimension of length N_s.
        Vz_b_s (torch.FloatTensor): silent observations (potential) prior variances with one dimension of length N_s.
        C_f (torch.FloatTensor): firing observation tensor with two dimensions of length N_f and K.
        C_a (torch.FloatTensor): firing observation tensor with two dimensions of length N_a and K.
        C_s (torch.FloatTensor): firing observation tensor with two dimensions of length N_s and K.

    Returns:
        (torch.FloatTensor): weight posterior means with one dimension of length K.
        (torch.FloatTensor): weight posterior variances with one dimension of length K.
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
    Forward Gaussian message passing through an observation block (Table V in Loeliger2016) with one scalar observation.

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