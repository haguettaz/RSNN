import torch

from .gmp import fgmp_obs_blck
from .nuv import compute_box_prior


def optimize(mw, C, nuv, err, max_iter=1000, err_tol=1e-3, return_err=False, device=None):

    # Assume mw is initialized in the correct range
    C_f, C_a, C_s = C
    err_w, err_f, err_a, err_s = err
    nuv_w, nuv_f, nuv_a, nuv_s = nuv

    # K = C_f.size(0), C_a.size(0), C_s.size(0), C_f.size(1)
    if device is not None:
        mw = mw.to(device)
        C_f, C_a, C_s = C_f.to(device), C_a.to(device), C_s.to(device)

    # mw = torch.FloatTensor(K).uniform_(-wb, wb).to(device)
    mz_f, mz_a, mz_s = C_f @ mw, C_a @ mw, C_s @ mw

    # compute_wb_err = lambda w_: ((w_ - wb).abs() + (w_ + wb).abs() - 2 * wb).sum()
    # compute_theta_err = lambda z_: (z_ - theta).abs().sum()
    # compute_a_err = lambda z_: ((z_ - a).abs() - (z_ - a)).sum()
    # compute_b_err = lambda z_: ((z_ - b).abs() + (z_ - b)).sum()

    for itr in range(max_iter):
        # compute the priors
        # mw_f, Vw_f = compute_box_prior(mw, -wb, wb, gamma_wb)
        # mz_b_theta, Vz_b_theta = compute_box_prior(mz_theta, theta, theta, gamma_theta)
        # mz_b_a, Vz_b_a = compute_box_prior(mz_a, a, None, gamma_a)
        # mz_b_b, Vz_b_b = compute_box_prior(mz_b, None, b, gamma_b)

        # compute the priors
        mw_f, Vw_f = nuv_w(mw)
        mz_b_f, Vz_b_f = nuv_f(mz_f)
        mz_b_a, Vz_b_a = nuv_a(mz_a)
        mz_b_s, Vz_b_s = nuv_s(mz_s)

        # compute the posteriors
        mw, _ = compute_weight_posterior(mw_f, Vw_f, mz_b_f, Vz_b_f, mz_b_a, Vz_b_a, mz_b_s, Vz_b_s, C_f, C_a, C_s)
        mz_f, mz_a, mz_s = C_f @ mw, C_a @ mw, C_s @ mw

        # stopping criterion
        if err_w(mw) < err_tol and err_f(mz_f) < err_tol and err_a(mz_a) < err_tol and err_s(mz_s) < err_tol:
            print(f"Optimization problem solved after {itr} iterations!", flush=True)
            break

    if return_err:
        return mw, (err_w(mw), err_f(mz_f), err_a(mz_a), err_s(mz_s))

    return mw

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