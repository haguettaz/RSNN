import torch

from .posterior import compute_weight_posterior
from .prior import compute_box_prior


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
