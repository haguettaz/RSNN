import torch

from .posterior import compute_weight_posterior
from .prior import compute_box_prior


def compute_weights(
    C_theta, C_a, C_b, wb, theta, a, b, gamma_wb, gamma_theta, gamma_a, gamma_b, max_iter=1000, tol=1e-3
):

    N_theta, N_a, N_b = C_theta.size(0), C_a.size(0), C_b.size(0)
    K = C_theta.size(2)

    mw = torch.FloatTensor(K).uniform_(-wb, wb)
    mz_theta = (C_theta @ mw).squeeze()
    mz_a = (C_a @ mw).squeeze()
    mz_b = (C_b @ mw).squeeze()

    compute_wb_err = lambda w_: ((w_ - wb).abs() + (w_ + wb).abs() - 2 * wb).sum()
    compute_theta_err = lambda z_: (z_ - theta).abs().sum()
    compute_a_err = lambda z_: ((z_ - a).abs() - (z_ - a)).sum()
    compute_b_err = lambda z_: ((z_ - b).abs() + (z_ - b)).sum()

    for itr in range(max_iter):
        # compute the priors
        mw_f, Vw_f = compute_box_prior(mw, -wb, wb, gamma_wb)
        mz_b_theta, Vz_b_theta = compute_box_prior(mz_theta, theta, theta, gamma_theta)
        mz_b_a, Vz_b_a = compute_box_prior(mz_a, a, None, gamma_a)
        mz_b_b, Vz_b_b = compute_box_prior(mz_b, None, b, gamma_b)

        # compute the posteriors
        mw, _ = compute_weight_posterior(
            mw_f.view(K, 1),
            torch.diag(Vw_f),
            mz_b_theta.view(N_theta, 1, 1),
            Vz_b_theta.view(N_theta, 1, 1),
            C_theta,
            mz_b_a.view(N_a, 1, 1),
            Vz_b_a.view(N_a, 1, 1),
            C_a,
            mz_b_b.view(N_b, 1, 1),
            Vz_b_b.view(N_b, 1, 1),
            C_b,
        )
        mz_theta = (C_theta @ mw).squeeze()
        mz_a = (C_a @ mw).squeeze()
        mz_b = (C_b @ mw).squeeze()

        # compute the error
        wb_err = compute_wb_err(mw)
        theta_err = compute_theta_err(mz_theta)
        a_err = compute_a_err(mz_a)
        b_err = compute_b_err(mz_b)

        if wb_err < tol and theta_err < tol and a_err < tol and b_err < tol:
            print(f"Optimization problem solved after {itr} iterations!", flush=True)
            break

    print("Errors: ", wb_err, theta_err, a_err, b_err, flush=True)

    return mw.squeeze()
