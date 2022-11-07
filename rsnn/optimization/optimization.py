import torch

from .posterior import compute_weight_posterior
from .prior import compute_box_prior


def compute_weights(C_f, C_a, C_s, wb, theta, a, b, gamma_wb, gamma_theta, gamma_a, gamma_b, num_iter=200, tol=1e-3):

    N_f, N_a, N_s = C_f.size(0), C_a.size(0), C_s.size(0)
    K = C_f.size(2)

    mw_f = torch.FloatTensor(K, 1).uniform_(-wb, wb)
    Vw_f = torch.eye(K)

    mz_b_f = torch.ones(N_f, 2, 1)
    mz_b_f[:, 0] *= theta
    mz_b_f[:, 1] *= a
    Vz_b_f = torch.eye(2).repeat(N_f, 1, 1)

    mz_b_a = torch.ones(N_a, 1, 1) * a
    Vz_b_a = torch.ones(N_a, 1, 1)

    mz_b_s = torch.ones(N_s, 1, 1) * b
    Vz_b_s = torch.ones(N_s, 1, 1)

    compute_wb_err = lambda w_: gamma_wb * ((w_ - wb).abs() + (w_ + wb).abs() - 2 * wb).sum()
    compute_theta_err = lambda z_: gamma_theta * (z_ - theta).abs().sum()
    compute_a_err = lambda z_: gamma_a * ((z_ - a).abs() - (z_ - a)).sum()
    compute_b_err = lambda z_: gamma_b * ((z_ - b).abs() + (z_ - b)).sum()

    for itr in range(num_iter):
        # posterior
        mw, _ = compute_weight_posterior(mw_f, Vw_f, mz_b_f, Vz_b_f, C_f, mz_b_a, Vz_b_a, C_a, mz_b_s, Vz_b_s, C_s)
        mz_f = C_f @ mw
        mz_a = C_a @ mw
        mz_s = C_s @ mw

        print(mz_b_f[:, 0, 0], mz_f[:, 0, 0])

        wb_err = compute_wb_err(mw[:, 0])
        theta_err = compute_theta_err(mz_f[:, 0, 0])
        a_err = compute_a_err(mz_f[:, 1, 0]) + compute_a_err(mz_a[:, 0, 0])
        b_err = compute_b_err(mz_s[:, 0, 0])

        print("Optimization update done with errors: ", wb_err, theta_err, a_err, b_err, flush=True)

        if wb_err < tol and theta_err < tol and a_err < tol and b_err < tol:
            print(f"Optimization problem solved after {itr} iterations!", flush=True)
            break

        # priors
        mw_f[:, 0], Vw_f[torch.arange(K), torch.arange(K)] = compute_box_prior(mw[:, 0], -wb, wb, gamma_wb)
        mz_b_f[:, 0, 0], Vz_b_f[:, 0, 0] = compute_box_prior(mz_f[:, 0, 0], theta, theta, gamma_theta)
        mz_b_f[:, 1, 0], Vz_b_f[:, 1, 1] = compute_box_prior(mz_f[:, 1, 0], a, None, gamma_a)
        mz_b_a[:, 0, 0], Vz_b_a[:, 0, 0] = compute_box_prior(mz_a[:, 0, 0], a, None, gamma_a)
        mz_b_s[:, 0, 0], Vz_b_s[:, 0, 0] = compute_box_prior(mz_s[:, 0, 0], None, b, gamma_b)

    print("Optimization done with errors: ", wb_err, theta_err, a_err, b_err, flush=True)

    return mw.squeeze()
