import torch

from .posterior import compute_weight_posterior
from .prior import compute_box_prior


def compute_weights(
    l,
    return_dict,
    C_f,
    C_a,
    C_s,
    wb,
    theta,
    a,
    b,
    gamma_wb=1,
    gamma_theta=1,
    gamma_a=1,
    gamma_b=1,
    num_iter=200,
    tol=1e-3,
):

    N_a, N_f, N_s = C_a.size(0), C_f.size(0), C_s.size(0)
    K = C_a.size(2)

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

    compute_w_err = lambda w_: gamma_wb * ((w_ - wb).abs() + (w_ + wb).abs() - 2 * wb).sum()
    compute_z_f_err = (
        lambda z_f: gamma_theta * (z_f[:, 0] - theta).abs().sum()
        + gamma_a * ((z_f[:, 1] - a).abs() + (z_f[:, 1] - 1e12).abs() - 1e12).sum()
    )
    compute_z_a_err = lambda z_a: gamma_a * ((z_a[:, 0] - a).abs() + (z_a[:, 0] - 1e12).abs() - 1e12).sum()
    compute_z_s_err = lambda z_s: gamma_b * ((z_s[:, 0] - b).abs() + (z_s[:, 0] + 1e12).abs() - 1e12).sum()

    for itr in range(num_iter):
        # posterior
        mw, _ = compute_weight_posterior(mw_f, Vw_f, mz_b_f, Vz_b_f, C_f, mz_b_a, Vz_b_a, C_a, mz_b_s, Vz_b_s, C_s)
        mz_f = C_f @ mw
        mz_a = C_a @ mw
        mz_s = C_s @ mw

        w_err = compute_w_err(mw)
        z_f_err = compute_z_f_err(mz_f)
        z_a_err = compute_z_a_err(mz_a)
        z_s_err = compute_z_s_err(mz_s)

        if w_err < tol and z_f_err < tol and z_a_err < tol and z_s_err < tol:
            print(f"Optimization problem solved after {itr} iterations!", flush=True)
            break

        # priors
        mw_f[:, 0], Vw_f[torch.arange(K), torch.arange(K)] = compute_box_prior(mw[:, 0], -wb, wb, gamma_wb)
        mz_b_f[:, 0, 0], Vz_b_f[:, 0, 0] = compute_box_prior(mz_f[:, 0, 0], theta, theta, gamma_theta)
        mz_b_f[:, 1, 0], Vz_b_f[:, 1, 1] = compute_box_prior(mz_f[:, 1, 0], a, None, gamma_a)
        mz_b_a[:, 0, 0], Vz_b_a[:, 0, 0] = compute_box_prior(mz_a[:, 0, 0], a, None, gamma_a)
        mz_b_s[:, 0, 0], Vz_b_s[:, 0, 0] = compute_box_prior(mz_s[:, 0, 0], None, b, gamma_b)

    print("Optimization done with errors: ", w_err, z_f_err, z_a_err, z_s_err, flush=True)
    return_dict[l] = mw

    # return mw.squeeze()
