import numpy as np
from tqdm.autonotebook import trange

from .nuv import binary_nuv, box_nuv, left_half_space_nuv
from .utils import all_close_to_one_of, obs_block, round_to_nearest


def solve_lp(A, b, xb, gamma_y=1.0, gamma_xb=1.0, max_iter=1000, rtol=1e-5, atol=1e-8):
    """
    Solve the linear program subject.

    Args:
        A (np.ndarray): the constraint matrix with shape (N, K).
        b (np.ndarray): the constraint vector with shape (N).
        xb (np.ndarray): the variable bounds with shape (K).
        gamma_y (float, optional): the constraints strength parameter. Defaults to 1.0.
        gamma_xb (float, optional): the bounds strength parameter. Defaults to 1.0.
        max_iter (int, optional): the maximum number of iteration. Defaults to 1000.
        rtol (float, optional): the convergence criterion relative tolerance. Defaults to 1e-5.
        atol (float, optional): the convergence criterion absolute tolerance. Defaults to 1e-8.

    Returns:
        (dict): the optimization summary.
    """
    N, K = A.shape

    # init with random posterior means
    mxf = np.random.uniform(-xb, xb, K)  # (K)
    prev_mxf = mxf

    for i in trange(max_iter):
        # nuv updates for the constraints
        myb, Vyb = left_half_space_nuv(A @ mxf, b, gamma_y)  # (N) and (N)

        # nuv updates for the bounds
        mxf, Vxf = box_nuv(mxf, xb, gamma_xb)  # (K) and (K)

        # posterior mean and variance of X0
        Vxf = np.diag(Vxf)  # (K, K)
        for n in range(N):
            mxf, Vxf = obs_block(mxf, Vxf, myb[n], Vyb[n], A[n])  # (K) and (K, K)

        # check convergence
        if np.allclose(mxf, prev_mxf, rtol=rtol, atol=atol):
            summary = {
                "num_iter": i,
                "status": "converged",
                "constraints": np.all(A @ mxf <= b + atol),
                "bounds": np.all(np.abs(mxf) <= xb + atol),
                "x": mxf,
            }
            return summary

        prev_mxf = mxf

    summary = {
        "num_iter": max_iter,
        "status": "max_iter",
        "constraints": np.all(A @ mxf <= b + atol),
        "bounds": np.all(np.abs(mxf) <= xb + atol),
        "x": mxf,
    }
    return summary


def solve_lp_l1(
    A,
    b,
    xb,
    gamma_y=1.0,
    gamma_xb=1.0,
    gamma_l1=1.0,
    max_iter=5000,
    rtol=1e-5,
    atol=1e-8,
):
    """
    Solve the linear program subject to l1 regularization.

    Args:
        A (np.ndarray): the constraint matrix with shape (N, K).
        b (np.ndarray): the constraint vector with shape (N).
        xb (np.ndarray): the variable bounds with shape (K).
        gamma_y (float, optional): the constraints strength parameter. Defaults to 1.0.
        gamma_xb (float, optional): the bounds strength parameter. Defaults to 1.0.
        gamma_l1 (float, optional): the l1 strength parameter. Defaults to 1.0.
        max_iter (int, optional): the maximum number of iteration. Defaults to 1000.
        rtol (float, optional): the convergence criterion relative tolerance. Defaults to 1e-5.
        atol (float, optional): the convergence criterion absolute tolerance. Defaults to 1e-8.
        
    Returns:
        (dict): the optimization summary.
    """
    N, K = A.shape

    # init with random posterior means
    mxf = np.random.uniform(-xb, xb, K)  # (K)
    prev_mxf = mxf

    for i in trange(max_iter):
        # nuv updates for the constraints
        myb, Vyb = left_half_space_nuv(A @ mxf, b, gamma_y)  # (N) and (N)

        # nuv updates for the bounds with L1 regularization
        mxbf, Vxbf = box_nuv(mxf, xb, gamma_xb)  # (K) and (K)
        Vxl1f = np.abs(mxf) / gamma_l1
        mxf = mxbf * Vxl1f / (Vxbf + Vxl1f)  # (K)
        Vxf = Vxbf * Vxl1f / (Vxbf + Vxl1f)  # (K)

        # posterior mean and variance of X0
        Vxf = np.diag(Vxf)  # (K, K)
        for n in range(N):
            mxf, Vxf = obs_block(mxf, Vxf, myb[n], Vyb[n], A[n])  # (K) and (K, K)

        # check convergence
        if np.allclose(mxf, prev_mxf, rtol=rtol, atol=atol):
            summary = {
                "num_iter": i,
                "status": "converged",
                "constraints": np.all(A @ mxf <= b + atol),
                "bounds": np.all(np.abs(mxf) <= xb + atol),
                "l1": np.mean(np.abs(mxf)),
                "x": mxf,
            }
            return summary

        prev_mxf = mxf

    summary = {
        "num_iter": max_iter,
        "status": "max_iter",
        "constraints": np.all(A @ mxf <= b + atol),
        "bounds": np.all(np.abs(mxf) <= xb + atol),
        "l1": np.mean(np.abs(mxf)),
        "x": mxf,
    }
    return summary


def solve_lp_l2(
    A,
    b,
    xb,
    gamma_y=1.0,
    gamma_xb=1.0,
    gamma_l2=1.0,
    max_iter=5000,
    rtol=1e-5,
    atol=1e-8,
):
    """
    Solve the linear program subject to l2 regularization.

    Args:
        A (np.ndarray): the constraint matrix with shape (N, K).
        b (np.ndarray): the constraint vector with shape (N).
        xb (np.ndarray): the variable bounds with shape (K).
        gamma_y (float, optional): the constraints strength parameter. Defaults to 1.0.
        gamma_xb (float, optional): the bounds strength parameter. Defaults to 1.0.
        gamma_l2 (float, optional): the l2 strength parameter. Defaults to 1.0.
        max_iter (int, optional): the maximum number of iteration. Defaults to 1000.
        rtol (float, optional): the convergence criterion relative tolerance. Defaults to 1e-5.
        atol (float, optional): the convergence criterion absolute tolerance. Defaults to 1e-8.

    Returns:
        (dict): the optimization summary.
    """
    N, K = A.shape

    # init with random posterior means
    mxf = np.random.uniform(-xb, xb, K)  # (K)
    prev_mxf = mxf

    Vxl2f = 1 / gamma_l2

    for i in trange(max_iter):
        # nuv updates for the constraints
        myb, Vyb = left_half_space_nuv(A @ mxf, b, gamma_y)  # (N) and (N)

        # nuv updates for the bounds with L2 regularization
        mxbf, Vxbf = box_nuv(mxf, xb, gamma_xb)  # (K) and (K)
        mxf = mxbf * Vxl2f / (Vxbf + Vxl2f)  # (K)
        Vxf = Vxbf * Vxl2f / (Vxbf + Vxl2f)  # (K)

        # posterior mean and variance of X0
        Vxf = np.diag(Vxf)  # (K, K)
        for n in range(N):
            mxf, Vxf = obs_block(mxf, Vxf, myb[n], Vyb[n], A[n])  # (K) and (K, K)

        # check convergence
        if np.allclose(mxf, prev_mxf, rtol=rtol, atol=atol):
            summary = {
                "num_iter": i,
                "status": "converged",
                "constraints": np.all(A @ mxf <= b + atol),
                "bounds": np.all(np.abs(mxf) <= xb + atol),
                "l2": np.mean(np.square(mxf)),
                "x": mxf,
            }
            return summary

        prev_mxf = mxf

    summary = {
        "num_iter": max_iter,
        "status": "max_iter",
        "constraints": np.all(A @ mxf <= b + atol),
        "bounds": np.all(np.abs(mxf) <= xb + atol),
        "l2": np.mean(np.square(mxf)),
        "x": mxf,
    }
    return summary


def solve_lp_q(
    A, b, xb, xlvl, gamma_y=1.0, max_iter=5000, rtol=1e-5, atol=1e-8
):
    """
    Solve the linear program subject to quantization constraints.

    Args:
        A (np.ndarray): the constraint matrix with shape (N, K).
        b (np.ndarray): the constraint vector with shape (N).
        xb (np.ndarray): the variable bounds with shape (K).
        xlvl (int): the number of quantization levels.
        gamma_y (float, optional): the constraints strength parameter. Defaults to 1.0.
        max_iter (int, optional): the maximum number of iteration. Defaults to 5000.
        rtol (float, optional): the convergence criterion relative tolerance. Defaults to 1e-5.
        atol (float, optional): the convergence criterion absolute tolerance. Defaults to 1e-8.

    Returns:
        (dict): the optimization summary.
    """
    N, K = A.shape
    M = xlvl - 1

    ub = xb / (xlvl - 1)

    mu = np.random.uniform(-ub, ub, (M, K))  # (M, K)
    Vu = np.ones_like(mu)  # (K)
    mxf = np.sum(mu, axis=0)  # (K)
    prev_mu = mu

    for i in trange(max_iter):

        # nuv updates for the m-levels
        muf, Vuf = binary_nuv(mu, Vu, ub)  # (M, K) and (M, K)

        # nuv updates for the constraints
        myb, Vyb = left_half_space_nuv(A @ mxf, b, gamma_y)  # (N) and (N)

        # forward messages at X0
        mxf = np.sum(muf, axis=0)  # (K)
        Vxf = np.sum(Vuf, axis=0)  # (K)
        Wxf = 1 / Vxf  # (K)
        xixf = Wxf * mxf  # (K)

        # posterior mean and variance of X0
        Vxf = np.diag(Vxf)  # (K, K)
        for n in range(N):
            mxf, Vxf = obs_block(mxf, Vxf, myb[n], Vyb[n], A[n])  # (K) and (K, K)

        # dual precision and weighted mean of X0 (we don't need the off-diagonal terms of the dual precision)
        Wx0t = Wxf - np.square(Wxf) * np.diag(Vxf)  # (K)
        xix0t = xixf - Wxf * mxf  # (K)

        # posterior means and variances of Us
        mu = muf - Vuf * xix0t  # (M, K)
        Vu = Vuf - np.square(Vuf) * Wx0t  # (M, K)

        # check convergence
        if np.allclose(mu, prev_mu, rtol=rtol, atol=atol):
            x_r = round_to_nearest(mxf, np.linspace(-xb, xb, xlvl))
            summary = {
                "num_iter": i,
                "status": "converged",
                "x": mxf,
                "constraints": np.all(A @ mxf <= b + atol),
                "quantization": all_close_to_one_of(
                    mxf, np.linspace(-xb, xb, xlvl), atol=1e-2
                ),
                "x_rounded": x_r,
                "constraints_after_rounding": np.all(A @ x_r <= b),
            }
            return summary

        prev_mu = mu

    x_r = round_to_nearest(mxf, np.linspace(-xb, xb, xlvl))
    summary = {
        "num_iter": max_iter,
        "status": "max_iter",
        "x": mxf,
        "constraints": np.all(A @ mxf <= b + atol),
        "quantization": all_close_to_one_of(
            mxf, np.linspace(-xb, xb, xlvl), atol=1e-2
        ),
        "x_rounded": x_r,
        "constraints_after_rounding": np.all(A @ x_r <= b),
    }
    return summary
