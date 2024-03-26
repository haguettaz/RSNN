import numpy as np


def obs_block(mxf, Vxf, myb, Vyb, C):
    """
    Gaussian message passing through a (scalar) observation block.

    Args:
        mxf (np.ndarray): the forward input mean vector with shape (K).
        Vxf (np.ndarray): the forward input covariance matrix with shape (K, K).
        myb (np.ndarray): the backward observation mean with shape (1).
        Vyb (np.ndarray): the backward observation variance with shape (1).
        C (np.ndarray): the observation matrix with shape (K).

    Returns:
        (np.ndarray): the forward output mean vector with shape (K).
        (np.ndarray): the forward output covariance matrix with shape (K, K).
    """
    CVxf = C @ Vxf
    g_inv = Vyb + np.inner(CVxf,C)
    return mxf + (myb - np.inner(C,mxf)) / g_inv * CVxf, Vxf - np.outer(CVxf, CVxf / g_inv)


def solve_lp(A, b, xb, max_iter=1000):
    """
    Solve the linear program subject.

    Args:
        A (np.ndarray): the constraint matrix with shape (N, K).
        b (np.ndarray): the constraint vector with shape (N).
        xb (np.ndarray): the variable bounds with shape (K).
        max_iter (int, optional): the maximum number of iteration. Defaults to 1000.

    Returns:
        (np.ndarray): a solution to the linear program with shape (K).
        (str): the status of the optimization.
    """
    K, N = A.shape

    # init with random posterior means
    mxf = np.random.uniform(-xb, xb, K)  # (K)
    my = A @ mxf  # (N)

    for _ in range(max_iter):
        # nuv updates for the bounds
        Vxlf = np.abs(mxf + xb)  # (K)
        Vxrl = np.abs(mxf - xb)  # (K)
        mxf = xb * (Vxlf - Vxrl) / (Vxlf + Vxrl)  # (K)
        Vxf = Vxlf * Vxrl / (Vxlf + Vxrl)  # (K)

        # nuv updates for the constraints
        myb = b - np.abs(my - b)  # (N)
        Vyb = np.abs(my - b)  # (N)

        # posterior mean and variance of X0
        Vxf = np.diag(Vxf)  # (K, K)
        for n in range(N):
            mxf, Vxf = obs_block(mxf, Vxf, myb[n], Vyb[n], A[n])  # (K) and (K, K)

        # check convergence
        if np.allclose(mxf, prev_mxf):
            status = "converged"
            return mxf, status
        
        # posterior means of Ys (we don't need the variances)
        my = A @ mxf  # (N)
        
        prev_mxf = mxf

    status = "max_iter"
    return mxf, status


def solve_lp_l1(A, b, xb, l1_reg=1.0, max_iter=1000):
    """
    Solve the linear program subject to l1 regularization.

    Args:
        A (np.ndarray): the constraint matrix with shape (N, K).
        b (np.ndarray): the constraint vector with shape (N).
        xb (np.ndarray): the variable bounds with shape (K).
        l1_reg (float): the l1 regularization parameter.
        max_iter (int, optional): the maximum number of iteration. Defaults to 1000.

    Returns:
        (np.ndarray): a solution to the linear program with shape (K).
        (str): the status of the optimization.
    """
    K, N = A.shape

    # init with random posterior means
    mxf = np.random.uniform(-xb, xb, K)  # (K)
    my = A @ mxf  # (N)

    for _ in range(max_iter):
        # nuv updates for the bounds with L1 regularization
        Vxlf = np.abs(mxf + xb)  # (K)
        Vxrl = np.abs(mxf - xb)  # (K)
        mxbf = xb * (Vxlf - Vxrl) / (Vxlf + Vxrl)  # (K)
        Vxbf = Vxlf * Vxrl / (Vxlf + Vxrl)  # (K)
        Vxl1f = np.abs(mxf) / l1_reg
        mxf = mxbf * Vxl1f / (Vxbf + Vxl1f)  # (K)
        Vxf = Vxbf * Vxl1f / (Vxbf + Vxl1f)  # (K)

        # nuv updates for the constraints
        myb = b - np.abs(my - b)  # (N)
        Vyb = np.abs(my - b)  # (N)

        # posterior mean and variance of X0
        Vxf = np.diag(Vxf)  # (K, K)
        for n in range(N):
            mxf, Vxf = obs_block(mxf, Vxf, myb[n], Vyb[n], A[n])  # (K) and (K, K)

        # check convergence
        if np.allclose(mxf, prev_mxf):
            status = "converged"
            return mxf, status
        
        # posterior means of Ys (we don't need the variances)
        my = A @ mxf  # (N)
        
        prev_mxf = mxf

    status = "max_iter"
    return mxf, status

def solve_lp_l2(A, b, xb, l2_reg=1.0, max_iter=1000):
    """
    Solve the linear program subject to l2 regularization.

    Args:
        A (np.ndarray): the constraint matrix with shape (N, K).
        b (np.ndarray): the constraint vector with shape (N).
        xb (np.ndarray): the variable bounds with shape (K).
        l2_reg (float): the l2 regularization parameter.
        max_iter (int, optional): the maximum number of iteration. Defaults to 1000.

    Returns:
        (np.ndarray): a solution to the linear program with shape (K).
        (str): the status of the optimization.
    """
    K, N = A.shape

    # init with random posterior means
    mxf = np.random.uniform(-xb, xb, K)  # (K)
    my = A @ mxf  # (N)

    Vxl2f = 1 / l2_reg

    for _ in range(max_iter):
        # nuv updates for the bounds with L2 regularization
        Vxlf = np.abs(mxf + xb)  # (K)
        Vxrl = np.abs(mxf - xb)  # (K)
        mxbf = xb * (Vxlf - Vxrl) / (Vxlf + Vxrl)  # (K)
        Vxbf = Vxlf * Vxrl / (Vxlf + Vxrl)  # (K)
        mxf = mxbf * Vxl2f / (Vxbf + Vxl2f)  # (K)
        Vxf = Vxbf * Vxl2f / (Vxbf + Vxl2f)  # (K)

        # nuv updates for the constraints
        myb = b - np.abs(my - b)  # (N)
        Vyb = np.abs(my - b)  # (N)

        # posterior mean and variance of X0
        Vxf = np.diag(Vxf)  # (K, K)
        for n in range(N):
            mxf, Vxf = obs_block(mxf, Vxf, myb[n], Vyb[n], A[n])  # (K) and (K, K)

        # check convergence
        if np.allclose(mxf, prev_mxf):
            status = "converged"
            return mxf, status
        
        # posterior means of Ys (we don't need the variances)
        my = A @ mxf  # (N)
        
        prev_mxf = mxf

    status = "max_iter"
    return mxf, status


def solve_lp_q(A, b, xb, xlvl, max_iter=1000):
    """
    Solve the linear program subject to quantization constraints.

    Args:
        A (np.ndarray): the constraint matrix with shape (N, K).
        b (np.ndarray): the constraint vector with shape (N).
        xb (np.ndarray): the variable bounds with shape (K).
        xlvl (int): the number of quantization levels.
        max_iter (int, optional): the maximum number of iteration. Defaults to 1000.

    Returns:
        (np.ndarray): a solution to the linear program with shape (K).
        (str): the status of the optimization.
    """
    # A x <= b subject to x in {-xb, ..., 0, ..., xb}
    # if xb is None: no bounds on the weights
    K, N = A.shape
    M = xlvl - 1

    ub = xb / (xlvl - 1)

    mu = np.random.uniform(-ub, ub, (M, K))  # (M, K)
    Vu = np.ones_like(mu)  # (K)
    my = A @ np.sum(mu, axis=0)  # (N)

    for _ in range(max_iter):

        # nuv updates for the m-levels
        Vufm = Vu + np.square(mu + ub)
        Vufp = Vu + np.square(mu - ub)
        Vuf = Vufm * Vufp / (Vufm + Vufp)  # (M, K)
        muf = (Vufm * ub - Vufp * ub) / (Vufm + Vufp)  # (M, K)

        # nuv updates for the constraints
        myb = b - np.abs(my - b)  # (N)
        Vyb = np.abs(my - b)  # (N)

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
        mu = muf - muf * xix0t  # (M, K)
        Vu = Vuf - np.square(Vuf) * Wx0t  # (M, K)

        # check convergence
        if np.allclose(mxf, prev_mxf):
            status = "converged"
            return mxf, status
        
        # posterior means of Ys (we don't need the variances)
        my = A @ mxf  # (N)
        
        prev_mxf = mxf

    status = "max_iter"
    return mxf, status
