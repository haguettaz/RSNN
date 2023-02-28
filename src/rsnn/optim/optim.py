from typing import Callable

import numpy as np

from .gmp import fgmp_obs_blck


def solve(
        mw: np.ndarray,
        C_firing: np.ndarray,
        C_active: np.ndarray,
        C_silent: np.ndarray,
        NUV_weights: Callable,
        NUV_firing: Callable,
        NUV_active: Callable,
        NUV_silent: Callable,
        ERR_weights: Callable,
        ERR_firing: Callable,
        ERR_active: Callable,
        ERR_silent: Callable,
        max_iter: int = 1000, 
        err_tol: float = 1e-4
        ):    

    mz_firing = C_firing @ mw
    mz_active = C_active @ mw 
    mz_silent = C_silent @ mw 

    for itr in range(max_iter):
        if np.isnan(mw).any():
            # print("Optimization failed: NaN in weights")
            # print("\tIteration:", itr)
            return mw, -1

        # compute the priors
        mw_forward, Vw_forward = NUV_weights(mw)
        mz_firing_backward, Vz_firing_backward = NUV_firing(mz_firing)
        mz_active_backward, Vz_active_backward = NUV_active(mz_active)
        mz_silent_backward, Vz_silent_backward = NUV_silent(mz_silent)

        # compute the posteriors
        mw, _ = compute_weight_posterior(mw_forward, Vw_forward, mz_firing_backward, Vz_firing_backward, mz_active_backward, Vz_active_backward, mz_silent_backward, Vz_silent_backward, C_firing, C_active, C_silent)
        mz_firing = C_firing @ mw 
        mz_active = C_active @ mw 
        mz_silent = C_silent @ mw

        # stopping criterion
        if ERR_weights(mw) < err_tol and ERR_firing(mz_firing) < err_tol and ERR_active(mz_active) < err_tol and ERR_silent(mz_silent) < err_tol:
            # print("Optimization done!")
            # print(" Iterations:", itr)
            # print(" Error weights:", ERR_weights(mw))
            # print(" Error firing:", ERR_firing(mz_firing))
            # print(" Error active:", ERR_active(mz_active))
            # print(" Error silent:", ERR_silent(mz_silent))
            return mw, 1

    # print("Optimization failed: max iterations reached")
    return mw, 0

def compute_weight_posterior(mw_forward, Vw_forward, mz_firing_backward, Vz_firing_backward, mz_active_backward, Vz_active_backward, mz_silent_backward, Vz_silent_backward, C_firing, C_active, C_silent):
    """
    Compute the weight posterior means and variances my forward Gaussian message passing.

    Args:
        mw_forward (torch.FloatTensor): weight prior mean with one dimension of length K.
        Vw_forward (torch.FloatTensor): weight prior variances with one dimension of length K.
        mz_firing_backward (torch.FloatTensor): firing observations (potential) prior means with one dimension of length N_firing.
        Vz_firing_backward (torch.FloatTensor): firing observations (potential) prior means with one dimension of length N_firing.
        mz_active_backward (torch.FloatTensor): active observations (potential derivative) prior means with one dimension of length N_active.
        Vz_active_backward (torch.FloatTensor): active observations (potential derivative) prior variances with one dimension of length N_active.
        mz_silent_backward (torch.FloatTensor): silent observations (potential) prior means with one dimension of length N_silent.
        Vz_silent_backward (torch.FloatTensor): silent observations (potential) prior variances with one dimension of length N_silent.
        C_firing (torch.FloatTensor): firing observation tensor with two dimensions of length N_firing and K.
        C_active (torch.FloatTensor): firing observation tensor with two dimensions of length N_active and K.
        C_silent (torch.FloatTensor): firing observation tensor with two dimensions of length N_silent and K.

    Returns:
        (torch.FloatTensor): weight posterior means with one dimension of length K.
        (torch.FloatTensor): weight posterior variances with one dimension of length K.
    """
    Vw_forward = np.diag(Vw_forward)

    N_firing, N_active, N_silent = C_firing.shape[0], C_active.shape[0], C_silent.shape[0]

    # Equality Constraints at Firing Times
    for n in range(N_firing):
        g_inv = (Vz_firing_backward[n] + np.inner(C_firing[n], Vw_forward @ C_firing[n]))
        if g_inv == 0:
            print("g_inv is zero", "firing", n)
        mw_forward, Vw_forward = fgmp_obs_blck(mw_forward, Vw_forward, mz_firing_backward[n], Vz_firing_backward[n], C_firing[n])

    # Unequality Constraints at Active Times
    for n in range(N_active):
        g_inv = (Vz_active_backward[n] + np.inner(C_active[n], Vw_forward @ C_active[n]))
        if g_inv == 0:
            print("g_inv is zero", "active", n)
        mw_forward, Vw_forward = fgmp_obs_blck(mw_forward, Vw_forward, mz_active_backward[n], Vz_active_backward[n], C_active[n])

    # Unequality Constraints at Silent Times
    for n in range(N_silent):
        g_inv = (Vz_silent_backward[n] + np.inner(C_silent[n], Vw_forward @ C_silent[n]))
        if g_inv == 0:
            print("g_inv is zero", "silent", n, np.max(np.abs(Vw_forward)), np.max(np.abs(C_silent[n])), np.max(np.abs(Vw_forward @ C_silent[n])), Vz_silent_backward[n], np.inner(C_silent[n], Vw_forward @ C_silent[n]))
        mw_forward, Vw_forward = fgmp_obs_blck(mw_forward, Vw_forward, mz_silent_backward[n], Vz_silent_backward[n], C_silent[n])

    return mw_forward, np.diag(Vw_forward)