from typing import Callable, List, Optional, Tuple

import numpy as np
from tqdm.autonotebook import tqdm

from .gmp import fgmp_obs_blck
from .nuv import box_prior, m_ary_prior


def solve(
        C: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        weights_lim: Tuple[float, float],
        weights_lvl: Optional[float]=None,
        max_iter: int = 1000, 
        err_tol: float = 1e-4,
        rng: np.random.Generator = None
        # C_firing: np.ndarray,
        # C_active: np.ndarray,
        # C_silent: np.ndarray,
        # NUV_weights: Callable,
        # NUV_firing: Callable,
        # NUV_active: Callable,
        # NUV_silent: Callable,
        # ERR_weights: Callable,
        # ERR_firing: Callable,
        # ERR_active: Callable,
        # ERR_silent: Callable,
        # max_iter: int = 1000, 
        # err_tol: float = 1e-4
        ):    

    rng = rng or np.random.default_rng()

    def box_error(x: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> float:
        err = np.zeros(1)

        mask_finite_xmin = np.isfinite(xmin)
        mask_finite_xmax = np.isfinite(xmax)

        # left half-space
        mask = (~mask_finite_xmin) & mask_finite_xmax
        if np.any(mask):
            err = np.maximum(err, np.max(np.abs(xmax[mask] - x[mask]) - (xmax[mask] - x[mask])))

        # right half-space
        mask = (~mask_finite_xmax) & mask_finite_xmin
        if np.any(mask):
            err = np.maximum(err, np.max(np.abs(x[mask] - xmin[mask]) - (x[mask] - xmin[mask])))

        # box
        mask = mask_finite_xmin & mask_finite_xmax
        if np.any(mask):
            err = np.maximum(err, np.max(np.abs(x[mask] - xmin[mask]) + np.abs(x[mask] - xmax[mask]) - (xmax[mask] - xmin[mask])))

        return err
    
    NUV_template = lambda z_: box_prior(z_, a, b, 1)
    err_template = lambda z_: box_error(z_, a, b)

    weights_min, weights_max = weights_lim
    if weights_lvl is None:
        NUV_weights = lambda w_: box_prior(w_, weights_min, weights_max, 1)
        err_weights = lambda w_: box_error(w_, weights_min, weights_max)
    else:
        NUV_weights = lambda w_: m_ary_prior(w_, weights_min, weights_max, weights_lvl)
        raise NotImplementedError("error for m-ary prior not implemented yet")

    K = C.shape[1]
    mw = rng.uniform(weights_min, weights_max, size=(K,))
    mz = C @ mw

    # with tqdm(total=max_iter, leave=False) as pbar:
    for _ in range(max_iter):
        if np.isnan(mw).any():
            raise ValueError("Optimization failed with NaN values")

        # compute the priors
        mw_forward, Vw_forward = NUV_weights(mw)
        mz_backward, Vz_backward = NUV_template(mz)

        # compute the posteriors
        mw, _ = compute_weight_posterior(C, mw_forward, Vw_forward, mz_backward, Vz_backward)
        mz = C @ mw

        # stopping criterion
        if err_weights(mw) < err_tol and err_template(mz) < err_tol:
            # print("Optimization done!")
            # print(" Iterations:", itr)
            # print(" Error weights:", err_weights(mw))
            # print(" Error template:", err_template(mz))
            # pbar.update(max_iter - itr)
            return mw
        
        # pbar.update(1)

        # print("Optimization failed: max iterations reached")
    raise ValueError("Optimization failed with maximum number of iterations reached")

def compute_weight_posterior(C, mw_forward, Vw_forward, mz_backward, Vz_backward):
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

    for C_n, mz_backward_n, Vz_backward_n in zip(C, mz_backward, Vz_backward):
        mw_forward, Vw_forward = fgmp_obs_blck(C_n, mw_forward, Vw_forward, mz_backward_n, Vz_backward_n)

    return mw_forward, np.diag(Vw_forward)