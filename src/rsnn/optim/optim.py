from typing import Tuple

import numpy as np
from tqdm.autonotebook import trange

from .gmp import observation_block_forward
from .nuv import binary_prior, box_prior
from .utils import bin_error, box_error


def compute_bounded_weights(
    yt: np.ndarray,
    zt_min: np.ndarray,
    zt_max: np.ndarray,
    w_min: np.ndarray,
    w_max: np.ndarray,
    max_iter: int = 1000,
    err_tol: float = 1e-3,
):
    """
    Compute the bounded weights to satisfy the potential template, using IRLS-based method.

    Args:
        yt (np.ndarray): the linear transformations from weights to observations.
        zt_min (np.ndarray): the smallest admissible observation values. 
        zt_max (np.ndarray): the largest admissible observation values.
        w_min (np.ndarray): the smallest admissible weight values.
        w_max (np.ndarray): the largest admissible weight values.
        max_iter (int, optional): the maximum number of iterations. Defaults to 1000.
        err_tol (float, optional): the error tolerance to stop the algorithm. Defaults to 1e-3.

    Returns:
        (np.ndarray): the bounded weights.
        (dict): the optimization summary.
    """
    # K = yt.shape[1]

    mw = np.random.uniform(w_min, w_max)
    mz = yt @ mw

    for i in trange(max_iter, desc="neuron optimization"):
        # compute nuv priors based on posterior means only (no variances)
        mfw, Vfw = box_prior(mw, w_min, w_max, 1)
        mbzt, Vbzt = box_prior(mz, zt_min, zt_max, 1)

        # compute weights posterior means by forward message passing
        mw = np.copy(mfw)
        Vw = np.diag(Vfw)
        for ytn, mbztn, Vbztn in zip(yt, mbzt, Vbzt):
            mw, Vw = observation_block_forward(ytn, mw, Vw, mbztn, Vbztn)

        # compute potential posterior means
        mz = yt @ mw

        # stopping criterion
        if box_error(mw, w_min, w_max) < err_tol and box_error(mz, zt_min, zt_max) < err_tol:
            summary = {
                "status": "solved",
                "num_iter": i,
                "weight_error": box_error(mw, w_min, w_max),
                "potential_error": box_error(mz, zt_min, zt_max),
            }
            return mw, summary

    summary = {
        "status": "not solved",
        "num_iter": i,
        "weight_error": box_error(mw, w_min, w_max),
        "potential_error": box_error(mz, zt_min, zt_max),
    }
    return mw, summary


def compute_bounded_discrete_weights(
    yt: np.ndarray,
    zt_min: np.ndarray,
    zt_max: np.ndarray,
    w_min: Tuple[float, float],
    w_max: np.ndarray,
    w_lvl: int,
    max_iter: int = 5000,
    # var_tol: float = 1e-3,
    err_tol: float = 1e-3,
):
    """
    Compute the discrete bounded weights to satisfy the potential template, using IRLS-based method.

    Args:
        yt (np.ndarray): the linear transformations from weights to observations.
        zt_min (np.ndarray): the smallest admissible observation values. 
        zt_max (np.ndarray): the largest admissible observation values.
        w_min (np.ndarray): the smallest admissible weight values.
        w_max (np.ndarray): the largest admissible weight values.
        w_lvl (int): the number of discrete levels for all weights.
        max_iter (int, optional): the maximum number of iterations. Defaults to 1000.
        err_tol (float, optional): the error tolerance to stop the algorithm. Defaults to 1e-3.

    Returns:
        (np.ndarray): the bounded weights.
        (dict): the optimization summary.
    """
    K = yt.shape[1]

    wm_min = np.repeat(w_min[:,None], w_lvl - 1, axis=-1)
    wm_max = np.repeat(w_max[:,None], w_lvl - 1, axis=-1)

    # weights are parametrized as zt_min mixture of wlvl-1 binary components, i.e., wk = wk,1 + wk,2 + ... + wk,wlvl-1
    mwm = np.random.uniform(wm_min, wm_max)
    Vwm = np.square(wm_max - wm_min)
    mw, Vw = np.sum(mwm, axis=-1), np.sum(Vwm, axis=-1)
    mz = yt @ mw

    for i in trange(max_iter, desc="neuron optimization"):
        if np.isnan(mw).any():
            return mw, "nan"

        # compute nuv priors based on posterior means and variances
        mfwm, Vfwm = binary_prior(mwm, Vwm, wm_min, wm_max)
        mfw, Vfw = np.sum(mfwm, axis=-1), np.sum(Vfwm, axis=-1)
        mbzt, Vbzt = box_prior(mz, zt_min, zt_max, 1e-2)

        # compute weights posterior means and variances by forward message passing
        mw = np.copy(mfw)
        Vw = np.diag(Vfw)
        for ytn, mbztn, Vbztn in zip(yt, mbzt, Vbzt):
            mw, Vw = observation_block_forward(ytn, mw, Vw, mbztn, Vbztn)

        # weights with prior variance zero are problematic BUT should not be updated anyway
        # selection = np.argwhere(Vfw > 1e-3).flatten()

        # compute the duals for propagation through additive boxes
        Wfw = np.diag(1 / Vfw)  # RuntimeWarning: divide by zero encountered in divide
        Wtw = np.diag(Wfw - Wfw @ Vw @ Wfw)  # RuntimeWarning: invalid value encountered in matmul
        xitw = Wfw @ (mfw - mw)  # RuntimeWarning: invalid value encountered in matmul

        # change back to posterior means and variances of weights discrete components
        # weight with prior variance zero are not updated
        mwm = mfwm - Vfwm * xitw[:, None]
        Vwm = Vfwm - Vfwm * Wtw[:, None] * Vfwm
        # print("means", mwm[0], "should be in {", wlim[0]/(wlvl-1), ",", wlim[1]/(wlvl-1), "}")
        # print("vars", Vwm[0])

        mw, Vw = np.sum(mwm, axis=-1), np.sum(Vwm, axis=-1)
        mz = yt @ mw

        if box_error(mz, zt_min, zt_max) < err_tol:
            if bin_error(mwm, wm_min, wm_max) < err_tol:
                summary = {
                    "status": "solved",
                    "num_iter": i,
                    "weight_error": bin_error(mwm, wm_min, wm_max),
                    "potential_error": box_error(mz, zt_min, zt_max),
                }
                return mw, summary
        elif bin_error(mwm, wm_min, wm_max) < err_tol:  # premature binarization
            summary = {
                "status": "not solved",
                "num_iter": i,
                "weight_error": bin_error(mwm, wm_min, wm_max),
                "potential_error": box_error(mz, zt_min, zt_max),
            }
            return mw, summary

    summary = {
        "status": "not solved",
        "num_iter": i,
        "weight_error": bin_error(mwm, wm_min, wm_max),
        "potential_error": box_error(mz, zt_min, zt_max),
    }
    return mw, summary
