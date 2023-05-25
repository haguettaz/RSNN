from typing import Any, Dict, Tuple, Union

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
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute the bounded weights to satisfy the potential template, using IRLS-based method.

    Args:
        yt (np.ndarray[float]): the linear transformations from weights to observations.
        zt_min (np.ndarray[float]): the smallest admissible observation values.
        zt_max (np.ndarray[float]): the largest admissible observation values.
        w_min (np.ndarray[float]): the smallest admissible weight values.
        w_max (np.ndarray[float]): the largest admissible weight values.
        max_iter (int, optional): the maximum number of iterations. Defaults to 1000.
        err_tol (float, optional): the maximum error to stop the algorithm. Defaults to 1e-3.

    Returns:
        Tuple[np.ndarray[float], Dict[str, Any]]: the bounded weights and the optimization summary.
    """

    firing_selection = np.logical_and(np.isfinite(zt_min), np.isfinite(zt_max))
    min_slope_selection = np.logical_and(np.isfinite(zt_min), ~np.isfinite(zt_max))
    max_level_selection = np.logical_and(~np.isfinite(zt_min), np.isfinite(zt_max))

    mw = np.random.uniform(w_min, w_max)
    mz = yt @ mw

    for i in trange(max_iter, desc="Neuron optimization"):
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
                "success": True,
                "num_iter": i,
                "weight_error": box_error(mw, w_min, w_max),
                "potential_error": box_error(mz, zt_min, zt_max),
                "firing_error": box_error(mz[firing_selection], zt_min[firing_selection], zt_max[firing_selection]),
                "min_slope_error": box_error(
                    mz[min_slope_selection], zt_min[min_slope_selection], zt_max[min_slope_selection]
                ),
                "max_level_error": box_error(
                    mz[max_level_selection], zt_min[max_level_selection], zt_max[max_level_selection]
                ),
            }
            return mw, summary

    summary = {
        "success": False,
        "num_iter": max_iter,
        "weight_error": box_error(mw, w_min, w_max),
        "potential_error": box_error(mz, zt_min, zt_max),
        "firing_error": box_error(mz[firing_selection], zt_min[firing_selection], zt_max[firing_selection]),
        "min_slope_error": box_error(
            mz[min_slope_selection], zt_min[min_slope_selection], zt_max[min_slope_selection]
        ),
        "max_level_error": box_error(
            mz[max_level_selection], zt_min[max_level_selection], zt_max[max_level_selection]
        ),
    }
    print("summary", summary)
    return mw, summary


def compute_bounded_discrete_weights(
    yt: np.ndarray,
    zt_min: np.ndarray,
    zt_max: np.ndarray,
    w_min: np.ndarray,
    w_max: np.ndarray,
    w_lvl: int,
    max_iter: int = 5000,
    err_tol: float = 1e-3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute the discrete bounded weights to satisfy the potential template, using IRLS-based method.

    Args:
        yt (np.ndarray[float]): the linear transformations from weights to observations.
        zt_min (np.ndarray[float]): the smallest admissible observation values.
        zt_max (np.ndarray[float]): the largest admissible observation values.
        w_min (np.ndarray[float]): the smallest admissible weight values.
        w_max (np.ndarray[float]): the largest admissible weight values.
        w_lvl (int): the number of discrete levels for all weights.
        max_iter (int, optional): the maximum number of iterations. Defaults to 5000.
        err_tol (float, optional): the maximum error to stop the algorithm. Defaults to 1e-3.

    Returns:
        Tuple[np.ndarray[float], Dict[str, Any]]: the bounded weights and the optimization summary.
    """

    firing_selection = np.logical_and(np.isfinite(zt_min), np.isfinite(zt_max))
    min_slope_selection = np.logical_and(np.isfinite(zt_min), ~np.isfinite(zt_max))
    max_level_selection = np.logical_and(~np.isfinite(zt_min), np.isfinite(zt_max))

    wm_min = np.repeat(w_min[:, None], w_lvl - 1, axis=-1)
    wm_max = np.repeat(w_max[:, None], w_lvl - 1, axis=-1)

    # weights are parametrized as zt_min mixture of wlvl-1 binary components, i.e., wk = wk,1 + wk,2 + ... + wk,wlvl-1
    mwm = np.random.uniform(wm_min, wm_max)
    Vwm = np.square(wm_max - wm_min)
    mw, Vw = np.sum(mwm, axis=-1), np.sum(Vwm, axis=-1)
    mz = yt @ mw

    for i in trange(max_iter, desc="Neuron optimization (discrete)"):
        # Compute nuv priors based on posterior means and variances
        mfwm, Vfwm = binary_prior(mwm, Vwm, wm_min, wm_max)
        mfw, Vfw = np.sum(mfwm, axis=-1), np.sum(Vfwm, axis=-1)
        mbzt, Vbzt = box_prior(mz, zt_min, zt_max, 1e-2)

        # Compute weights posterior means and variances by forward message passing
        mw = np.copy(mfw)
        Vw = np.diag(Vfw)
        for ytn, mbztn, Vbztn in zip(yt, mbzt, Vbzt):
            mw, Vw = observation_block_forward(ytn, mw, Vw, mbztn, Vbztn)

        # Compute the duals for propagation through additive boxes
        Wfw = np.diag(1 / Vfw)  # RuntimeWarning: divide by zero encountered in divide
        Wtw = np.diag(Wfw - Wfw @ Vw @ Wfw)  # RuntimeWarning: invalid value encountered in matmul
        xitw = Wfw @ (mfw - mw)  # RuntimeWarning: invalid value encountered in matmul

        # Change back to posterior means and variances of weights discrete components
        mwm = mfwm - Vfwm * xitw[:, None]
        Vwm = Vfwm - Vfwm * Wtw[:, None] * Vfwm

        mw, Vw = np.sum(mwm, axis=-1), np.sum(Vwm, axis=-1)
        mz = yt @ mw

        if box_error(mz, zt_min, zt_max) < err_tol:
            if bin_error(mwm, wm_min, wm_max) < err_tol:
                summary = {
                    "success": True,
                    "num_iter": i,
                    "weight_error": bin_error(mwm, wm_min, wm_max),
                    "potential_error": box_error(mz, zt_min, zt_max),
                    "firing_error": box_error(
                        mz[firing_selection], zt_min[firing_selection], zt_max[firing_selection]
                    ),
                    "min_slope_error": box_error(
                        mz[min_slope_selection], zt_min[min_slope_selection], zt_max[min_slope_selection]
                    ),
                    "max_level_error": box_error(
                        mz[max_level_selection], zt_min[max_level_selection], zt_max[max_level_selection]
                    ),
                }
                return mw, summary
        elif bin_error(mwm, wm_min, wm_max) < err_tol:  # premature binarization
            summary = {
                "success": False,
                "num_iter": i,
                "weight_error": bin_error(mwm, wm_min, wm_max),
                "potential_error": box_error(mz, zt_min, zt_max),
                "firing_error": box_error(mz[firing_selection], zt_min[firing_selection], zt_max[firing_selection]),
                "min_slope_error": box_error(
                    mz[min_slope_selection], zt_min[min_slope_selection], zt_max[min_slope_selection]
                ),
                "max_level_error": box_error(
                    mz[max_level_selection], zt_min[max_level_selection], zt_max[max_level_selection]
                ),
            }
            return mw, summary

    summary = {
        "success": False,
        "num_iter": max_iter,
        "weight_error": bin_error(mwm, wm_min, wm_max),
        "potential_error": box_error(mz, zt_min, zt_max),
        "firing_error": box_error(mz[firing_selection], zt_min[firing_selection], zt_max[firing_selection]),
        "min_slope_error": box_error(
            mz[min_slope_selection], zt_min[min_slope_selection], zt_max[min_slope_selection]
        ),
        "max_level_error": box_error(
            mz[max_level_selection], zt_min[max_level_selection], zt_max[max_level_selection]
        ),
    }
    return mw, summary
