from typing import Callable, List

import numpy as np

from ..utils.math import dist_mod, mod


def single_channel_correlation(
    ref_firing_times: np.ndarray, firing_times: np.ndarray, period: float, absolute_refractory:float, eps: float
):
    """
    Args:
        ref_firing_times (np.ndarray): The reference single-channel periodic spike train.
        firing_times (np.ndarray): The single-channel spike train.
        period (float): The period of the spike trains in [ms].
        absolute_refractory (float): The absolute refractory time in [ms].
        eps (float, optional): The half-width of the triangular kernel in [ms]. Defaults to 1.0.

    Raises:
        TypeError: If periodic_spike_trains is not a list of PeriodicSpikeTrain.
        TypeError: If spike_trains is not a list of SpikeTrain.
        ValueError: If periodic_spike_trains and spike_trains do not have the same number of channels.

    Returns:
        (float): The precision of the single-channel spike trains w.r.t the reference.
        (float): The recall of the single-channel spike trains w.r.t the reference.
        (float): The lag of the single-channel spike trains w.r.t the reference.
    """
    if eps <= 0 or eps > absolute_refractory/2:
        raise ValueError("The half-width of the triangular kernel must be positive and smaller than the absolute refractory period.")

    # Worst precision for empty reference spike train
    if firing_times.size and not ref_firing_times.size:
        return 0.0, np.nan, np.nan
    
    # Worst recall for empty spike train
    if not firing_times.size and ref_firing_times.size:
        return np.nan, 0.0, np.nan

    # Perfect precision and recall for empty spike trains
    if not firing_times.size and not ref_firing_times.size:
        return 1.0, 1.0, 0.0

    kernel = lambda t_: (np.abs(t_) < eps) * (eps - np.abs(t_)) / eps

    lags = (ref_firing_times[None, :] - firing_times[:, None]).reshape(-1)
    corr = kernel(dist_mod(lags[None, :], lags[:, None], period)).sum(axis=0)
    argmax = np.argmax(corr)

    precision = corr[argmax] / firing_times.size
    recall = corr[argmax] / ref_firing_times.size
    lag = mod(lags[argmax], period, -period / 2)

    return precision, recall, lag

def multi_channel_correlation(
    ref_firing_times: List[np.ndarray], firing_times: List[np.ndarray], period: float, absolute_refractory:float, eps: float
):
    """
    Args:
        ref_firing_times (List[np.ndarray]): The reference multi-channel periodic spike train.
        firing_times (List[np.ndarray]): The multi-channel spike train.
        period (float): The period of the spike trains in [ms].
        absolute_refractory (float): The absolute refractory time in [ms].
        eps (float, optional): The half-width of the triangular kernel in [ms]. Defaults to 1.0.

    Raises:
        TypeError: If periodic_spike_trains is not a list of PeriodicSpikeTrain.
        TypeError: If spike_trains is not a list of SpikeTrain.
        ValueError: If periodic_spike_trains and spike_trains do not have the same number of channels.

    Returns:
        (float): The precision of the multi-channel spike trains w.r.t the reference.
        (float): The recall of the multi-channel spike trains w.r.t the reference.
        (float): The lag of the multi-channel spike trains w.r.t the reference.
    """
    # Check the number of channels
    if len(firing_times) != len(ref_firing_times):
        raise ValueError("Number of channel does not match.")
    
    if eps <= 0 or eps > absolute_refractory/2:
        raise ValueError("The half-width of the triangular kernel must be positive and smaller than the absolute refractory period.")

    num_firing_times = np.sum([fts.size for fts in firing_times])
    num_ref_firing_times = np.sum([rfts.size for rfts in ref_firing_times])

    # Worst precision for empty reference spike train
    if num_firing_times and not num_ref_firing_times:
        return 0.0, np.nan, np.nan

    # Worst recall for empty spike train
    if not num_firing_times and num_ref_firing_times:
        return np.nan, 0.0, np.nan
        # return 0.0, 0.0,  np.nan

    # Perfect precision and recall for empty spike trains
    if not num_firing_times and not num_ref_firing_times:
        return 1.0, 1.0, 0.0

    # Compute all possible correlation maximizers
    kernel = lambda t_: (np.abs(t_) < eps) * (eps - np.abs(t_)) / eps
    
    lags = np.concatenate(
        [
            (rfts[None, :] - fts[:, None]).flatten()
            for fts, rfts in zip(
                firing_times, ref_firing_times
            )
        ]
    )

    corr = np.zeros(lags.size)

    for rfts, fts in zip(
        ref_firing_times, firing_times
    ):
        if not rfts.size and not fts.size:
            corr += 1.0  # contribute to every lag
            continue

        corr += kernel(dist_mod(lags[None, :], (fts[None, :] - rfts[:, None]).reshape(-1, 1), period)).sum(axis=0)

    # Find the maximum correlation and its lag
    argmax = np.argmax(corr)

    precision = corr[argmax] / num_firing_times
    recall = corr[argmax] / num_ref_firing_times
    lag = mod(lags[argmax], period, -period / 2)

    return precision, recall, lag