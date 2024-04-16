from typing import Callable, List

import numpy as np

from ..utils.math import dist_mod, mod
from .utils import check_refractoriness, check_refractoriness_periodicity


def single_channel_correlation(nominal_spike_train: np.ndarray, spike_train: np.ndarray, t0: float, period: float, return_min:bool = False):
    """
    Args:
        nominal_spike_trains (np.ndarray): The nominal single-channel periodic spike train.
        spike_train (np.ndarray): The single-channel spike train.
        t0 (float): The starting time of the spike trains [in tau_0].
        period (float): The period of the spike trains [in tau_0].
        return_min (bool, optional): Whether to return the minimum of precision and recall. Defaults to False.

    Raises:
        TypeError: If periodic_spike_trains is not a list of PeriodicSpikeTrain.
        TypeError: If spike_train is not a list of SpikeTrain.
        ValueError: If periodic_spike_trains and spike_train do not have the same number of channels.

    Returns:
        (float): The precision of the single-channel spike trains w.r.t the reference.
        (float): The recall of the single-channel spike trains w.r.t the reference.
    """
    if not check_refractoriness_periodicity(nominal_spike_train, period):
        raise ValueError("The nominal spike train does not satisfy the refractory condition.")

    tmp_firing_times = np.sort(spike_train[(spike_train >= t0 - 1) & (spike_train < t0 + period)])
    if not check_refractoriness(tmp_firing_times):
        raise ValueError("The spike train does not satisfy the refractory condition.")
    
    # if the first and last firing times are too close, i.e., they periodically overlap violating the refractory period, remove the last one
    if tmp_firing_times.size > 1 and np.abs(tmp_firing_times[-1] - period - tmp_firing_times[0]) < 1.0:
        tmp_firing_times = tmp_firing_times[1:]

    # Bad precision and good recall for empty nominal spike train
    if tmp_firing_times.size > 0 and nominal_spike_train.size == 0:
        if return_min:
            return 0.0
        return 0.0, 1.0

    # Bad recall and good precision for empty spike train
    if tmp_firing_times.size == 0 and nominal_spike_train.size > 0:
        if return_min:
            return 0.0
        return 1.0, 0.0

    # Perfect precision and recall for empty spike trains
    if tmp_firing_times.size == 0 and nominal_spike_train.size == 0:
        if return_min:
            return 1.0
        return 1.0, 1.0

    kernel = lambda t_: (np.abs(t_) < 0.5) * (0.5 - np.abs(t_)) / 0.5

    lags = (tmp_firing_times[None, :] - nominal_spike_train[:, None]).flatten()
    corr = kernel(dist_mod(tmp_firing_times[None, :, None] - lags[None, None, :], nominal_spike_train[:, None, None], period)).sum(axis=(0, 1))

    argmax = np.argmax(corr)
    precision = corr[argmax] / tmp_firing_times.size
    recall = corr[argmax] / nominal_spike_train.size

    if return_min:
        return np.minimum(precision, recall)
    return precision, recall


def multi_channel_correlation(
    nominal_spike_trains: List[np.ndarray], spike_trains: List[np.ndarray], t0: float, period: float, return_min:bool = False
):
    """
    Args:
        nominal_spike_trains (List[np.ndarray]): The reference multi-channel periodic spike train.
        spike_trains (List[np.ndarray]): The multi-channel spike train.
        t0 (float): The starting time of the spike trains [in tau_0].
        period (float): The period of the spike trains [in tau_0].
        return_min (bool, optional): Whether to return the minimum of precision and recall. Defaults to False.
    Raises:
        TypeError: If periodic_spike_trains is not a list of PeriodicSpikeTrain.
        TypeError: If spike_trains is not a list of SpikeTrain.
        ValueError: If periodic_spike_trains and spike_trains do not have the same number of channels.

    Returns:
        (float): The precision of the multi-channel spike trains w.r.t the reference.
        (float): The recall of the multi-channel spike trains w.r.t the reference.
    """
    # Check the number of channels
    if len(spike_trains) != len(nominal_spike_trains):
        raise ValueError("Number of channel does not match.")
    num_channels = len(spike_trains)

    kernel = lambda t_: (np.abs(t_) < 0.5) * (0.5 - np.abs(t_)) / 0.5

    lags = []
    tmp_spike_trains = []
    for nominal_spike_train, spike_train in zip(nominal_spike_trains, spike_trains):
        if not check_refractoriness_periodicity(nominal_spike_train, period):
            raise ValueError(f"The nominal spike train {nominal_spike_train} does not satisfy the refractory and periodicity conditions.")

        tmp_spike_train = np.sort(spike_train[(spike_train >= t0 - 1) & (spike_train < t0 + period)])
        if not check_refractoriness(tmp_spike_train):
            raise ValueError(f"The spike train {tmp_spike_train} does not satisfy the refractory condition.")
        
        # if the first and last firing times are too close, i.e., they periodically overlap violating the refractory period, remove the last one
        if tmp_spike_train.size > 1 and np.abs(tmp_spike_train[-1] - period - tmp_spike_train[0]) < 1.0:
            tmp_spike_train = tmp_spike_train[1:]

        tmp_spike_trains.append(tmp_spike_train)
        lags.append((tmp_spike_train[None, :] - nominal_spike_train[:, None]).flatten())

    lags = np.concatenate(lags)

    if sum(fts.size for fts in nominal_spike_trains) == 0 and sum(fts.size for fts in tmp_spike_trains) == 0:
        if return_min:
            return 1.0
        return 1.0, 1.0
    if sum(fts.size for fts in nominal_spike_trains) == 0 and sum(fts.size for fts in tmp_spike_trains) > 0:
        if return_min:
            return 0.0
        return 0.0, 1.0
    if sum(fts.size for fts in nominal_spike_trains) > 0 and sum(fts.size for fts in tmp_spike_trains) == 0:
        if return_min:
            return 0.0
        return 1.0, 0.0

    precision = np.zeros_like(lags)
    recall = np.zeros_like(lags)

    for nominal_spike_train, spike_train in zip(nominal_spike_trains, tmp_spike_trains):
        # Both are empty
        if nominal_spike_train.size == 0 and spike_train.size == 0:
            precision += 1.0  # contribute to every lag
            recall += 1.0  # contribute to every lag
            continue

        # Bad precision
        if nominal_spike_train.size == 0 and spike_train.size > 0:
            recall += 1.0
            continue

        # Bad recall
        if nominal_spike_train.size > 0 and spike_train.size == 0:
            precision += 1.0
            continue

        corr = kernel(dist_mod(spike_train[None, :, None] - lags[None, None, :], nominal_spike_train[:, None, None], period)).sum(axis=(0, 1))
        precision += corr / spike_train.size
        recall += corr / nominal_spike_train.size

    if return_min:
        return np.minimum(np.max(precision), np.max(recall)) / num_channels
    
    return np.max(precision) / num_channels, np.max(recall) / num_channels
