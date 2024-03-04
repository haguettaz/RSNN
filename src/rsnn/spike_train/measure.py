from typing import Callable, List

import numpy as np

from ..utils.math import dist_mod, mod


def single_channel_correlation(ref_firing_times: np.ndarray, firing_times: np.ndarray, t0: float, period: float, eps: float = 0.5, return_min:bool = False):
    """
    Args:
        ref_firing_times (np.ndarray): The reference single-channel periodic spike train.
        firing_times (np.ndarray): The single-channel spike train.
        period (float): The period of the spike trains in [ms].
        eps (float, optional): The half-width of the triangular kernel in [ms]. Defaults to 0.5.

    Raises:
        TypeError: If periodic_spike_trains is not a list of PeriodicSpikeTrain.
        TypeError: If spike_trains is not a list of SpikeTrain.
        ValueError: If periodic_spike_trains and spike_trains do not have the same number of channels.

    Returns:
        (float): The precision of the single-channel spike trains w.r.t the reference.
        (float): The recall of the single-channel spike trains w.r.t the reference.
        (float): The lag of the single-channel spike trains w.r.t the reference.
    """
    if eps <= 0 or eps > 0.5:
        raise ValueError("The half-width of the triangular kernel must be positive and smaller than the absolute refractory period.")

    # make sure no firing times is too close to the edges of the window
    tmin = firing_times[np.searchsorted(firing_times, t0, side="right") - 1]  # np.searchsorted(firing_times, t0, side="right") - 1
    tmax = firing_times[np.searchsorted(firing_times, t0 + period, side="right") - 1]  #
    if (tmin - t0) < 0.5:
        tmin -= 0.5
        tmax = tmin + period
    elif (tmax - (t0 + period)) < 0.5:
        tmax += 0.5
        tmin = tmax - period
    else:
        tmin = t0
        tmax = tmin + period
    firing_times = firing_times[(firing_times >= tmin) & (firing_times < tmax)] % period

    # Bad precision and good recall for empty reference spike train
    if firing_times.size and not ref_firing_times.size:
        if return_min:
            return 0.0
        return 0.0, 1.0

    # Bad recall and good precision for empty spike train
    if not firing_times.size and ref_firing_times.size:
        if return_min:
            return 0.0
        return 1.0, 0.0

    # Perfect precision and recall for empty spike trains
    if not firing_times.size and not ref_firing_times.size:
        if return_min:
            return 1.0
        return 1.0, 1.0

    kernel = lambda t_: (np.abs(t_) < eps) * (eps - np.abs(t_)) / eps

    lags = (firing_times[None, :] - ref_firing_times[:, None]).flatten()
    corr = kernel(dist_mod(firing_times[None, :, None] - lags[None, None, :], ref_firing_times[:, None, None], period)).sum(axis=(0, 1))

    argmax = np.argmax(corr)
    precision = corr[argmax] / firing_times.size
    recall = corr[argmax] / ref_firing_times.size

    if return_min:
        return np.minimum(precision, recall)
    return precision, recall


def multi_channel_correlation(
    ref_firing_times: List[np.ndarray], firing_times: List[np.ndarray], t0: float, period: float, eps: float=0.5, return_min:bool = False
):
    """
    Args:
        ref_firing_times (List[np.ndarray]): The reference multi-channel periodic spike train.
        firing_times (List[np.ndarray]): The multi-channel spike train.
        period (float): The period of the spike trains in [ms].
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
    num_channels = len(firing_times)

    if eps <= 0 or eps > 0.5:
        raise ValueError("The half-width of the triangular kernel must be positive and smaller than the absolute refractory period.")

    # Compute all possible correlation maximizers
    kernel = lambda t_: (np.abs(t_) < eps) * (eps - np.abs(t_)) / eps

    lags = []
    for c in range(num_channels):
        # make sure no firing times is too close to the edges of the window
        tmin = firing_times[c][np.searchsorted(firing_times[c], t0, side="right") - 1]  # np.searchsorted(firing_times, t0, side="right") - 1
        tmax = firing_times[c][np.searchsorted(firing_times[c], t0 + period, side="right") - 1]  #
        if (tmin - t0) < 0.5:
            tmin -= 0.5
            tmax = tmin + period
        elif (tmax - (t0 + period)) < 0.5:
            tmax += 0.5
            tmin = tmax - period
        else:
            tmin = t0
            tmax = tmin + period
        firing_times[c] = firing_times[c][(firing_times[c] >= tmin) & (firing_times[c] < tmax)] % period
        lags.append((firing_times[c][None, :] - ref_firing_times[c][:, None]).flatten())
    lags = np.concatenate(lags)

    precision = np.zeros_like(lags)
    recall = np.zeros_like(lags)

    for c in range(num_channels):
        # Both are empty
        if not ref_firing_times[c].size and not firing_times[c].size:
            precision += 1.0  # contribute to every lag
            recall += 1.0  # contribute to every lag
            continue

        # Bad precision
        if not ref_firing_times[c].size and firing_times[c].size:
            recall += 1.0
            continue

        # Bad recall
        if ref_firing_times[c].size and not firing_times[c].size:
            precision += 1.0
            continue

        corr = kernel(dist_mod(firing_times[c][None, :, None] - lags[None, None, :], ref_firing_times[c][:, None, None], period)).sum(axis=(0, 1))
        precision += corr / firing_times[c].size
        recall += corr / ref_firing_times[c].size

    if return_min:
        return np.minimum(np.max(precision), np.max(recall)) / num_channels
    
    return np.max(precision) / num_channels, np.max(recall) / num_channels
    # if return_singles:
    #     single_precision = np.zeros(num_channels)
    #     single_recall = np.zeros(num_channels)

    #     lag_precision = lags[idx_precision]
    #     lag_recall = lags[idx_recall]

    #     for c in range(num_channels):
    #         single_precision[c] = kernel(dist_mod(firing_times[c][None, :] - lag_precision, ref_firing_times[c][:, None], period)).sum() / firing_times[c].size
    #         single_recall[c] = kernel(dist_mod(firing_times[c][None, :] - lag_recall, ref_firing_times[c][:, None], period)).sum() / ref_firing_times[c].size

    #     return precision[idx_precision] / num_channels, recall[idx_recall] / num_channels, single_precision, single_recall

    # return precision[idx_precision] / num_channels, recall[idx_recall] / num_channels
