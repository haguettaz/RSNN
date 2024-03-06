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

    tmp_firing_times = firing_times[(firing_times >= t0 - 1) & (firing_times < t0 + period)] % period
    # if the first and last firing times are too close, i.e., they periodically overlap violating the refractory period, remove the last one
    if tmp_firing_times.size > 1 and abs(tmp_firing_times[-1] - tmp_firing_times[0]) < 1.0:
        tmp_firing_times = tmp_firing_times[1:]

    # Bad precision and good recall for empty reference spike train
    if tmp_firing_times.size > 0 and ref_firing_times.size == 0:
        if return_min:
            return 0.0
        return 0.0, 1.0

    # Bad recall and good precision for empty spike train
    if tmp_firing_times.size == 0 and ref_firing_times.size > 0:
        if return_min:
            return 0.0
        return 1.0, 0.0

    # Perfect precision and recall for empty spike trains
    if tmp_firing_times.size == 0 and ref_firing_times.size == 0:
        if return_min:
            return 1.0
        return 1.0, 1.0

    kernel = lambda t_: (np.abs(t_) < eps) * (eps - np.abs(t_)) / eps

    lags = (tmp_firing_times[None, :] - ref_firing_times[:, None]).flatten()
    corr = kernel(dist_mod(tmp_firing_times[None, :, None] - lags[None, None, :], ref_firing_times[:, None, None], period)).sum(axis=(0, 1))

    argmax = np.argmax(corr)
    precision = corr[argmax] / tmp_firing_times.size
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

    # A BIT SLOPPY... NEEDS TO BE CLEANED
    lags = []
    tmp_firing_times = []
    for c in range(num_channels):
        # add some tolerance at the right edge of the window
        fts = firing_times[c][(firing_times[c] >= t0 - 1) & (firing_times[c] < t0 + period)] % period
        # if the first and last firing times are too close, i.e., they periodically overlap violating the refractory period, remove the last one
        if fts.size > 1 and abs(fts[-1] - fts[0]) < 1.0:
            fts = fts[1:]

        tmp_firing_times.append(fts)
        lags.append((tmp_firing_times[-1][None, :] - ref_firing_times[c][:, None]).flatten())
    lags = np.concatenate(lags)

    if sum(fts.size for fts in ref_firing_times) == 0 and sum(fts.size for fts in tmp_firing_times) == 0:
        if return_min:
            return 1.0
        return 1.0, 1.0
    if sum(fts.size for fts in ref_firing_times) == 0 and sum(fts.size for fts in tmp_firing_times) > 0:
        if return_min:
            return 0.0
        return 0.0, 1.0
    if sum(fts.size for fts in ref_firing_times) > 0 and sum(fts.size for fts in tmp_firing_times) == 0:
        if return_min:
            return 0.0
        return 1.0, 0.0

    precision = np.zeros_like(lags)
    recall = np.zeros_like(lags)

    for c in range(num_channels):
        # Both are empty
        if ref_firing_times[c].size == 0 and tmp_firing_times[c].size == 0:
            precision += 1.0  # contribute to every lag
            recall += 1.0  # contribute to every lag
            continue

        # Bad precision
        if ref_firing_times[c].size == 0 and tmp_firing_times[c].size > 0:
            recall += 1.0
            continue

        # Bad recall
        if ref_firing_times[c].size > 0 and tmp_firing_times[c].size == 0:
            precision += 1.0
            continue

        corr = kernel(dist_mod(tmp_firing_times[c][None, :, None] - lags[None, None, :], ref_firing_times[c][:, None, None], period)).sum(axis=(0, 1))
        precision += corr / tmp_firing_times[c].size
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
