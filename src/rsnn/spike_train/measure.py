from typing import List

import numpy as np

from ..utils.utils import mod
from .periodic_spike_train import MultiChannelPeriodicSpikeTrain, PeriodicSpikeTrain
from .spike_train import MultiChannelSpikeTrain, SpikeTrain

# def local_correlation(
#     periodic_spike_trains: List[np.ndarray],
#     spike_trains: List[np.ndarray],
#     period: float,
#     eps: float = 1.0,
# ):
#     """
#     Compute the local correlation between an ideal (periodic) spike train and a real one.
#     Local means the correlation is computed for each channel separately.

#     Args:
#         periodic_spike_trains (List[np.ndarray]): the firing times of the ideal (period) spike train.
#         spike_trains (List[np.ndarray]): the firing times of the actual spike train.
#         period (float): the period of the ideal spike train.
#         eps (float, optional): the half-width of the triangular kernel in [ms]. Defaults to 1.0.

#     Returns:
#         (float): the maximal correlation between the two spike trains.
#         (float): the lag at which the correlation is maximal.
#     """

#     kernel = lambda x_: (np.abs(x_) < eps) * (eps - np.abs(x_)) / eps

#     if not isinstance(periodic_spike_trains, List):
#         raise TypeError("periodic_spike_trains must be a list of numpy arrays.")
#     if not isinstance(spike_trains, List):
#         raise TypeError("spike_trains must be a list of numpy arrays.")

#     max_corr, max_shift = [], []
#     for hat_s, s in zip(periodic_spike_trains, spike_trains):
#         if (hat_s.shape[0] == 0) != (
#             s.shape[0] == 0
#         ):  # exactly one spike train is empty
#             max_corr.append(0.0)
#             max_shift.append(0.0)
#             continue

#         if (hat_s.shape[0] == 0) and (s.shape[0] == 0):  # both spike trains are empty
#             max_corr.append(1.0)
#             max_shift.append(0.0)
#             continue

#         t = (hat_s[None, :] - s[:, None]).flatten()
#         corr = kernel((t[None, :] - t[:, None]) % period).sum(axis=1)
#         argmax = np.argmax(corr)
#         max_corr.append(corr[argmax] / max(hat_s.shape[0], s.shape[0]))
#         max_shift.append(mod(t[argmax], period, -period / 2))

#     return np.array(max_corr), np.array(max_shift)


def single_channel_correlation(periodic_spike_train: PeriodicSpikeTrain, spike_train: SpikeTrain, eps: float = 1.0):
    """
    Compute the correlation between a (periodic) spike train and a spike train, using a triangular kernel.

    Args:
        periodic_spike_train (PeriodicSpikeTrain): the periodic spike train.
        spike_train (SpikeTrain): the spike train.
        eps (float, optional): the half-width of the triangular kernel in [ms]. Defaults to 1.0.

    Returns:
        (float): the maximal correlation between the two spike trains.
        (float): the lag at which the correlation is maximal.
    """
    # Define the triangular kernel
    kernel = lambda x_: (np.abs(x_) < eps) * (eps - np.abs(x_)) / eps

    if periodic_spike_train.num_spikes == 0:
        if spike_train.num_spikes == 0:
            return 1.0, 0.0
        return 0.0, np.nan

    if spike_train.num_spikes == 0:
        return 0.0, np.nan

    # Compute all possible correlation maximizers
    lags = (periodic_spike_train.firing_times[None, :] - spike_train.firing_times[:, None]).flatten()

    corr = kernel(
        mod(lags[None, :] - lags[:, None], periodic_spike_train.period, -periodic_spike_train.period / 2)
    ).sum(axis=0)

    argmax = np.argmax(corr)
    return corr[argmax] / max(periodic_spike_train.num_spikes, spike_train.num_spikes), mod(
        lags[argmax], periodic_spike_train.period, -periodic_spike_train.period / 2
    )


# Compute the correlation between a (periodic) multichannel spike train and a multichannel spike train.
def multi_channel_correlation(
    multi_channel_periodic_spike_train: MultiChannelPeriodicSpikeTrain,
    multi_channel_spike_train: MultiChannelSpikeTrain,
    eps: float = 1.0,
):
    """
    Args:
        multi_channel_periodic_spike_train (MultiChannelPeriodicSpikeTrain): the multi-channel periodic spike train.
        multi_channel_spike_train (MultiChannelSpikeTrain): the multi-channel spike train.
        eps (float, optional): the half-width of the triangular kernel in [ms]. Defaults to 1.0.

    Raises:
        TypeError: if periodic_spike_trains is not a list of PeriodicSpikeTrain.
        TypeError: if spike_trains is not a list of SpikeTrain.
        ValueError: if periodic_spike_trains and spike_trains do not have the same number of channels.

    Returns:
        (float): the maximal correlation between the two spike trains.
        (float): the lag at which the correlation is maximal.
    """

    # TODO: implement this function

    # Define the triangular kernel
    kernel = lambda x_: (np.abs(x_) < eps) * (eps - np.abs(x_)) / eps

    # Check the number of channels
    if multi_channel_periodic_spike_train.num_channels != multi_channel_spike_train.num_channels:
        raise ValueError("Number of channel does not match.")

    if multi_channel_periodic_spike_train.num_spikes == 0:
        if multi_channel_spike_train.num_spikes == 0:
            return 1.0, 0.0
        return 0.0, np.nan

    if multi_channel_spike_train.num_spikes == 0:
        return 0.0, np.nan

    # Compute all possible correlation maximizers
    lags = np.concatenate(
        [
            (periodic_spike_train.firing_times[None, :] - spike_train.firing_times[:, None]).flatten()
            for periodic_spike_train, spike_train in zip(multi_channel_periodic_spike_train.spike_trains, multi_channel_spike_train.spike_trains)
        ]
    )

    # Compute correlation for each channel
    corr = np.zeros(lags.size)
    for periodic_spike_train, spike_train in zip(multi_channel_periodic_spike_train.spike_trains, multi_channel_spike_train.spike_trains):
        if periodic_spike_train.num_spikes == 0:
            if spike_train.num_spikes == 0:
                corr += 1.0 # contribute to every lag
            continue

        if spike_train.num_spikes == 0:
            continue

        tmp = (spike_train.firing_times[None, :] - periodic_spike_train.firing_times[:, None]).flatten()
        corr += kernel(
            mod(lags[None, :] - tmp[:, None], multi_channel_periodic_spike_train.period, multi_channel_periodic_spike_train.period / 2)
        ).sum(axis=0) / max(periodic_spike_train.num_spikes, spike_train.num_spikes)

    # Find the maximum correlation and its lag
    argmax = np.argmax(corr)
    return corr[argmax] / multi_channel_periodic_spike_train.num_channels, mod(
        lags[argmax], multi_channel_periodic_spike_train.period, multi_channel_periodic_spike_train.period / 2
    )
