from typing import List, Optional, Union

import numpy as np
from scipy.stats import truncnorm

from .utils import pmf_num_spikes


def sample_spike_trains(
    period: float,
    firing_rate: float,
    num_channels: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Returns a single- or multi-channel periodic spike train.

    Args:
        period (float): The period of the spike train in [tau_0].
        firing_rate (float): The firing rate of the spike train in [1/\tau_0].
        num_channels (int, optional): The number of channels / neurons. Defaults to 1.
        rng (np.random.Generator, optional): The random number generator. Defaults to None.

    Raises:
        ValueError: If the period is negative.
        ValueError: If the firing rate is negative.

    Returns:
        (np.ndarray or List[np.ndarray]): a single- or multi-channel periodic spike train.
    """
    if period < 0:
        raise ValueError(f"The period should be non-negative.")

    if firing_rate < 0:
        raise ValueError(f"The firing rate should be non-negative.")

    if period <= 1 or firing_rate == 0:
        if num_channels is None:
            return np.array([])
        return [np.array([])] * num_channels

    if rng is None:
        rng = np.random.default_rng()

    spike_trains = []

    ns, pns = pmf_num_spikes(period, firing_rate)

    if num_channels is None:
        # sample the number of spikes in [0, period)
        n = rng.choice(ns, p=pns)
        if n == 0:
            return np.array([])

        # sample the effective poisson process in [0, period-n)
        ts = np.full(n, rng.uniform(0, period))
        ts[1:] += np.sort(rng.uniform(0, period - n, n - 1)) + np.arange(1, n)
        # transform the effective poisson process into a periodic spike train ...
        return np.sort(ts % period)

    for _ in range(num_channels):
        # sample the number of spikes in [0, period)
        n = rng.choice(ns, p=pns)
        if n == 0:
            spike_trains.append(np.array([]))
            continue
        # sample the effective poisson process in [0, period-n)
        ts = np.full(n, rng.uniform(0, period))
        ts[1:] += np.sort(rng.uniform(0, period - n, n - 1)) + np.arange(1, n)
        # transform the effective poisson process into a periodic spike train ...
        spike_trains.append(np.sort(ts % period))

    return spike_trains


def sample_jittered_spike_train(
    spike_train: np.ndarray,
    std_jitter: float,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    niter: Optional[int] = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Returns a list of jittered spike trains, generated by adding Gaussian noise to the spike times and checking for refractoriness.
    It uses the Gibbs sampler.

    Args:
        spike_train (np.ndarray): the nominal firing locations
        std_jitter (float): the standard deviation of the Gaussian jitter noise.
        tmin (float, optional): the lower bound of the time range. Defaults to None.
        tmax (float, optional): the upper bound of the time range. Defaults to None.
        niter (int, optional): the maximum number of iterations. Defaults to 1000.
        rng (np.random.Generator, optional): the random number generator. Defaults to None.

    Returns:
        np.ndarray: the jittered spike train
    """
    if rng is None:
        rng = np.random.default_rng()

    n = spike_train.size

    if n == 0:
        return np.array([])

    if tmin is None:
        tmin = -np.inf
    if tmax is None:
        tmax = np.inf

    sampler = lambda a_, b_, loc_: truncnorm.rvs(
        (a_ - loc_) / std_jitter,
        (b_ - loc_) / std_jitter,
        loc=loc_,
        scale=std_jitter,
        random_state=rng,
    )

    s = np.empty_like(spike_train)

    if n == 1:
        s[0] = sampler(tmin, tmax, spike_train)
        return s

    n_is_odd = bool(n % 2)
    odd_indices = np.arange(1, n - 1, 2)
    even_indices = np.arange(2, n - 1, 2)

    prev_s = np.copy(spike_train)
    if np.isfinite(tmin):
        prev_s = np.maximum(prev_s, tmin + np.arange(n) + 1e-3)
    if np.isfinite(tmax):
        prev_s = np.minimum(prev_s, tmax - np.arange(n - 1, -1, -1) - 1e-3)

    for _ in range(1, niter):
        # fix odd indices and sample the even ones
        s[0] = sampler(tmin, prev_s[1] - 1, spike_train[0])
        if n_is_odd:
            s[-1] = sampler(prev_s[-2] + 1, tmax, spike_train[-1])
        s[even_indices] = sampler(
            prev_s[even_indices - 1] + 1,
            prev_s[even_indices + 1] - 1,
            spike_train[even_indices],
        )

        # fix even indices and sample odd ones
        s[odd_indices] = sampler(
            s[odd_indices - 1] + 1,
            s[odd_indices + 1] - 1,
            spike_train[odd_indices],
        )
        if not n_is_odd:
            s[-1] = sampler(s[-2] + 1, tmax, spike_train[-1])

        prev_s = np.copy(s)

    return s
