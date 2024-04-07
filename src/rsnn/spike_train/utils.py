from typing import Tuple

import numpy as np


def pmf_num_spikes(period: float, firing_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the probability mass function of the number of spikes in a periodic spike train with a given period and firing rate. Note: for numerical stability, the pmf is computed first in the log domain.

    Args:
        period (float): the period of the spike train in [tau_min].
        firing_rate (float): the firing rate of the spike train in [1/tau_min].

    Returns:
        Tuple[np.ndarray, np.ndarray]: the support of the pmf and the pmf.
    """
    ns = np.arange(period, dtype=int)
    logpns = (ns - 1) * np.log(period - ns) + ns * np.log(firing_rate)
    logpns[1:] -= np.cumsum(np.log(ns[1:]))
    logpns -= np.max(logpns) # to avoid overflow when exponentiating
    pns = np.exp(logpns)
    return ns, pns / np.sum(pns)


def expected_num_spikes(period: float, firing_rate: float) -> float:
    """
    Returns the expected number of spikes in a periodic spike train with a given period and firing rate.

    Args:
        period (float): the period of the spike train in [tau_min].
        firing_rate (float): the firing rate of the spike train in [1/tau_min].

    Returns:
        float: the expected number of spikes.
    """
    ns, pns = pmf_num_spikes(period, firing_rate)
    return np.inner(ns, pns)
