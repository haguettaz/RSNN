from typing import Tuple

import numpy as np


def pmf_num_spikes(period: float, firing_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the probability mass function of the number of spikes in a periodic spike train with a given period and firing rate. Note: for numerical stability, the pmf is computed first in the log domain.

    Args:
        period (float): the period of the spike train in [tau_0].
        firing_rate (float): the firing rate of the spike train in [1/tau_0].

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
        period (float): the period of the spike train in [tau_0].
        firing_rate (float): the firing rate of the spike train in [1/tau_0].

    Returns:
        float: the expected number of spikes.
    """
    ns, pns = pmf_num_spikes(period, firing_rate)
    return np.inner(ns, pns)


def check_refractoriness(firing_times: np.ndarray, tol:float=1e-6) -> bool:
    """
    Returns a boolean indicating whether the spike train satisfies the refractory condition.

    Args:
        firing_times (np.ndarray): the spike times in [tau_0].
        tol (float, optional): the tolerance for the refractory condition. Defaults to 1e-6.

    Returns:
        (bool): the boolean indicating satisfaction of the refractory condition.
    """
    return (firing_times.size < 2) or np.all(np.diff(np.sort(firing_times)) > 1.0 - tol)
        
def check_refractoriness_periodicity(firing_times: np.ndarray, period:float, tol:float=1e-6) -> bool:
    """
    Returns a boolean indicating whether the spike train satisfies the refractory condition.

    Args:
        firing_times (np.ndarray): the spike times in [tau_0].
        period (float): the period of the spike train in [tau_0].
        tol (float, optional): the tolerance for the refractory condition. Defaults to 1e-6.

    Returns:
        (bool): the boolean indicating satisfaction of the refractory condition.
    """
    if firing_times.size == 0:
        return True
    
    if firing_times.size == 1:
        return period > 1.0
    
    tmp = np.sort(firing_times)
        
    return (tmp[-1] - tmp[0] < period) and (np.abs(tmp[-1] - period - tmp[0]) > 1.0 - tol) and np.all(np.diff(tmp) > 1.0 - tol)