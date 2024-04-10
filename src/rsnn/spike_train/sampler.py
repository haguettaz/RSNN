
from typing import List, Optional, Union

import numpy as np

from .utils import check_refractoriness, pmf_num_spikes


def sample_spike_trains(period:float, firing_rate:float, num_channels:Optional[int]=None, rng: Optional[np.random.Generator]=None) -> Union[np.ndarray, List[np.ndarray]]:
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
        return [np.array([])]*num_channels
    
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
        ts[1:] += np.sort(rng.uniform(0, period-n, n-1)) + np.arange(1, n)
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
        ts[1:] += np.sort(rng.uniform(0, period-n, n-1)) + np.arange(1, n)
        # transform the effective poisson process into a periodic spike train ...
        spike_trains.append(np.sort(ts % period))

    return spike_trains

def sample_jittered_spike_trains(spike_trains:Union[np.ndarray, List[np.ndarray]], tmin:float, tmax:float, sigma:float, gauss:Optional[bool]=True, maxiter:Optional[int]=1000, rng: Optional[np.random.Generator]=None) -> List[np.ndarray]:
    """
    Returns a list of jittered spike trains, generated by adding Gaussian noise to the spike times and checking for refractoriness.
    This is a rejection sampling method.

    Args:
        spike_trains (List[np.ndarray]): the nominal firing locations
        tmin (float): the lower bound of the time range
        tmax (float): the upper bound of the time range
        sigma (float): the standard deviation of the Gaussian noise or the half-width of the uniform noise
        gauss (bool, optional): whether to use Gaussian noise. Defaults to True.
        maxiter (int, optional): the maximum number of iterations. Defaults to 1000.
        rng (np.random.Generator, optional): the random number generator. Defaults to None.

    Returns:
        List[np.ndarray]: the jittered spike trains
    """
    if rng is None:
        rng = np.random.default_rng()

    if gauss:
        jitter_dist = lambda size: rng.normal(0, sigma, size)
    else:
        jitter_dist = lambda size: rng.uniform(-sigma, sigma, size)

    if isinstance(spike_trains, np.ndarray):
        for _ in range(maxiter):
            tmp_spike_train = spike_trains + jitter_dist(spike_trains.size)
            if np.all(tmp_spike_train >= tmin) and np.all(tmp_spike_train < tmax) and check_refractoriness(tmp_spike_train):
                return np.sort(tmp_spike_train)
        
        raise ValueError(f"Could not generate a jittered spike train for the given spike train {firing_times}.")

    jittered_spike_trains = []
    for spike_train in spike_trains:
        for i in range(maxiter):
            tmp_spike_train = spike_train + jitter_dist(spike_train.size)
            if np.all(tmp_spike_train >= tmin) and np.all(tmp_spike_train < tmax) and check_refractoriness(tmp_spike_train):
                jittered_spike_trains.append(np.sort(tmp_spike_train))
                break
        
        if i == maxiter - 1:
            raise ValueError(f"Could not generate a jittered spike train for the given spike train {firing_times}.")

    return jittered_spike_trains