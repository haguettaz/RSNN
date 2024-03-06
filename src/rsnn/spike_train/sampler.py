from typing import List, Optional, Union

import numpy as np
from tqdm.autonotebook import trange


def backward_sampling(period:float, firing_rate:float, num_channels:int=1, res:float=1e-3) -> List[np.ndarray]:
    """
    Returns a single- or multi-channel periodic spike train by backward sampling.

    Args:
        period (float): The period of the spike train in [tau_min].
        firing_rate (float): The firing rate of the spike train in [1/\tau_min].
        num_channels (int, optional): The number of channels / neurons. Defaults to 1.
        res (float, optional): The resolution of the sampling procedure. Defaults to 1e-3.
        
    Raises:
        ValueError: If the period is negative.
        ValueError: If the firing rate is negative.

    Returns:
        (List[np.ndarray]): a single- or multi-channel periodic spike train.
    """            
    if period < 0:
        raise ValueError(f"The period should be non-negative.")
        
    if firing_rate < 0:
        raise ValueError(f"The firing rate should be non-negative.")
    
    if period <= 1 or firing_rate == 0:
        return [np.array([])]*num_channels
    
    ns = np.arange(1, period, dtype=int)
    
    #pns = np.nan_to_num(np.power(firing_rate * (period - ns)*np.exp(firing_rate), ns) / ((period-ns)*factorial(ns-1)))
    #pns /= np.sum(pns) # normalize
    
    # for numerical stability, first compute log p(n | n > 0) (unnormalized)
    lnpns = (ns-1) * (np.log(firing_rate) + np.log(period - ns)) 
    lnpns += np.log(firing_rate) 
    lnpns -= firing_rate * (period - ns)
    lnpns[1:] -= np.cumsum(np.log(ns[:-1]))
    
    # then, compute and normalize p(n | n > 0)
    pns = np.exp(lnpns)
    pns /= np.sum(pns)

    # init the list of arrays of firing times
    firing_times = []

    for _ in trange(num_channels, desc="Sampling"):        
        # empty spike train?
        if np.random.binomial(1, np.exp(-firing_rate*(period - 1))):
            firing_times.append(np.array([]))
            continue

        # number of spikes (conditionned on non empty)? (WARNING: possibility of overflow error)
        n = np.random.choice(ns, p=pns)

        # firing times? (conditionned on number of spikes and spikes at location 0 and period)
        tfs = np.full(n, period, dtype=float)
        for m in range(n - 1, 0, -1):
            ts = np.arange(m+res, tfs[m]-1, res)
            pts = np.nan_to_num(np.power((ts-m)/(tfs[m]-m-1), m-1) * m / (ts-m))
            pts = pts / pts.sum()

            #psm = norm(fmus[m] * self.survival_dist.pdf(s[m + 1] - z))
            tfs[m-1] = np.random.choice(ts, p=pts)

        # Ramdomly shift the firing times cyclically
        firing_times.append(np.sort((tfs + np.random.uniform(0, period)) % period))

    return firing_times

def forward_sampling(period:float, firing_rate:float, num_channels:int=1, res:float=1e-3) -> List[np.ndarray]:
    """
    Returns a single- or multi-channel periodic spike train by forward sampling.

    Args:
        period (float): The period of the spike train in [tau_min].
        firing_rate (float): The firing rate of the spike train in [1/\tau_min].
        num_channels (int, optional): The number of channels / neurons. Defaults to 1.
        res (float, optional): The resolution of the sampling procedure. Defaults to 1e-3.
        
    Raises:
        ValueError: If the period is negative.
        ValueError: If the firing rate is negative.

    Returns:
        (List[np.ndarray]): a single- or multi-channel periodic spike train.
    """            
    if period < 0:
        raise ValueError(f"The period should be non-negative.")
        
    if firing_rate < 0:
        raise ValueError(f"The firing rate should be non-negative.")
    
    if period <= 1 or firing_rate == 0:
        return [np.array([])]*num_channels
    
    ns = np.arange(1, period, dtype=int)
    
    #pns = np.nan_to_num(np.power(firing_rate * (period - ns)*np.exp(firing_rate), ns) / ((period-ns)*factorial(ns-1)))
    #pns /= np.sum(pns) # normalize
    
    # for numerical stability, first compute log p(n | n > 0) (unnormalized)
    lnpns = (ns-1) * (np.log(firing_rate) + np.log(period - ns)) 
    lnpns += np.log(firing_rate) 
    lnpns -= firing_rate * (period - ns)
    lnpns[1:] -= np.cumsum(np.log(ns[:-1]))
    
    # then, compute and normalize p(n | n > 0)
    pns = np.exp(lnpns)
    pns /= np.sum(pns)

    # init the list of arrays of firing times
    firing_times = []

    for _ in trange(num_channels, desc="Sampling"):        
        # empty spike train?
        if np.random.binomial(1, np.exp(-firing_rate*(period - 1))):
            firing_times.append(np.array([]))
            continue

        # number of spikes (given n > 0)? (WARNING: possibility of overflow error)
        n = np.random.choice(ns, p=pns)

        # firing times? (conditionned on number of spikes and spikes at location 0 and period)
        tfs = np.zeros(n, dtype=float)
        for m in range(1, n):
            ts = np.arange(tfs[m-1]+1+res, period - (n - m), res)
            pts = np.nan_to_num((n-m)*np.power((period - (n-m) - ts)/(period - (n - m + 1) - tfs[m-1]), n-m-1)/(period - (n - m + 1) - tfs[m-1]))
            pts = pts / pts.sum()
            tfs[m] = np.random.choice(ts, p=pts)
            
        # Ramdomly shift the firing times cyclically
        firing_times.append(np.sort((tfs + np.random.uniform(0, period)) % period))

    return firing_times