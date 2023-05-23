
from typing import Iterable

import numpy as np

from .utils import mod


def local_correlation(hat_firing_times, firing_times, period, eps=1.0):
    """
    Compute the local correlation between an ideal (periodic) spike train and a real one.
    Local means the correlation is computed for each channel separately.
    
    Args:
        hat_firing_times (Iterable[np.ndarray]): the firing times of the ideal (period) spike train.
        firing_times (Iterable[np.ndarray]): the firing times of the actual spike train.
        period (float): the period of the ideal spike train.
        eps (float, optional): the half-width of the triangular kernel in [ms]. Defaults to 1.0.

    Returns:
        (float): the maximal correlation between the two spike trains.
        (float): the lag at which the correlation is maximal.
    """

    kernel = lambda x_: (np.abs(x_) < eps) * (eps - np.abs(x_)) / eps

    if not isinstance(hat_firing_times, list):
        hat_firing_times = [hat_firing_times]
    if not isinstance(firing_times, list):
        firing_times = [firing_times]
        
    assert len(hat_firing_times) == len(firing_times)

    max_corr, max_shift = [], []
    for ft1, ft2 in zip(hat_firing_times, firing_times):        
        if ((ft1.shape[0] == 0) != (ft2.shape[0] == 0)): # exactly one spike train is empty
            max_corr.append(0.)
            max_shift.append(0.)
            continue

        if (ft1.shape[0] == 0) and (ft2.shape[0] == 0): # both spike trains are empty
            max_corr.append(1.)
            max_shift.append(0.)
            continue

        t = (ft1[None,:] - ft2[:,None]).flatten()
        corr = kernel((t[None,:] - t[:,None]) % period).sum(axis=1)
        argmax = np.argmax(corr)
        max_corr.append(corr[argmax] / max(ft1.shape[0], ft2.shape[0]))
        max_shift.append(mod(t[argmax], period, -period/2))
        
    return np.array(max_corr), np.array(max_shift)

def global_correlation(hat_firing_times:Iterable[np.ndarray], firing_times:Iterable[np.ndarray], period:float, eps:float=1.0):
    """
    Compute the global correlation between an ideal (periodic) spike train and a real one.

    Args:
        hat_firing_times (Iterable[np.ndarray]): the firing times of the ideal (period) spike train.
        firing_times (Iterable[np.ndarray]): the firing times of the actual spike train.
        period (float): the period of the ideal spike train.
        eps (float, optional): the half-width of the triangular kernel in [ms]. Defaults to 1.0.

    Returns:
        (float): the maximal correlation between the two spike trains.
        (float): the lag at which the correlation is maximal.
    """

    kernel = lambda x_: (np.abs(x_) < eps) * (eps - np.abs(x_)) / eps

    if not isinstance(hat_firing_times, list):
        hat_firing_times = [hat_firing_times]
    if not isinstance(firing_times, list):
        firing_times = [firing_times]
        
    assert len(hat_firing_times) == len(firing_times)
    num_channels = len(hat_firing_times)

    # compute all possible correlation maximizers
    lags = np.concatenate([(sim_ft[None,:] - ref_ft[:,None]).flatten() for ref_ft, sim_ft in zip(hat_firing_times, firing_times)])

    if len(lags) == 0:
        return 0.0, np.nan
    
    corr = np.zeros(lags.size)
    for ref_ft, sim_ft in zip(hat_firing_times, firing_times):
        if ((ref_ft.shape[0] == 0) != (sim_ft.shape[0] == 0)): # exactly one spike train is empty
            continue

        if (ref_ft.shape[0] == 0) and (sim_ft.shape[0] == 0): # both spike trains are empty
            corr += 1
            continue

        tmp = (sim_ft[None,:] - ref_ft[:,None]).flatten()
        corr += kernel(mod(lags[None,:] - tmp[:,None], period, -period/2)).sum(axis=0) / max(ref_ft.shape[0], sim_ft.shape[0])
    
    argmax = np.argmax(corr)
    return corr[argmax] / num_channels, mod(lags[argmax], period, -period/2)