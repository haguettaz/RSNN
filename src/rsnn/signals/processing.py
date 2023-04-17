
import numpy as np

from .utils import mod


def local_correlation(firing_times_1, firing_times_2, period, eps=1.0):
    """
    Compute the correlation between two spike trains, the first one being periodic.
    The correlation function being piecing-wise linear, a global maximum is necessarily attained 
    at a lag such that at least one firing time of spike_train_1 is aligned with a firing time of spike_train_2.
    """

    kernel = lambda x_: (np.abs(x_) < eps) * (eps - np.abs(x_)) / eps

    if not isinstance(firing_times_1, list):
        firing_times_1 = [firing_times_1]
    if not isinstance(firing_times_2, list):
        firing_times_2 = [firing_times_2]
        
    assert len(firing_times_1) == len(firing_times_2)

    max_corr, max_shift = [], []
    for ft1, ft2 in zip(firing_times_1, firing_times_2):        
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

def global_correlation(ref_firing_times, sim_firing_times, period, eps=1.0):
    """
    Compute the correlation between two spike trains, the first one being periodic.
    The correlation function being piecing-wise linear, a global maximum is necessarily attained 
    at a shift such that at least one firing time of spike_train_1 is aligned with a firing time of spike_train_2.
    """

    kernel = lambda x_: (np.abs(x_) < eps) * (eps - np.abs(x_)) / eps

    if not isinstance(ref_firing_times, list):
        ref_firing_times = [ref_firing_times]
    if not isinstance(sim_firing_times, list):
        sim_firing_times = [sim_firing_times]
        
    assert len(ref_firing_times) == len(sim_firing_times)
    num_channels = len(ref_firing_times)

    # compute all possible correlation maximizers
    lags = np.concatenate([(sim_ft[None,:] - ref_ft[:,None]).flatten() for ref_ft, sim_ft in zip(ref_firing_times, sim_firing_times)])

    if len(lags) == 0:
        return np.zeros(1), np.nan
    
    corr = np.zeros(lags.size)
    for ref_ft, sim_ft in zip(ref_firing_times, sim_firing_times):
        if ((ref_ft.shape[0] == 0) != (sim_ft.shape[0] == 0)): # exactly one spike train is empty
            continue

        if (ref_ft.shape[0] == 0) and (sim_ft.shape[0] == 0): # both spike trains are empty
            corr += 1
            continue

        tmp = (sim_ft[None,:] - ref_ft[:,None]).flatten()
        corr += kernel(mod(lags[None,:] - tmp[:,None], period, -period/2)).sum(axis=0) / max(ref_ft.shape[0], sim_ft.shape[0])
    
    argmax = np.argmax(corr)
    return corr[argmax] / num_channels, mod(lags[argmax], period, -period/2)