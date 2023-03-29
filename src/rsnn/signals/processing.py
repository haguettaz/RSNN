
import numpy as np

from .utils import mod


def local_correlation(firing_times_1, firing_times_2, period, eps=1.0):
    """
    Compute the correlation between two spike trains, the first one being periodic.
    The correlation function being piecing-wise linear, a global maximum is necessarily attained 
    at a shift such that at least one firing time of spike_train_1 is aligned with a firing time of spike_train_2.
    local measure: correlation is computed per channel and every metrics/shifts are returned
    global measure: correlation is computed over all channels and the global metric/shift is returned
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

def global_correlation(firing_times_1, firing_times_2, period, eps=1.0):
    """
    Compute the correlation between two spike trains, the first one being periodic.
    The correlation function being piecing-wise linear, a global maximum is necessarily attained 
    at a shift such that at least one firing time of spike_train_1 is aligned with a firing time of spike_train_2.
    local measure: correlation is computed per channel and every metrics/shifts are returned
    global measure: correlation is computed over all channels and the global metric/shift is returned
    """

    kernel = lambda x_: (np.abs(x_) < eps) * (eps - np.abs(x_)) / eps

    if not isinstance(firing_times_1, list):
        firing_times_1 = [firing_times_1]
    if not isinstance(firing_times_2, list):
        firing_times_2 = [firing_times_2]
        
    assert len(firing_times_1) == len(firing_times_2)
    num_channels = len(firing_times_1)

    # compute all possible correlation maximizers
    shift = np.concatenate([(ft1[None,:] - ft2[:,None]).flatten() for ft1, ft2 in zip(firing_times_1, firing_times_2)])

    # TODO: rewrite global correlation function
    corr = np.zeros(1 + t.size) # 1 for the case where t is empty
    for ft1, ft2 in zip(firing_times_1, firing_times_2):
        if ((ft1.shape[0] == 0) != (ft2.shape[0] == 0)): # exactly one spike train is empty
            continue

        if (ft1.shape[0] == 0) and (ft2.shape[0] == 0): # both spike trains are empty
            corr[1:] += 1
            continue

        t = (ft1[None,:] - ft2[:,None]).flatten()
        corr[1:] += kernel((shift[None,:] - t[:,None]) % period).sum(axis=0) / max(ft1.shape[0], ft2.shape[0])
    
    argmax = np.argmax(corr)
    if argmax == 0:
        return np.array(0.0), np.array(0.0)

    return corr[argmax] / num_channels, mod(shift[argmax-1], period, -period/2)