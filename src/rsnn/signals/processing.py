
import numpy as np


def correlation(firing_times_1, firing_times_2, period, eps=1.0, local=False):
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

    if local:
        corr, shift = np.zeros(num_channels), np.zeros(num_channels)
        for c in range(num_channels):
            num_firing_1 = firing_times_1[c].size
            num_firing_2 = firing_times_2[c].size
            
            if ((num_firing_1 == 0) and (num_firing_2 > 0)) or ((num_firing_1 > 0) and (num_firing_2 == 0)):
                corr[c] = 0
                shift[c] = 0
                continue

            if (num_firing_1 == 0) and (num_firing_2 == 0):
                corr[c] = 1
                shift[c] = 0
                continue

            diff = (firing_times_1[c][None,:] - firing_times_2[c][:,None]).flatten()
            arr = kernel((diff[None,:] - diff[:,None]) % period).sum(axis=1)
            argmax = np.argmax(arr)
            corr[c] = arr[argmax] / max(num_firing_1, num_firing_2)
            shift[c] = (diff[argmax] + period/2) % period - period/2
        return corr, shift

    diff = np.concatenate([(firing_times_1[c][None,:] - firing_times_2[c][:,None]).flatten() for c in range(num_channels)])

    arr = np.zeros(1 + diff.size)
    for c in range(num_channels):
        num_firing_1 = firing_times_1[c].size
        num_firing_2 = firing_times_2[c].size
        
        if ((num_firing_1 == 0) and (num_firing_2 > 0)) or ((num_firing_1 > 0) and (num_firing_2 == 0)):
            continue

        if (num_firing_1 == 0) and (num_firing_2 == 0):
            arr[1:] += 1
            continue

        tmp = (firing_times_1[c][None,:] - firing_times_2[c][:,None]).flatten()
        arr[1:] += kernel((diff[None,:] - tmp[:,None]) % period).sum(axis=0) / max(num_firing_1, num_firing_2)
    
    argmax = np.argmax(arr)
    corr = arr[argmax] / num_channels
    if argmax == 0:
        shift = np.array(0)
    else:
        shift = (diff[argmax-1] + period/2) % period - period/2
    return corr, shift