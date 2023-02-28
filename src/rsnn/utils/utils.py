import numpy as np


def simulation_cycle_generator(firing_times, duration):
    if firing_times.size == 0:
        yield firing_times 

    else:
        t0 = 0
        last_firing_times = np.max(firing_times)
        while (t0 < last_firing_times):
            mask = (firing_times >= t0) & (firing_times < t0 + duration)
            yield firing_times[mask]
            t0 += duration

def correlation(firing_times_1, firing_times_2, duration, kernel=None, tol=1e-6):
    """
    Compute the correlation between two spike trains.
    """
    # reference signal is periodic
    # sim_ftimes and ref_ftimes are supposed to be one period
    if kernel is None:
        kernel = lambda x_: (np.abs(x_) < 1) * (1 - np.abs(x_))

    if ((firing_times_1.size == 0) and (firing_times_2.size > 0)) or (
        (firing_times_1.size == 0) and (firing_times_2.size > 0)
    ):
        return 0.0, np.nan

    if (firing_times_1.size == 0) and (firing_times_2.size == 0):
        return 1.0, 0.0

    tmp_left, tmp_right = 0, duration
    while (tmp_right - tmp_left) > tol:
        tmp_mid = (tmp_left + tmp_right) / 2
        tmp = np.linspace(tmp_left, tmp_right, 1000)
        corr = (
            kernel((firing_times_1[None, None, :] - tmp[:, None, None] - firing_times_2[None, :, None]) % duration)
            .reshape(tmp.shape[0], -1)
            .sum(axis=1)
        )
        shift = tmp[np.argmax(corr)]
        if shift < tmp_mid:
            tmp_right = tmp_mid
        else:
            tmp_left = tmp_mid

    Z = max(firing_times_1.size, firing_times_2.size)
    return np.max(corr) / Z, (shift + duration/2) % duration - duration/2
