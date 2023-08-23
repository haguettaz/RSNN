import numpy as np

from ..rsnn.filters import input_spike_resp, input_spike_resp_deriv, refractory_spike_resp
from ..utils.utils import cyclic_after, cyclic_neighborhood


def compute_template(
    input_spike_trains, target_spike_train, firing_threshold, max_level, firing_region, min_slope, time_step
):
    yf, ys, yl = [], [], []
    zf, zs, zl = [], [], []

    # equality at firing times
    for t in target_spike_train.firing_times:
        yf.append(
            np.array(
                [
                    np.sum(input_spike_resp((t - spike_train.firing_times) % input_spike_trains.period))
                    for spike_train in input_spike_trains.spike_trains
                ]
            )
        )
        zf.append(
            firing_threshold
            - np.sum(refractory_spike_resp((t - target_spike_train.firing_times) % target_spike_train.period))
        )

    # smaller than the firing threshold everywhere (except during the absolute refractory period)
    for t in cyclic_after(
        target_spike_train.firing_times,
        target_spike_train.period,
        target_spike_train.abs_refractory_period,
        time_step,
        True,
    ):
        yl.append(
            np.array(
                [
                    np.sum(input_spike_resp((t - spike_train.firing_times) % input_spike_trains.period))
                    for spike_train in input_spike_trains.spike_trains
                ]
            )
        )
        zl.append(
            firing_threshold
            - np.sum(refractory_spike_resp((t - target_spike_train.firing_times) % target_spike_train.period))
        )

    # smaller than the maximum level not close to firing (except during the absolute refractory period)
    for t in cyclic_neighborhood(
        target_spike_train.firing_times,
        target_spike_train.period,
        firing_region,
        target_spike_train.abs_refractory_period,
        time_step,
        True,
    ):
        yl.append(
            np.array(
                [
                    np.sum(input_spike_resp((t - spike_train.firing_times) % input_spike_trains.period))
                    for spike_train in input_spike_trains.spike_trains
                ]
            )
        )
        zl.append(
            max_level
            - np.sum(refractory_spike_resp((t - target_spike_train.firing_times) % target_spike_train.period))
        )

    # slope larger than the minimum slope close to firing
    for t in cyclic_neighborhood(
        target_spike_train.firing_times, target_spike_train.period, firing_region, firing_region, time_step
    ):
        ys.append(
            np.array(
                [
                    np.sum(input_spike_resp_deriv((t - spike_train.firing_times) % input_spike_trains.period))
                    for spike_train in input_spike_trains.spike_trains
                ]
            )
        )
        zs.append(min_slope)

    return np.vstack(yf), np.array(zf), np.vstack(yl), np.array(zl), np.vstack(ys), np.array(zs)
