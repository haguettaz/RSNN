import random
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from ..optim.optim import (compute_bounded_discrete_weights,
                           compute_bounded_weights)
from ..signals.spike_train import PeriodicSpikeTrain

# from ..signals.utils import sphere_intersection, sphere_intersection_complement


class Neuron:
    def __init__(self, idx:int, firing_threshold:float):
        """
        Args:
            idx (int): the neuron index.
            firing_threshold (float): the firing threshold in [theta].
        """
        self.idx = idx
        self.firing_threshold = firing_threshold
        self.inputs = []
        self.firing_times = np.array([])

    def potential(self, t:float):
        """
        Returns the instantaneous neuron potential at time t.

        Args:
            t (float): the time in [ms].

        Returns:
            (float): the neuron potential in [theta].
        """
        return sum([input.signal(t) for input in self.inputs])

    def potential_deriv(self, t:float):
        """
        Returns the instantaneous neuron potential derivative at time t.

        Args:
            t (float): the time in [ms].

        Returns:
            (float): the neuron potential derivative in [theta/ms].
        """
        return sum([input.signal_deriv(t) for input in self.inputs])

    def reset(self):
        """
        Reset the neuron firing times.
        """
        self.firing_times = np.array([])

    def step(self, t:float, dt:float, std_theta:float, tol:float=1e-6):
        """
        Advance the neuron state by one time step.

        Args:
            t (float): the current time in [ms].
            dt (float): the time step in [ms].
            std_theta (float): the standard deviation of the firing threshold in [theta].
            tol (float, optional): the time tolerance in [ms]. Defaults to 1e-6.
        """
        z = self.potential(t)
        if z > self.noisy_firing_threshold:
            # find the firing time by bisection
            ft = t - dt / 2
            dt /= 2
            z = self.potential(ft)
            while np.abs(z - self.noisy_firing_threshold) > tol and dt > tol:
                if z > self.noisy_firing_threshold:
                    ft -= dt / 2
                else:
                    ft += dt / 2
                dt /= 2
                z = self.potential(ft)
            self.firing_times = np.append(self.firing_times, ft)
            self.noisy_firing_threshold = self.firing_threshold + random.gauss(0, std_theta)

    def memorize(
        self,
        spike_trains:Union[PeriodicSpikeTrain, Iterable[PeriodicSpikeTrain]],
        synapse_weight_lim: Tuple[float, float],
        refractory_weight_lim: Tuple[float, float],
        max_level: float,
        min_slope: float,
        firing_region: float,
        sampling_rate: float,
        discretization: Optional[int]=None,
    ):
        """
        Memorize the spike trains, i.e., solve the corresponding otimization problem by IRLS.

        Args:
            spike_trains (Union[PeriodicSpikeTrain, Iterable[PeriodicSpikeTrain]]): the (periodic) spike trains.
            synapse_weight_lim (Tuple[float, float]): the synaptic weight range in [theta].
            refractory_weight_lim (Tuple[float, float]): the refractory weight range in [theta].
            max_level (float, optional): the maximum level at rest in [theta]. Defaults to 0.0.
            min_slope (float, optional): the minimum slope around a spike in [theta / ms]. Defaults to 0.0.
            firing_surrounding (float, optional): the surrounding of a spike in [ms]. Defaults to 1.0.
            sampling_rate (float, optional): the sampling rate in [kHz]. Defaults to 5.0.
            discretization (Optional[int], optional): the weight discretization level. Defaults to None.

        Raises:
            ValueError: if the maximum level at rest is greater than the firing threshold.
            ValueError: if the minimum slope around firing is negative.
            ValueError: if the sampling rate is greater than the firing surrounding.

        Returns:
            Dict: optimization summary dictionnary.
        """
        dt = 1 / sampling_rate
        if max_level >= self.firing_threshold:
            raise ValueError("max_level must be lower than the firing threshold")
        if min_slope < 0:
            raise ValueError("min_slope must be non negative")
        if dt > firing_region:
            raise ValueError("dt must be smaller than firing_region")

        num_synapses = len(self.inputs) - 1
        w_min = np.array([synapse_weight_lim[0]] * num_synapses + [refractory_weight_lim[0]])
        w_max = np.array([synapse_weight_lim[1]] * num_synapses + [refractory_weight_lim[1]])

        if isinstance(spike_trains, PeriodicSpikeTrain):
            spike_trains = [spike_trains]

        zt_min = []
        zt_max = []
        yt = []

        for spike_train in spike_trains:
            for t in spike_train.firing_times[self.idx]:
                zt_min.append(self.firing_threshold)
                zt_max.append(self.firing_threshold)
                yt.append(
                    np.array(
                        [
                            np.sum(
                                input.resp(
                                    (t - spike_train.firing_times[input.source.idx] - input.delay)
                                    % spike_train.period
                                )
                            )
                            for input in self.inputs
                        ]
                    )
                )
                # print("firing time:", t, yt[-1][-5:])

            for t in np.arange(0, spike_train.period, dt):
                if np.any(
                    np.minimum(
                        np.abs(t - spike_train.firing_times[self.idx]),
                        np.abs(spike_train.period - np.abs(t - spike_train.firing_times[self.idx])),
                    )
                    < firing_region
                ):
                    zt_min.append(min_slope)
                    zt_max.append(np.inf)
                    yt.append(
                        np.array(
                            [
                                np.sum(
                                    input.resp_deriv(
                                        (t - spike_train.firing_times[input.source.idx] - input.delay)
                                        % spike_train.period
                                    )
                                )
                                for input in self.inputs[:-1]
                            ]
                            + [0]  # refractory weight is excluded from slope constraint
                        )
                    )
                    # print("firing surrounding:", t, yt[-1][-5:])
                else:
                    zt_min.append(-np.inf)
                    zt_max.append(max_level)
                    yt.append(
                        np.array(
                            [
                                np.sum(
                                    input.resp(
                                        (t - spike_train.firing_times[input.source.idx] - input.delay)
                                        % spike_train.period
                                    )
                                )
                                for input in self.inputs
                            ]
                        )
                    )
                    # print("silent time:", t, yt[-1][-5:])

        zt_min = np.array(zt_min)
        zt_max = np.array(zt_max)
        yt = np.vstack(yt)

        if discretization is None:
            weights, summary = compute_bounded_weights(yt, zt_min, zt_max, w_min, w_max)
        else:
            weights, summary = compute_bounded_discrete_weights(yt, zt_min, zt_max, w_min, w_max, discretization)

        for i, input in enumerate(self.inputs):
            input.weight = weights[i]

        return summary
