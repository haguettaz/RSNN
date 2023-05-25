import random
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

from ..optim.optim import compute_bounded_discrete_weights, compute_bounded_weights
from ..spike_train.periodic_spike_train import MultiChannelPeriodicSpikeTrain
from ..spike_train.spike_train import SpikeTrain

# from ..signals.utils import sphere_intersection, sphere_intersection_complement


class Neuron:
    def __init__(self, idx: int, firing_threshold: float, soma_decay: float):
        """
        Args:
            idx (int): the neuron index.
            firing_threshold (float): the firing threshold in [theta].
        """
        self.idx = idx
        self.firing_threshold = firing_threshold
        self.synapses = []
        self.refractory = Refractory(self, 0, soma_decay)
        self.spike_train = SpikeTrain()

    @property
    def num_synapses(self) -> int:
        """
        Returns:
            (int): the number of synapses.
        """
        return len(self.synapses)

    @property
    def config(self) -> Dict[str, Any]:
        """
        Returns:
            (Dict[str, Any]): the neuron configuration.
        """
        return {
            "idx": self.idx,
            "firing_threshold": self.firing_threshold,
            "synapses": [synapse.config for synapse in self.synapses],
            "refractory": self.refractory.config if self.refractory is not None else None,
            "firing_times": self.spike_train.firing_times,
        }

    def potential(self, t: float):
        """
        Returns the instantaneous neuron potential at time t.

        Args:
            t (float): the time in [ms].

        Returns:
            (float): the neuron potential in [theta].
        """
        return sum([synapse.information_flow(t) for synapse in self.synapses]) + self.refractory.information_flow(t)

    def potential_deriv(self, t: float):
        """
        Returns the instantaneous neuron potential derivative at time t.

        Args:
            t (float): the time in [ms].

        Returns:
            (float): the neuron potential derivative in [theta/ms].
        """
        return sum(
            [synapse.information_flow_deriv(t) for synapse in self.synapses]
        ) + self.refractory.information_flow_deriv(t)

    def reset(self):
        """
        Reset the neuron firing times.
        """
        self.spike_train.reset()

    def step(self, t: float, dt: float, std_theta: float, tol: float = 1e-6):
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
            self.spike_train.append(ft)
            # firing_times = np.append(self.firing_times, ft)
            self.noisy_firing_threshold = self.firing_threshold + random.gauss(0, std_theta)

    def memorize(
        self,
        multi_channel_periodic_spike_trains: Union[
            MultiChannelPeriodicSpikeTrain, Iterable[MultiChannelPeriodicSpikeTrain]
        ],
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
            multi_channel_periodic_spike_trains (Union[MultiChannelPeriodicSpikeTrain, Iterable[MultiChannelPeriodicSpikeTrain]]): the multi-channel periodic spike train(s).
            synapse_weight_lim (Tuple[float, float]): the synaptic weight range in [theta].
            refractory_weight_lim (Tuple[float, float]): the refractory weight range in [theta].
            max_level (float, optional): the maximum level at rest in [theta]. Defaults to 0.0.
            min_slope (float, optional): the minimum slope around a spike in [theta / ms]. Defaults to 0.0.
            firing_surrounding (float, optional): the surrounding of a spike in [ms]. Defaults to 1.0.
            sampling_rate (float): the sampling rate in [kHz]. Defaults to 5.0.
            discretization (int or None): the weight discretization level. Defaults to None.

        Raises:
            ValueError: if the maximum level at rest is greater than the firing threshold.
            ValueError: if the minimum slope around firing is negative.
            ValueError: if the sampling rate is greater than the firing surrounding.

        Returns:
            (Dict): optimization summary dictionnary.
        """
        time_step = 1 / sampling_rate
        if max_level >= self.firing_threshold:
            raise ValueError("max_level must be lower than the firing threshold")
        if min_slope < 0:
            raise ValueError("min_slope must be non negative")
        if time_step > firing_region:
            raise ValueError("time_step must be smaller than firing_region")

        if not isinstance(multi_channel_periodic_spike_trains, Iterable):
            multi_channel_periodic_spike_trains = [multi_channel_periodic_spike_trains]

        w_min = np.array([synapse_weight_lim[0]] * self.num_synapses + [refractory_weight_lim[0]])
        w_max = np.array([synapse_weight_lim[1]] * self.num_synapses + [refractory_weight_lim[1]])

        zt_min = []
        zt_max = []
        yt = []

        for multi_channel_periodic_spike_train in multi_channel_periodic_spike_trains:
            for t in multi_channel_periodic_spike_train.spike_trains[self.idx].firing_times:
                zt_min.append(self.firing_threshold)
                zt_max.append(self.firing_threshold)
                yt.append(
                    np.array(
                        [
                            np.sum(
                                synapse.response(
                                    (
                                        t
                                        - multi_channel_periodic_spike_train.spike_trains[
                                            synapse.source.idx
                                        ].firing_times
                                        - synapse.delay
                                    )
                                    % multi_channel_periodic_spike_train.period
                                )
                            )
                            for synapse in self.synapses
                        ]
                        + [
                            np.sum(
                                self.refractory.response(
                                    (
                                        t
                                        - multi_channel_periodic_spike_train.spike_trains[
                                            self.refractory.source.idx
                                        ].firing_times
                                    )
                                    % multi_channel_periodic_spike_train.period
                                )
                            )
                        ]
                    )
                )

            for t in np.arange(0, multi_channel_periodic_spike_train.period, time_step):
                if np.any(
                    np.minimum(
                        np.abs(t - multi_channel_periodic_spike_train.spike_trains[self.idx].firing_times),
                        np.abs(
                            multi_channel_periodic_spike_train.period
                            - np.abs(t - multi_channel_periodic_spike_train.spike_trains[self.idx].firing_times)
                        ),
                    )
                    < firing_region
                ):
                    zt_min.append(min_slope)
                    zt_max.append(np.inf)
                    yt.append(
                        np.array(
                            [
                                np.sum(
                                    synapse.response_deriv(
                                        (
                                            t
                                            - multi_channel_periodic_spike_train.spike_trains[
                                                synapse.source.idx
                                            ].firing_times
                                            - synapse.delay
                                        )
                                        % multi_channel_periodic_spike_train.period
                                    )
                                )
                                for synapse in self.synapses
                            ]
                            + [0]
                        )
                    )
                else:
                    zt_min.append(-np.inf)
                    zt_max.append(max_level)
                    yt.append(
                        np.array(
                            [
                                np.sum(
                                    synapse.response(
                                        (
                                            t
                                            - multi_channel_periodic_spike_train.spike_trains[
                                                synapse.source.idx
                                            ].firing_times
                                            - synapse.delay
                                        )
                                        % multi_channel_periodic_spike_train.period
                                    )
                                )
                                for synapse in self.synapses
                            ]
                            + [
                                np.sum(
                                    self.refractory.response(
                                        (
                                            t
                                            - multi_channel_periodic_spike_train.spike_trains[
                                                self.refractory.source.idx
                                            ].firing_times
                                        )
                                        % multi_channel_periodic_spike_train.period
                                    )
                                )
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

        for i, synapse in enumerate(self.synapses):
            synapse.weight = weights[i]
        self.refractory.weight = weights[-1]

        return summary


class Refractory:
    """
    Refractory class for RSNN.
    """

    def __init__(self, source: Neuron, weight: float, soma_decay: float):
        """
        Initialize a refractory.

        Args:
            source (Neuron): the source neuron.
            weight (float): the synaptic weight.
            soma_decay (float): the somatic impulse response decay in [ms].

        Raises:
            ValueError: if weight is non-negative.
            ValueError: if soma_decay is non-negative.
        """
        if weight < 0:
            raise ValueError("Weight must be non-negative.")
        self.weight = weight

        if soma_decay < 0:
            raise ValueError("Somatic decay must be non-negative.")
        self.soma_decay = soma_decay

        self.source = source

        self.init_responses()

    def init_responses(self):
        """
        Initializes the response functions.
        """

        self.response = lambda t_: (t_ > 0) * -np.exp(-t_ / self.soma_decay)
        self.response_deriv = lambda t_: (t_ > 0) * np.exp(-t_ / self.soma_decay) / self.soma_decay
                    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Returns:
            (Dict[str, Any]): the refractory configuration.
        """
        return {
            "source": self.source.idx,
            "weight": self.weight,
            "soma_decay": self.soma_decay,
        }

    def information_flow(self, t: float):
        """
        Returns the information flow through the refractory at time t.

        Args:
            t (float): the time in [ms].

        Returns:
            (float): the information flow.
        """
        return self.weight * np.sum(self.response(t - self.source.spike_train.firing_times))

    def information_flow_deriv(self, t: float):
        """
        Returns the information flow derivative through the refractory at time t.

        Args:
            t (float): the time in [ms].

        Returns:
            (float): the information flow derivative.
        """
        return self.weight * np.sum(self.response_deriv(t - self.source.spike_train.firing_times))
