import random
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from ..optim.optim import compute_bounded_discrete_weights, compute_bounded_weights
from ..spike_train.periodic_spike_train import MultiChannelPeriodicSpikeTrain
from ..spike_train.spike_train import SpikeTrain


class Neuron:
    """A class representing a spiking neuron.

    Attributes:
        idx (int): The neuron unique index.
        firing_threshold (float): The neuron firing threshold in [theta].
        synapses (List[Synapse]): The neuron list of incoming synapses.
        refractory (Refractory): The neuron refractory.
        spike_train (SpikeTrain): The neuron spike train.

    Methods:
        reset: Reset the spike train of the neuron.
        update: Simulate the network on a given time range.
        memorize: Memorize the given spike train.
        potential: Compute the neuron potential.
        potential_deriv: Compute the neuron potential derivative.

    Properties:
        num_synapses (int): The number of incoming synapses.
        config (Dict[str, Any]): The neuron configuration dictionnary.
    """

    def __init__(self, idx: int, firing_threshold: float, soma_decay: float):
        """
        Args:
            idx (int): The neuron index.
            firing_threshold (float): The firing threshold in [theta].
        """
        self.idx = idx
        self.firing_threshold = firing_threshold
        self.synapses = []
        self.refractory = Refractory(self, 0, soma_decay)
        self.spike_train = SpikeTrain()

    @property
    def num_synapses(self) -> int:
        """Get the number of incoming synapses of the neuron.

        Returns:
            int: The number of synapses.
        """
        return len(self.synapses)

    @property
    def config(self) -> Dict[str, Any]:
        """Get the neuron configuration.

        Returns:
            Dict[str, Any]: The neuron configuration.
        """
        return {
            "idx": self.idx,
            "firing_threshold": self.firing_threshold,
            "synapses": [synapse.config for synapse in self.synapses],
            "refractory": self.refractory.config,
            "firing_times": self.spike_train.firing_times,
        }

    def potential(self, t: float) -> np.ndarray:
        """
        Returns the instantaneous neuron potential at time t.

        Args:
            t (float): The time in [ms].

        Returns:
            np.ndarray[float]: The neuron potential in [theta].
        """
        return np.sum([synapse.information_flow(t) for synapse in self.synapses]) + self.refractory.information_flow(t)

    def potential_deriv(self, t: float) -> np.ndarray:
        """
        Returns the instantaneous neuron potential derivative at time t.

        Args:
            t (float): The time in [ms].

        Returns:
            np.ndarray[float]: The neuron potential derivative in [theta/ms].
        """
        return np.sum(
            [synapse.information_flow_deriv(t) for synapse in self.synapses]
        ) + self.refractory.information_flow_deriv(t)

    def reset(self):
        """Reset the neuron spike train."""
        self.spike_train.reset()

    def update(self, t: float, dt: float, std_theta: float, tol: float = 1e-6):
        """Update the neuron state.

        Args:
            t (float): The current time in [ms].
            dt (float): The time step in [ms].
            std_theta (float): The standard deviation of the firing threshold in [theta].
            tol (float, optional): The time tolerance in [ms]. Defaults to 1e-6.
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
        multi_channel_periodic_spike_trains: Iterable[MultiChannelPeriodicSpikeTrain],
        synapse_weight_lim: Tuple[float, float],
        refractory_weight_lim: Tuple[float, float],
        max_level: float,
        min_slope: float,
        firing_region: float,
        sampling_rate: float,
        discretization: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Memorize the spike trains, i.e., solve the corresponding otimization problem by IRLS.

        Args:
            multi_channel_periodic_spike_trains (Iterable[MultiChannelPeriodicSpikeTrain]): The multi-channel periodic spike train(s).
            synapse_weight_lim (Tuple[float, float]): The synaptic weight range in [theta].
            refractory_weight_lim (Tuple[float, float]): The refractory weight range in [theta].
            max_level (float, optional): The maximum level at rest in [theta]. Defaults to 0.0.
            min_slope (float, optional): The minimum slope around a spike in [theta / ms]. Defaults to 0.0.
            firing_surrounding (float, optional): The surrounding of a spike in [ms]. Defaults to 1.0.
            sampling_rate (float): The sampling rate in [kHz]. Defaults to 5.0.
            discretization (int or None): The weight discretization level. Defaults to None.

        Raises:
            ValueError: If the maximum level at rest is greater than the firing threshold.
            ValueError: If the minimum slope around firing is negative.
            ValueError: If the sampling rate is greater than the firing surrounding.

        Returns:
            (Dict[str, Any]): The optimization summary dictionnary.
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
    """A class representing a refractory effect.

    Attributes:
        source (Neuron): The source neuron.
        weight (float): The synaptic weight.
        soma_decay (float): The somatic impulse response decay in [ms].
        response (Callable): The (unit) refractory response to a unit impulse.
        response_deriv (Callable): The (unit) refractory response derivative to a unit impulse.

    Methods:
        init_responses: Init the refractory response and derivative.
        information_flow: Return the information flow through the refractory loop at a given time.
        information_flow_deriv: Return the information flow derivative through the refractory loop at a given time.

    Properties:
        config (Dict[str, Any]): The refractory configuration dictionnary.
    """

    def __init__(self, source: Neuron, weight: float, soma_decay: float):
        """Initialize a refractory.

        Args:
            source (Neuron): The source neuron.
            weight (float): The synaptic weight.
            soma_decay (float): The somatic impulse response decay in [ms].

        Raises:
            ValueError: If weight is non-negative.
            ValueError: If soma_decay is non-negative.
        """
        self.source = source

        if weight < 0:
            raise ValueError("Weight must be non-negative.")
        self.weight = weight

        if soma_decay < 0:
            raise ValueError("Somatic decay must be non-negative.")
        self.soma_decay = soma_decay

        self.init_responses()

    def init_responses(self):
        """Initialize the response functions."""

        self.response = lambda t_: (t_ > 0) * -np.exp(-t_ / self.soma_decay)
        self.response_deriv = lambda t_: (t_ > 0) * np.exp(-t_ / self.soma_decay) / self.soma_decay

    @property
    def config(self) -> Dict[str, Any]:
        """Get the refractory configuration.

        Returns:
            (Dict[str, Any]): The refractory configuration.
        """
        return {
            "source": self.source.idx,
            "weight": self.weight,
            "soma_decay": self.soma_decay,
        }

    def information_flow(self, t: float) -> np.ndarray:
        """Get the information flow through the refractory loop at time t.

        Args:
            t (float): The time in [ms].

        Returns:
            (np.ndarray[float]): The information flow.
        """
        return self.weight * np.sum(self.response(t - self.source.spike_train.firing_times))

    def information_flow_deriv(self, t: float) -> np.ndarray:
        """Get the information flow derivative through the refractory loop at time t.

        Args:
            t (float): The time in [ms].

        Returns:
            (np.ndarray[float]): The information flow derivative.
        """
        return self.weight * np.sum(self.response_deriv(t - self.source.spike_train.firing_times))
