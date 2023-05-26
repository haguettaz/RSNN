import math
import os
import pickle
import random
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from tqdm.autonotebook import tqdm

from ..spike_train.periodic_spike_train import MultiChannelPeriodicSpikeTrain
from .neuron import Neuron, Refractory


class Network:
    """A class representing a spiking neural network.

    Attributes:
        neurons (List[Neuron]): The list of neurons in the network.
        num_neurons (int): The number of neurons in the network.

    Methods:
        reset: Reset the spike train of every neuron.
        simulate: Simulate the network on a given time range.
        memorize: Memorize the spike trains, i.e., solve the corresponding optimization problem for every neuron.
        load_from_file: Load the network from a file.
        save_to_file: Save the network configuration to a file.
    """

    def __init__(
        self,
        num_neurons: int,
        firing_threshold: float,
        soma_decay: float,
    ):
        """Initialize a `Network` object with a list of `Neuron` objects.

        Args:
            num_neurons (int): The number of neurons.
            firing_threshold (optional, float): The firing threshold in [theta]. Defaults to None.
            soma_decay (optional, float): The somatic impulse response decay in [ms]. Defaults to None.
            num_synapses (int): The number of (input) synapses per neuron.
            synapse_decay (float): The synaptic impulse response decay in [ms].
            synapse_delay_lim (Tuple[float, float]): The synaptic delay range in [ms].
        """
        if num_neurons < 0:
            raise ValueError("The number of neurons must be non-negative.")
        self.num_neurons = num_neurons
        self.neurons = [Neuron(idx, firing_threshold, soma_decay) for idx in range(num_neurons)]

    def reset(self):
        """Reset all neurons."""
        for neuron in self.neurons:
            neuron.reset()

    def simulate(self, t0: float, duration: float, dt: float, std_theta: float, sim_indices: Iterable[int]):
        """Simulate the network in a given time range.

        Args:
            t0 (float): The starting time of the simulation in [ms].
            duration (float): The duration of the simulation in [ms].
            dt (float): The time step of the simulation in [ms]. Note that the precision on firing time is significantly better.
            std_theta (float): The standard deviation of the firing threshold noise in [theta].
            sim_indices (Iterable[int]): The indices of neurons to simulate. Every other neuron has a fixed spike train.
        """
        free_neurons = [self.neurons[i] for i in sim_indices]

        for neuron in free_neurons:
            neuron.noisy_firing_threshold = neuron.firing_threshold + random.gauss(0, std_theta)

        for t in tqdm(np.arange(t0, t0 + duration, dt), desc="Network simulation"):
            for neuron in free_neurons:
                neuron.update(t, dt, std_theta)

    def memorize(
        self,
        multi_channel_periodic_spike_trains: Union[
            MultiChannelPeriodicSpikeTrain, Iterable[MultiChannelPeriodicSpikeTrain]
        ],
        synapse_weight_lim: Tuple[float, float],
        refractory_weight_lim: Tuple[float, float],
        max_level: float = 0.0,
        min_slope: float = 0.0,
        firing_surrounding: float = 1.0,
        sampling_rate: float = 5.0,
        discretization: Optional[int] = None,
    ):
        """Memorize the prescribed spike trains, by solving the corresponding otimization problem for every neuron.

        Args:
            multi_channel_periodic_spike_trains (Union[PeriodicSpikeTrain, Iterable[PeriodicSpikeTrain]]): The (periodic) spike trains.
            synapse_weight_lim (Tuple[float, float]): The synaptic weight range in [theta].
            refractory_weight_lim (Tuple[float, float]): The refractory weight range in [theta].
            max_level (float, optional): The maximum level at rest in [theta]. Defaults to 0.0.
            min_slope (float, optional): The minimum slope around a spike in [theta / ms]. Defaults to 0.0.
            firing_surrounding (float, optional): The surrounding of a spike in [ms]. Defaults to 1.0.
            sampling_rate (float, optional): The sampling rate in [kHz]. Defaults to 5.0.
            discretization (Optional[int], optional): The weight discretization level. Defaults to None.

        Returns:
            List[Dict]: The optimization summaries.
        """
        if not isinstance(multi_channel_periodic_spike_trains, Iterable):
            multi_channel_periodic_spike_trains = [multi_channel_periodic_spike_trains]

        summmaries = []
        for neuron in tqdm(self.neurons, desc="Network optimization"):
            summmaries.append(
                neuron.memorize(
                    multi_channel_periodic_spike_trains,
                    synapse_weight_lim,
                    refractory_weight_lim,
                    max_level,
                    min_slope,
                    firing_surrounding,
                    sampling_rate,
                    discretization,
                )
            )
        return summmaries

    def load_from_file(self, filename: str):
        """Load the network from a file.

        Args:
            filename (str): The name of the file to load from.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If number of neurons does not match.
            ValueError: If error loading the file.
        """

        # Check if the file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")

        # Load the configuration from the file
        try:
            with open(filename, "rb") as f:
                configs = pickle.load(f)
        except (FileNotFoundError, PermissionError) as e:
            raise ValueError(f"Error loading network configuration: {e}")

        if self.num_neurons != len(configs):
            raise ValueError(f"Number of neurons does not match: {len(configs)} != {self.num_neurons}")

        # Initialize all neurons
        for neuron, config in zip(self.neurons, configs):
            neuron.idx = config["idx"]
            neuron.firing_threshold = config["firing_threshold"]
            neuron.spike_train.firing_times = config["firing_times"]
            neuron.refractory = Refractory(
                self.neurons[config["refractory"]["source"]],
                config["refractory"]["weight"],
                config["refractory"]["soma_decay"],
            )

        # Connect neurons
        for neuron, config in zip(self.neurons, configs):
            neuron.synapses = []
            for synapse_config in config["synapses"]:
                neuron.synapses.append(
                    Synapse(
                        synapse_config["idx"],
                        self.neurons[synapse_config["source"]],
                        synapse_config["delay"],
                        synapse_config["weight"],
                        synapse_config["soma_decay"],
                        synapse_config["synapse_decay"],
                    )
                )

    def save_to_file(self, filename: str):
        """Save the network configuration to a file.

        Args:
            filename (str): The name of the file to save to.

        Raises:
            ValueError: If error saving the network configuration.
        """
        config = [neuron.config for neuron in self.neurons]

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the configuration to a file
        try:
            with open(filename, "wb") as f:
                pickle.dump(config, f)
        except (FileNotFoundError, PermissionError) as e:
            raise ValueError(f"Error saving network configuration: {e}")


class Synapse:
    """A class representing a synapse.

    Attributes:
        idx: The synapse unique index.
        source (Neuron): The source neuron.
        delay (float): The synaptic delay in [ms].
        weight (float): The synaptic weight.
        soma_decay (float): The somatic impulse response decay in [ms].
        synapse_decay (float): The synaptic impulse response decay in [ms].
        response (Callable): The (unit) synaptic response to a unit impulse.
        response_deriv (Callable): The (unit) synaptic response derivative to a unit impulse.

    Methods:
        init_responses: Init the synaptic response and derivative.
        information_flow: Return the information flow through the synapse at a given time.
        information_flow_deriv: Return the information flow derivative through the synapse at a given time.

    Properties:
        config (Dict[str, Any]): The synapse configuration dictionnary.
    """

    def __init__(self, idx: int, source: Neuron, delay: float, weight: float, soma_decay: float, synapse_decay: float):
        """
        Args:
            idx (int): The synapse index.
            source (Neuron): The source neuron.
            delay (float): The synaptic delay in [ms].
            weight (float): The synaptic weight.
            soma_decay (float): The somatic impulse response decay in [ms].
            synapse_decay (float): The synaptic impulse response decay in [ms].

        Raises:
            ValueError: If delay is non-positive.
            ValueError: If soma_decay is non-positive.
            ValueError: If synapse_decay is non-positive.
        """
        self.idx = idx
        self.source = source

        if delay <= 0:
            raise ValueError("Delay must be non-negative.")
        self.delay = delay

        self.weight = weight

        if soma_decay <= 0:
            raise ValueError("Somatic decay must be positive.")
        self.soma_decay = soma_decay

        if synapse_decay <= 0:
            raise ValueError("Synaptic decay must be positive.")
        self.synapse_decay = synapse_decay

        self.init_responses()

    def init_responses(self):
        """Initializes the response functions."""

        if self.soma_decay == self.synapse_decay:
            self.response = lambda t_: (t_ > 0) * t_ / self.soma_decay * np.exp(1 - t_ / self.soma_decay)
            self.response_deriv = (
                lambda t_: (t_ > 0) * np.exp(1 - t_ / self.soma_decay) * (1 - t_ / self.soma_decay) / self.soma_decay
            )
        else:
            tmax = (math.log(self.synapse_decay) - math.log(self.soma_decay)) / (
                1 / self.soma_decay - 1 / self.synapse_decay
            )
            gamma = 1 / (math.exp(-tmax / self.soma_decay) - math.exp(-tmax / self.synapse_decay))
            self.response = (
                lambda t_: (t_ > 0) * gamma * (np.exp(-t_ / self.soma_decay) - np.exp(-t_ / self.synapse_decay))
            )
            self.response_deriv = (
                lambda t_: (t_ > 0)
                * gamma
                * (
                    np.exp(-t_ / self.synapse_decay) / self.synapse_decay
                    - np.exp(-t_ / self.soma_decay) / self.soma_decay
                )
            )

    @property
    def config(self) -> Dict[str, Any]:
        """Get the synapse configuration.

        Returns:
            (Dict[str, Any]): The synapse configuration.
        """
        return {
            "idx": self.idx,
            "source": self.source.idx,
            "weight": self.weight,
            "delay": self.delay,
            "soma_decay": self.soma_decay,
            "synapse_decay": self.synapse_decay,
        }

    def information_flow(self, t: float):
        """Get the information flow through the synapse at time t.

        Args:
            t (float): The time in [ms].

        Returns:
            (ndarray[float]): The information flow.
        """
        return self.weight * np.sum(self.response(t - self.delay - self.source.spike_train.firing_times))

    def information_flow_deriv(self, t: float):
        """Get the information flow derivative through the synapse at time t.

        Args:
            t (float): The time in [ms].

        Returns:
            (nd.arrray[float]): The information flow derivative.
        """
        return self.weight * np.sum(self.response_deriv(t - self.delay - self.source.spike_train.firing_times))
