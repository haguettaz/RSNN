import math
import os
import pickle
import random
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import numpy as np
from tqdm.autonotebook import tqdm

from ..spike_train.periodic_spike_train import MultiChannelPeriodicSpikeTrain
from .neuron import Neuron, Refractory


class Network:
    """
    Network class for Network.
    """

    def __init__(
        self,
        num_neurons: int,
        firing_threshold: float,
        soma_decay: float,
        # num_synapses: int,
        # synapse_decay: float,
        # synapse_delay_lim: Tuple[float, float],
    ):
        """
        Args:
            num_neurons (int): the number of neurons.
            firing_threshold (optional, float): the firing threshold in [theta]. Defaults to None.
            soma_decay (optional, float): the somatic impulse response decay in [ms]. Defaults to None.
            num_synapses (int): the number of (input) synapses per neuron.
            synapse_decay (float): the synaptic impulse response decay in [ms].
            synapse_delay_lim (Tuple[float, float]): the synaptic delay range in [ms].
        """
        if num_neurons < 0:
            raise ValueError("The number of neurons must be non-negative.")
        self.num_neurons = num_neurons
        self.neurons = [Neuron(idx, firing_threshold, soma_decay) for idx in range(num_neurons)]

        # if num_synapses < 0:
        #     raise ValueError("The number of synapses must be non-negative.")

        # for neuron in self.neurons:
        #     neuron.synapses = [
        #         Synapse(
        #             idx, random.choice(self.neurons), random.uniform(*synapse_delay_lim), 0, soma_decay, synapse_decay
        #         )
        #         for idx in range(num_synapses)
        #     ]

    # def random_connect(
    #     self,
    #     num_synapses: int,
    #     soma_decay: float,
    #     synapse_decay: float,
    #     synapse_delay_lim: Tuple[float, float],
    # ):
    #     """
    #     Randomly connect network neurons.

    #     Args:
    #         num_synapses (int): the number of synapses per neuron.
    #         soma_decay (float): the somatic impulse response decay in [ms].
    #         synapse_decay (float): the synaptic impulse response decay in [ms].
    #         synapse_delay_lim (Tuple[float, float]): the synaptic delay range in [ms].
    #     """
    #     # self.num_synapses = num_synapses
    #     # self.soma_decay = soma_decay
    #     # self.synapse_decay = synapse_decay
    #     # self.synapse_delay_lim = synapse_delay_lim

    #     for neuron in tqdm(self.neurons, desc="Create connections between neurons"):
    #         neuron.synapses = [
    #             Synapse(
    #                 idx, random.choice(self.neurons), random.uniform(*synapse_delay_lim), 0, soma_decay, synapse_decay
    #             )
    #             for idx in range(num_synapses)
    #         ]

    def reset(self):
        """
        Reset all neurons.
        """
        for neuron in self.neurons:
            neuron.reset()

    def run(self, t0: float, duration: float, dt: float, std_theta: float, autonomous_indices: Iterable[int]):
        """
        Run the network.

        Args:
            t0 (float): the start time in [ms].
            duration (float): the duration in [ms].
            dt (float): the time step in [ms].
            std_theta (float): the standard deviation of the firing threshold noise in [theta].
            autonomous_indices (Iterable[int]): the indices of autonomous neurons, i.e., neurons with no inputs.
        """
        free_neurons = [self.neurons[i] for i in autonomous_indices]

        for neuron in free_neurons:
            neuron.noisy_firing_threshold = neuron.firing_threshold + random.gauss(0, std_theta)

        for t in tqdm(np.arange(t0, t0 + duration, dt), desc="Network simulation"):
            for neuron in free_neurons:
                neuron.step(t, dt, std_theta)

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
        # discretization:Optional[int]=None,
    ):
        """
        Memorize the spike trains, i.e., solve the corresponding otimization problem for every neuron.

        Args:
            multi_channel_periodic_spike_trains (Union[PeriodicSpikeTrain, Iterable[PeriodicSpikeTrain]]): the (periodic) spike trains.
            synapse_weight_lim (Tuple[float, float]): the synaptic weight range in [theta].
            refractory_weight_lim (Tuple[float, float]): the refractory weight range in [theta].
            max_level (float, optional): the maximum level at rest in [theta]. Defaults to 0.0.
            min_slope (float, optional): the minimum slope around a spike in [theta / ms]. Defaults to 0.0.
            firing_surrounding (float, optional): the surrounding of a spike in [ms]. Defaults to 1.0.
            sampling_rate (float, optional): the sampling rate in [kHz]. Defaults to 5.0.
            discretization (Optional[int], optional): the weight discretization level. Defaults to None.

        Returns:
            List[Dict]: optimization summary dictionnaries.
        """
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
                    # discretization,
                )
            )
        return summmaries

    def load_from_file(self, filename: str):
        """
        Load the network from a file.

        Args:
            filename (str): the name of the file to load from.

        Raises:
            FileNotFoundError: if the file does not exist
            ValueError: if number of neurons does not match
            ValueError: if error loading the file
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
        """
        Save the network configuration to a file.

        Args:
            filename (str): the name of the file to save to.
        """
        config = [neuron.config for neuron in self.neurons]

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the configuration to a file
        try:
            with open(filename, "wb") as f:
                pickle.dump(config, f)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error saving network configuration: {e}")


class Synapse:
    """
    Synapse class for Network.
    """

    def __init__(self, idx: int, source: Neuron, delay: float, weight: float, soma_decay: float, synapse_decay: float):
        """
        Initialize a synapse.

        Args:
            idx (int): the synapse index.
            source (Neuron): the source neuron.
            delay (float): the synaptic delay in [ms].
            weight (float): the synaptic weight.
            soma_decay (float): the somatic impulse response decay in [ms].
            synapse_decay (float): the synaptic impulse response decay in [ms].

        Raises:
            ValueError: if delay is non-positive.
            ValueError: if soma_decay is non-positive.
            ValueError: if synapse_decay is non-positive.
        """
        if delay <= 0:
            raise ValueError("Delay must be non-negative.")
        if soma_decay <= 0:
            raise ValueError("Somatic decay must be positive.")
        if synapse_decay <= 0:
            raise ValueError("Synaptic decay must be positive.")

        self.idx = idx
        self.source = source
        self.weight = weight
        self.delay = delay
        self.soma_decay = soma_decay
        self.synapse_decay = synapse_decay

        self.init_responses()

    def init_responses(self):
        """
        Initializes the response functions.
        """

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
        """
        Returns:
            (Dict[str, Any]): the synapse configuration.
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
        """
        Returns the information flow through the synapse at time t.

        Args:
            t (float): the time in [ms].

        Returns:
            (float): the information flow.
        """
        return self.weight * np.sum(self.response(t - self.delay - self.source.spike_train.firing_times))

    def information_flow_deriv(self, t: float):
        """
        Returns the information flow derivative through the synapse at time t.

        Args:
            t (float): the time in [ms].

        Returns:
            (float): the information flow derivative.
        """
        return self.weight * np.sum(self.response_deriv(t - self.delay - self.source.spike_train.firing_times))
