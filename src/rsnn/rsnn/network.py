import math
import os
import pickle
import random
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from tqdm.autonotebook import tqdm

from ..signals.spike_train import PeriodicSpikeTrain
from .input import Input
from .neuron import Neuron


class Network:
    """
    Network class for RSNN.
    """
    def __init__(self, num_neurons: int, firing_threshold: float):
        """
        Args:
            num_neurons (int): the number of neurons.
            firing_threshold (float): the firing threshold in [theta].
        """
        self.num_neurons = num_neurons
        self.neurons = [Neuron(idx, firing_threshold) for idx in range(self.num_neurons)]
        self.firing_times = np.array([])
        self.firing_threshold = firing_threshold

    def connect_at_random(
        self,
        num_synapses: int,
        soma_decay: float,
        synapse_decay: float,
        synapse_delay_lim: Tuple[float, float],
    ):
        """
        Randomly connect network neurons.

        Args:
            num_synapses (int): the number of synapses per neuron.
            soma_decay (float): the somatic impulse response decay in [ms].
            synapse_decay (float): the synaptic impulse response decay in [ms].
            synapse_delay_lim (Tuple[float, float]): the synaptic delay range in [ms].
        """
        self.num_synapses = num_synapses
        self.soma_decay = soma_decay
        self.synapse_decay = synapse_decay
        self.synapse_delay_lim = synapse_delay_lim

        if soma_decay == synapse_decay:
            synapse_resp = lambda t_: (t_ > 0) * t_ / soma_decay * np.exp(1 - t_ / self.soma_decay)
            synapse_resp_deriv = lambda t_: (t_ > 0) * np.exp(1 - t_ / soma_decay) * (1 - t_ / soma_decay) / soma_decay
        else:
            tmax = (math.log(synapse_decay) - math.log(soma_decay)) / (1 / soma_decay - 1 / synapse_decay)
            gamma = 1 / (math.exp(-tmax / soma_decay) - math.exp(-tmax / synapse_decay))
            synapse_resp = lambda t_: (t_ > 0) * gamma * (np.exp(-t_ / soma_decay) - np.exp(-t_ / synapse_decay))
            synapse_resp_deriv = (
                lambda t_: (t_ > 0)
                * gamma
                * (np.exp(-t_ / synapse_decay) / synapse_decay - np.exp(-t_ / soma_decay) / soma_decay)
            )

        refractory_resp = lambda t_: (t_ > 0) * -np.exp(-t_ / soma_decay)
        refractory_resp_deriv = lambda t_: (t_ > 0) * np.exp(-t_ / soma_decay) / soma_decay

        for neuron in tqdm(self.neurons, desc="connect neurons"):
            neuron.inputs = [
                Input(
                    random.choice(self.neurons),
                    random.uniform(*synapse_delay_lim),
                    0,
                    synapse_resp,
                    synapse_resp_deriv,
                )
                for _ in range(num_synapses)
            ]
            neuron.inputs.append(
                Input(
                    neuron,
                    0,
                    0,
                    refractory_resp,
                    refractory_resp_deriv,
                )
            )

    def reset(self):
        """
        Reset all firing times.
        """
        for neuron in self.neurons:
            neuron.reset()

    def run(self, t0:float, duration:float, dt:float, std_theta:float, autonomous_indices:Iterable[int]):
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

        for t in tqdm(np.arange(t0, t0 + duration, dt), desc="network simulation"):
            for neuron in free_neurons:
                neuron.step(t, dt, std_theta)

    def memorize(
        self,
        spike_trains:Union[PeriodicSpikeTrain, Iterable[PeriodicSpikeTrain]],
        synapse_weight_lim:Tuple[float, float],
        refractory_weight_lim:Tuple[float, float],
        max_level:float=0.0,
        min_slope:float=0.0,
        firing_surrounding:float=1.0,
        sampling_rate:float=5.0,
        discretization:Optional[int]=None,
    ):
        """
        Memorize the spike trains, i.e., solve the corresponding otimization problem for every neuron.

        Args:
            spike_trains (Union[PeriodicSpikeTrain, Iterable[PeriodicSpikeTrain]]): the (periodic) spike trains.
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
        for neuron in tqdm(self.neurons, desc="network optimization"):
            summmaries.append(
                neuron.memorize(
                    spike_trains,
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

    def load(self, dirname:str):
        """
        Load the network from a directory.

        Args:
            dirname (str): directory to load from

        Raises:
            ValueError: if number of neurons does not match
        """
        with open(os.path.join(dirname, "network.pkl"), "rb") as f:
            config = pickle.load(f)

        if self.num_neurons != config["num_neurons"]:
            raise ValueError(f"number of neuron does not match")

        self.firing_threshold = config["firing_threshold"]
        self.num_synapses = config["num_synapses"]
        self.soma_decay = config["soma_decay"]
        self.synapse_decay = config["synapse_decay"]

        if self.soma_decay == self.synapse_decay:
            synapse_resp = lambda t_: (t_ > 0) * t_ / self.soma_decay * np.exp(1 - t_ / self.soma_decay)
            synapse_resp_deriv = (
                lambda t_: (t_ > 0) * np.exp(1 - t_ / self.soma_decay) * (1 - t_ / self.soma_decay) / self.soma_decay
            )
        else:
            tmax = (math.log(self.synapse_decay) - math.log(self.soma_decay)) / (
                1 / self.soma_decay - 1 / self.synapse_decay
            )
            gamma = 1 / (math.exp(-tmax / self.soma_decay) - math.exp(-tmax / self.synapse_decay))
            synapse_resp = (
                lambda t_: (t_ > 0) * gamma * (np.exp(-t_ / self.soma_decay) - np.exp(-t_ / self.synapse_decay))
            )
            synapse_resp_deriv = (
                lambda t_: (t_ > 0)
                * gamma
                * (
                    np.exp(-t_ / self.synapse_decay) / self.synapse_decay
                    - np.exp(-t_ / self.soma_decay) / self.soma_decay
                )
            )

        refractory_resp = lambda t_: (t_ > 0) * -np.exp(-t_ / self.soma_decay)
        refractory_resp_deriv = lambda t_: (t_ > 0) * np.exp(-t_ / self.soma_decay) / self.soma_decay

        for neuron in self.neurons:
            neuron.inputs = [
                Input(self.neurons[source_id], delay, weight, synapse_resp, synapse_resp_deriv)
                for source_id, delay, weight in zip(
                    config["sources_ids"][neuron.idx], config["delays"][neuron.idx], config["weights"][neuron.idx]
                )
            ]
            neuron.inputs[-1].resp = refractory_resp
            neuron.inputs[-1].resp_deriv = refractory_resp_deriv
            neuron.firing_times = np.array(config["firing_times"][neuron.idx])

    def save(self, dirname:str):
        """
        Save the network to a directory.

        Args:
            dirname (str): the directory to save to
        """
        os.makedirs(dirname, exist_ok=True)

        # save network config as dict
        config = {
            "num_neurons": self.num_neurons,
            "firing_threshold": self.firing_threshold,
            "num_synapses": self.num_synapses,
            "soma_decay": self.soma_decay,
            "synapse_decay": self.synapse_decay,
            "sources_ids": [[input.source.idx for input in neuron.inputs] for neuron in self.neurons],
            "delays": [[input.delay for input in neuron.inputs] for neuron in self.neurons],
            "weights": [[input.weight for input in neuron.inputs] for neuron in self.neurons],
            "firing_times": [neuron.firing_times.tolist() for neuron in self.neurons],
        }
        with open(os.path.join(dirname, "network.pkl"), "wb") as f:
            pickle.dump(config, f)
