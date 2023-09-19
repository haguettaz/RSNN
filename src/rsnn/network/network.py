from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import brentq
from tqdm.autonotebook import tqdm

from ..neuron.neuron import Neuron


class Network:
    """A class representing a spiking neural network.

    Attributes:
        neurons (List[Neuron]): The list of neurons in the network.
        num_neurons (int): The number of neurons in the network.

    Methods:
        sim: Simulate the network on a given time range.
    """

    def __init__(self, neurons: Iterable[Neuron]):
        """Initialize a `Network` object with a list of `Neuron` objects.

        Args:
            num_neurons (int): The number of neurons.
            firing_threshold (optional, float): The firing threshold in [theta]. Defaults to None.
            soma_decay (optional, float): The somatic impulse response beta in [ms]. Defaults to None.
            num_synapses (int): The number of (input) synapses per neuron.
            synapse_beta (float): The synaptic impulse response beta in [ms].
            synapse_delay_lim (Tuple[float, float]): The synaptic delay range in [ms].
        """
        self.neurons = neurons
        self.num_neurons = len(neurons)

    def sim(
        self,
        tmax: float,
        dt: float,
        std_threshold: Optional[float]=0.0,
        sim_indices: Optional[Iterable[int]]=None,
        nmax: Optional[int] = 20,
        tol: Optional[float] = 1e-4,
    ):
        """Simulate the network on a given time range.

        Args:
            tmax (float): time range upper bound in [ms].
            dt (float): time step in [ms].
            std_threshold (float, optional): firing threshold nominal value standard deviation. Defaults to 0.0.
            sim_indices (list of int, optional): indices of neurons to simulate. Defaults to None, meaning all neurons.
            nmax (int, optional): maximum number of firings to consider per inputs. Defaults to 20.
            tol (float, optional): tolerance for negligible contribution. Defaults to 1e-4.

        Raises:
            ValueError: if the time step is larger than the minimum delay of any neuron.
            ValueError: if the effect of the (worst) nmax-th latest spike is not negligible.
        """
        if sim_indices is None:
            sim_neurons = self.neurons
        else:
            sim_neurons = [self.neurons[i] for i in sim_indices]

        for neuron in sim_neurons:
            if dt > np.min(neuron.delays):
                raise ValueError("The simulation time step must be smaller than the minimum delay.")

            if neuron.input_kernel(neuron.absolute_refractory * (nmax) - np.max(neuron.delays)) > tol:
                raise ValueError(f"The effect of the {nmax}th latest spike is not negligible (>{tol}).")
            
            neuron.input_firing_times = np.vstack(
                [
                    np.pad(self.neurons[src].firing_times + delay, (nmax - self.neurons[src].firing_times.size, 0), constant_values=np.nan) if self.neurons[src].firing_times.size < nmax
                    else self.neurons[src].firing_times[-nmax:] + delay
                    for (src, delay) in zip(neuron.sources, neuron.delays)
                ]
            )

            neuron.noisy_threshold = np.random.normal(self.nominal_threshold, std_threshold)
            neuron.potential = lambda t_: np.inner(
                neuron.weights, np.nansum(neuron.input_kernel((t_ - neuron.input_firing_times)), axis=-1)
            )
            neuron.adaptive_threshold = lambda t_: neuron.firing_threshold + np.nansum(
                neuron.refractory_kernel((t_ - neuron.firing_times)), axis=-1
            )
            neuron.fun = lambda t_: neuron.potential(t_) - neuron.adaptive_threshold(t_)

        targets = {neuron: [] for neuron in self.neurons}
        for neuron in self.neurons:
            for k, src in enumerate(neuron.sources):
                targets[self.neurons[src]].append((neuron, k))

        t0 = np.max([neuron.firing_times[-1] + neuron.absolute_refractory for neuron in sim_neurons if neuron.firing_times.size], initial=0.0)

        for t in tqdm(np.arange(t0, tmax, dt), desc="Network simulation"):
            for neuron in sim_neurons:
                if neuron.firing_times[-1] + neuron.absolute_refractory > t:
                    continue

                if neuron.fun(t) > 0:
                    # determine the exact firing times
                    ft = brentq(neuron.fun, t - dt, t)
                    neuron.firing_times = np.append(self.firing_times, ft)

                    # update the input firing times of all neurons targeted by the current neuron
                    for n_, k_ in targets[neuron]:
                        n_.input_firing_times[k_] = np.append(n_.input_firing_times[k_][1:], ft + n_.delays[k_])

                    # update the firing threshold
                    neuron.firing_threshold = np.random.normal(self.nominal_threshold, std_threshold)