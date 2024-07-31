from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw
from tqdm.autonotebook import tqdm

from .neuron import Neuron


class Network:
    """A class representing a spiking neural network.

    Attributes:
        neurons (List[Neuron]): The list of neurons composing the network.
        num_neurons (int): The number of neurons in the network.

    Methods:
        sim: Simulate the network on a given time range.
    """

    def __init__(self, neurons: Iterable[Neuron]):
        """Initialize a `Network` object with a list of `Neuron` objects.

        Args:
            neurons (int): the list of neurons composing the network.
        """
        self.neurons = neurons
        self.num_neurons = len(neurons)

    def sim(
        self,
        t0:float,
        duration: float,
        dt: Optional[float]=1e-2,
        std_threshold: Optional[float]=0.0,
        sim_indices: Optional[Iterable[int]]=None,
        tol: Optional[float] = 1e-4,
        rng: Optional[np.random.Generator]=None
    ):
        """Simulate the network on a given time range.

        Args:
            t0 (float): starting time of the simulation (in tau_min).
            duration (float): duration of the simulation (in tau_min).
            dt (float, optional): time step in (in tau_min). Defaults to 1e-2.
            std_threshold (float, optional): firing threshold nominal value standard deviation. Defaults to 0.0.
            sim_indices (list of int, optional): indices of neurons to simulate. Defaults to None, meaning all neurons.
            tol (float, optional): tolerance for negligible contribution. Defaults to 1e-4.
            rng (np.random.Generator, optional): random number generator. Defaults to None.

        Raises:
            ValueError: if the input sources, delays, and weights of any neuron are not specified.
            ValueError: if the time step is larger than the minimum delay of any neuron.
        """

        if sim_indices is None:
            sim_neurons = self.neurons
        else:
            sim_neurons = [self.neurons[i] for i in sim_indices]

        for neuron in sim_neurons:
            if neuron.sources is None:
                raise ValueError("The input sources of all neurons must be specified.")
            if neuron.delays is None:
                raise ValueError("The transmission delays of all neurons must be specified.")
            if neuron.weights is None:
                raise ValueError("The synaptic weights of all neurons must be specified.")
            if dt > np.min(neuron.delays):
                raise ValueError("The simulation time step must be smaller than the minimum delay.")
        
        for neuron in self.neurons:
            if neuron.firing_times is None:
                neuron.firing_times = np.array([])

        if rng is None:
            rng = np.random.default_rng()

        tmax = t0 + duration
        
        for neuron in sim_neurons:
            nmax = 0
            tmp = -neuron.input_beta * lambertw(-tol/np.exp(1), k=-1).real            
            for source, delay in zip(neuron.sources, neuron.delays):
                # free neurons can influence from the past (tmin) -> the number of spikes is determined using lambertw
                if self.neurons[source] in sim_neurons:
                    nmax = max(nmax, int(tmp + delay) + 1)
                # controllable neurons can influence from the past (tmin) and the future (tmax) but with a fixed number of spikes
                else:
                    tmin = t0 - tmp - delay
                    firing_times = self.neurons[source].firing_times
                    nmax = max(nmax, np.sum((firing_times>=tmin)&(firing_times<tmax)))
        
            # print("nmax:", nmax)
            # tmp = -neuron.input_beta * lambertw(-tol/np.exp(1), k=-1).real
            input_firing_times = []
            for source, delay in zip(neuron.sources, neuron.delays):
                tmin = t0 - tmp - delay
                firing_times = self.neurons[source].firing_times
                firing_times = firing_times[(firing_times>=tmin)&(firing_times<tmax)]
                # if nmax < firing_times.size:
                #     print(f"nmax={nmax} < firing_times.size={firing_times.size} for neuron {neuron.idx} and source {source}")
                input_firing_times.append(np.pad(firing_times + delay, (nmax - firing_times.size, 0), constant_values=np.nan))

            neuron.input_firing_times = np.vstack(input_firing_times)

            if neuron.firing_threshold is None:
                neuron.firing_threshold = rng.normal(neuron.nominal_threshold, std_threshold)

            neuron.potential = lambda t_: np.inner(
                neuron.weights, np.nansum(neuron.input_kernel((t_ - neuron.input_firing_times)), axis=-1)
            )

            neuron.fun = lambda t_: neuron.potential(t_) - neuron.firing_threshold

            if neuron.firing_times.size == 0:
                neuron.firing_times = np.array([-np.inf]) # to avoid checking for empty arrays in the simulation loop

        targets = {neuron: [] for neuron in self.neurons}
        for neuron in sim_neurons:
            for k, source in enumerate(neuron.sources):
                if self.neurons[source] in sim_neurons:
                    targets[self.neurons[source]].append((neuron, k))

        for neuron in sim_neurons:
            if neuron.firing_times[-1] + 1.0 >= t0:
                continue

            if neuron.fun(t0) > 0:
                neuron.firing_times = np.append(neuron.firing_times, t0)

                # update the input firing times of all neurons targeted by the current neuron
                for n_, k_ in targets[neuron]:
                    n_.input_firing_times[k_] = np.append(n_.input_firing_times[k_][1:], t0 + n_.delays[k_])

                # update the firing threshold
                neuron.firing_threshold = rng.normal(neuron.nominal_threshold, std_threshold)
        
        for t in tqdm(np.arange(t0, tmax, dt), desc="Network simulation"):
            for neuron in sim_neurons:
                if neuron.firing_times[-1] + 1.0 >= t:
                    continue

                if neuron.fun(t) > 0:
                    t_prev = np.maximum(t - dt, neuron.firing_times[-1] + 1.0)
                    if neuron.fun(t_prev) >= 0:
                        # print(f"warning at t:{t} for neuron {neuron.idx} (last spikes at {neuron.firing_times[-1]})")
                        ft = t_prev
                    else:
                        ft = brentq(neuron.fun, t_prev, t)
                    
                    neuron.firing_times = np.append(neuron.firing_times, ft)

                    # update the input firing times of all neurons targeted by the current neuron
                    for n_, k_ in targets[neuron]:
                        n_.input_firing_times[k_] = np.append(n_.input_firing_times[k_][1:], ft + n_.delays[k_])

                    # update the firing threshold
                    neuron.firing_threshold = rng.normal(neuron.nominal_threshold, std_threshold)