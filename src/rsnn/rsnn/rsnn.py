import hashlib
import math
import os
import pickle
import random
# from multiprocessing import Pool
from typing import List, Optional, Tuple, Union

import numpy as np
from tqdm.autonotebook import tqdm

from ..optim.nuv import box_prior, m_ary_prior
from ..optim.optim import solve
from .utils import (active_times_generator, firing_times_generator,
                    silent_times_generator)


class Neuron:
    def __init__(
            self,
            idx: int,
            num_synapses: int, 
            refractory_period: float, 
            firing_threshold: float,
            synaptic_decay: Union[float, List[float]], # inverse synaptic time constant c_k, e.g., 1/5 ms
            somatic_decay: float, # inverse somatic time constant c_0, e.g., 1/10 ms
            weights_lim: Union[Tuple, List[Tuple]],
            weights_lvl: Optional[int],
            delays_lim: Union[Tuple[float, float], List[Tuple[float, float]]],
            rng: Optional[np.random.Generator] = None,
    ):
        self.idx = idx
        self.rng = rng or np.random.default_rng()

        self.num_synapses = num_synapses
        self.refractory_period = refractory_period
        self.firing_threshold = firing_threshold

        if isinstance(synaptic_decay, (int, float)):
            synaptic_decay = np.array([synaptic_decay for _ in range(self.num_synapses)])
        assert len(synaptic_decay) == self.num_synapses
        self.synaptic_decay = synaptic_decay
        if isinstance(somatic_decay, (int, float)):
            somatic_decay = np.array([somatic_decay for _ in range(self.num_synapses)])
        assert len(somatic_decay) == self.num_synapses
        self.somatic_decay = somatic_decay
        self.gamma = 1 / np.array([np.exp(-self.somatic_decay[k] * np.log(self.somatic_decay[k]/self.synaptic_decay[k])/(self.somatic_decay[k] - self.synaptic_decay[k])) - np.exp(-self.synaptic_decay[k] * np.log(self.somatic_decay[k]/self.synaptic_decay[k])/(self.somatic_decay[k] - self.synaptic_decay[k])) if self.synaptic_decay[k] != self.somatic_decay[k] else 1/self.somatic_decay[k] for k in range(self.num_synapses)])

        assert isinstance(weights_lim, tuple) or len(weights_lim) == self.num_synapses
        if isinstance(weights_lim, tuple):        
            self.weights_min = np.full(self.num_synapses, weights_lim[0])
            self.weights_max = np.full(self.num_synapses, weights_lim[1])
        else:
            self.weights_min = np.array([w_lim[0] for w_lim in weights_lim])
            self.weights_max = np.array([w_lim[1] for w_lim in weights_lim])
        assert np.all(self.weights_min < self.weights_max)
        assert np.all(np.isfinite(self.weights_min))
        assert np.all(np.isfinite(self.weights_max))
        self.weights_lvl = weights_lvl or np.inf

        assert isinstance(delays_lim, tuple) or len(delays_lim) == self.num_synapses
        if isinstance(delays_lim, tuple):        
            self.delays_min = np.full(self.num_synapses, delays_lim[0] or 0)
            self.delays_max = np.full(self.num_synapses, delays_lim[1])
        else:
            self.delays_min = np.array([d_lim[0] or 0 for d_lim in delays_lim])
            self.delays_max = np.array([d_lim[1] for d_lim in delays_lim])
        assert np.all(self.delays_min < self.delays_max)
        assert np.all(np.isfinite(self.weights_max))

        self.firing_times = np.array([])
        self.firing_patterns = [] # a list of firing patterns to memorize   

        self.sources = []
        self.targets = []

    def __repr__(self):
        string = f" Neuron {self.idx}:\n"
        string += f"  Refractory Period: {self.refractory_period}\n"
        string += f"  Firing Threshold: {self.firing_threshold}\n"
        
        for k in range(self.num_synapses):
            string += f"  Synapse {k}:\n"
            string += f"   Weight: {self.weights_min[k], self.weights_max[k]} with {self.weights_lvl} levels\n"
            string += f"   Delay: {self.delays_min[k], self.delays_max[k]}\n"
            string += f"   Impulse Response: {self.synaptic_decay[k], self.somatic_decay[k]}\n"
        
        return string

    def impulse_resp(self, t, k):
        if self.synaptic_decay[k] == self.somatic_decay[k]:
            return np.nansum((t >= 0) * t * np.exp(1 - t * self.somatic_decay[k]) * self.somatic_decay[k])

        return np.nansum((t >= 0) * (np.exp(-t * self.somatic_decay[k]) - np.exp(-t * self.synaptic_decay[k])) * self.gamma[k])
    
    def impulse_resp_deriv(self, t, k):
        if self.synaptic_decay[k] == self.somatic_decay[k]:
            return np.nansum((t >= 0) * (1 - t * self.somatic_decay[k]) * np.exp(1 - t * self.somatic_decay[k]) * self.somatic_decay[k])

        return np.nansum((t >= 0) * (- self.somatic_decay[k] * np.exp(-t * self.somatic_decay[k]) + self.synaptic_decay[k] * np.exp(-t * self.synaptic_decay[k])) * self.gamma[k])

    def set_noise(self, noise_std, noise_lim):
        self.noise_std = noise_std
        self.noise_min, self.noise_max = noise_lim
        self.noise_prev = self.rng.uniform(self.noise_min, self.noise_max)

    # potential noise
    def noise(self, time_step):
        mean = self.noise_prev
        std = self.noise_std * math.sqrt(time_step)
        noise = self.rng.normal(mean, std)
        while noise < self.noise_min or noise > self.noise_max:
            noise = self.rng.normal(mean, std)
        return noise

    # noise free potential
    def potential(self, t):
        if self.num_synapses == 0:
            return 0.0

        # find last firing time before t
        idx = np.searchsorted(self.firing_times, t, "left")
        if idx > 0 and (t - self.firing_times[idx - 1] <= self.refractory_period):
            return 0.0
        
        y = np.array([self.impulse_resp(t - self.sources[k].firing_times - self.delays[k], k) for k in range(self.num_synapses)])
        return np.inner(self.weights, y)  

    # noise free potential derivative
    def potential_deriv(self, t):
        if self.num_synapses == 0:
            return 0.0

        # find last firing time before t
        idx = np.searchsorted(self.firing_times, t, "left")
        if idx > 0 and (t - self.firing_times[idx - 1] <= self.refractory_period):
            return 0.0
        
        y = np.array([self.impulse_resp_deriv(t - self.sources[k].firing_times - self.delays[k], k) for k in range(self.num_synapses)])
        return np.inner(self.weights, y)  
    
    def add_firing_time(self, t_left, noise_left, t_right, noise_right, tol):
        # cut interval in half
        t_mid = (t_left + t_right) / 2
        
        # interpolate noise at t_mid
        mean = (noise_left * (t_right - t_mid) + noise_right * (t_mid - t_left)) / (t_right - t_left)
        std = self.noise_std * math.sqrt((t_mid - t_left) * (t_right - t_mid) / (t_right - t_left))
        noise_mid = self.rng.normal(mean, std)
        while noise_mid < self.noise_min or noise_mid > self.noise_max:
            noise_mid = self.rng.normal(mean, std)

        # get noisy potential at t_mid
        z_mid = self.potential(t_mid) + noise_mid

        if abs(z_mid - self.firing_threshold) < tol or abs(t_right - t_left) < tol:
            self.firing_times = np.append(self.firing_times, t_mid)
            return

        # recursive call on left or right interval
        if z_mid > self.firing_threshold:
            self.add_firing_time(t_left, noise_left, t_mid, noise_mid, tol)
        else:
            self.add_firing_time(t_mid, noise_mid, t_right, noise_right, tol)

    def clear_firing_times(self):
        self.firing_times = np.array([])

    def clear_firing_patterns(self):
        self.firing_patterns = []
        self.durations_patterns = []

    def set_firing_patterns(self, firing_patterns, durations_patterns):
        self.firing_patterns = [np.array(firing_pattern) for firing_pattern in firing_patterns]
        self.durations_patterns = durations_patterns

    def set_firing_times(self, firing_times):
        self.firing_times = np.array(firing_times)

    def set_num_active_spikes(self, tol):
        self.num_active_spikes = 0
        for neuron in self.targets:
            synapses = [k for k in range(neuron.num_synapses) if neuron.sources[k] is self]
            tmp = np.min(neuron.delays[synapses]) / self.refractory_period + 1
            while sum([neuron.weights[k] * neuron.impulse_resp[k](tmp * self.refractory_period - neuron.delays[k]) for k in synapses]) > tol:
                tmp += 1
            if tmp > self.num_active_spikes:
                self.num_active_spikes = tmp     
        return self.num_active_spikes   

    def active_spikes(self, t, idx):
        last_idx = np.searchsorted((t - self.firing_patterns[idx]) % self.duration_patterns[idx], 0, side="right")
        for i in range(self.num_active_spikes):
            yield self.firing_patterns[idx][(last_idx - i) % self.firing_patterns[idx].size]

    def source_spike_influence(self, t, source, s):
        return np.nansum([self.weights[k] * self.impulse_resp(t - s - self.delays[k], k) for k in range(self.num_synapses) if self.sources[k] is source])

    def source_spike_influence_deriv(self, t, source, s):
        return np.nansum([self.weights[k] * self.impulse_resp_deriv(t - s - self.delays[k], k) for k in range(self.num_synapses) if self.sources[k] is source])

    def memorize_firing_patterns(
            self, 
            duration: float,
            time_step: float = 0.1,
            active_lim: tuple = (-1.0, 1.0),
            active_deriv_min: float = 0.0,
            silent_max: float = 0.0,
            ):
        
        if self.weights_lvl < np.inf:
            raise NotImplementedError("Memorization is not implemented for discrete weights yet.")

        # check if template params are valid
        assert active_deriv_min > 0
        active_min, active_max = active_lim
        
        # create observation vectors
        num_patterns = len(self.firing_patterns)
        C_firing_list, C_active_list, C_silent_list = [], [], []
        for i in range(num_patterns):
            # access firing times
            for t in firing_times_generator(self.firing_patterns[i], duration):
                C_firing_list.append([self.impulse_resp((t - self.sources[k].firing_patterns[i] - self.delays[k]) % duration, k) for k in range(self.num_synapses)])

            # access active times
            for t in active_times_generator(self.firing_patterns[i], active_min, active_max, time_step, duration):
                C_active_list.append([self.impulse_resp_deriv((t - self.sources[k].firing_patterns[i] - self.delays[k]) % duration, k) for k in range(self.num_synapses)])

            # access silent times
            for t in silent_times_generator(self.firing_patterns[i], active_min, self.refractory_period, time_step, duration):
                C_silent_list.append([self.impulse_resp((t - self.sources[k].firing_patterns[i] - self.delays[k]) % duration, k) for k in range(self.num_synapses)])

        C_firing = np.array(C_firing_list) if len(C_firing_list) > 0 else np.empty((0, self.num_synapses))
        C_active = np.array(C_active_list) if len(C_active_list) > 0 else np.empty((0, self.num_synapses))
        C_silent = np.array(C_silent_list) if len(C_silent_list) > 0 else np.empty((0, self.num_synapses))

        assert ~np.any(np.all(C_firing == 0, axis=1))
        assert ~np.any(np.all(C_active == 0, axis=1))
        assert ~np.any(np.all(C_silent == 0, axis=1))

        weights = self.rng.uniform(self.weights_min, self.weights_max)

        self.weights, status = solve(
            weights, 
            C_firing,
            C_active,
            C_silent,
            lambda w_: box_prior(w_, self.weights_min, self.weights_max, 1),
            lambda z_: box_prior(z_, np.full_like(z_, self.firing_threshold), np.full_like(z_, self.firing_threshold), 1),
            lambda z_: box_prior(z_, np.full_like(z_, active_deriv_min), np.full_like(z_, np.inf), 1),
            lambda z_: box_prior(z_, np.full_like(z_, -np.inf), np.full_like(z_, silent_max), 1),
            lambda w_: np.max(np.abs(w_ - self.weights_min) + np.abs(self.weights_max - w_) - (self.weights_max - self.weights_min)),
            lambda z_: np.max(np.abs(z_ - self.firing_threshold)),
            lambda z_: np.max(np.abs(z_ - active_deriv_min) - (z_ - active_deriv_min)),
            lambda z_: np.max(np.abs(z_ - silent_max) + (z_ - silent_max)),
            )
        print(f"Optimization of neuron {self.idx} finished with status {status}", flush=True)

        
    def run(self, t, time_step, tol):
        noise = self.noise(time_step)
        if self.potential(t) + noise > self.firing_threshold:
            self.add_firing_time(t - time_step, self.noise_prev, t, noise, tol)
        self.noise_prev = noise

class Network:
    def __init__(
            self, 
            num_neurons: int, 
            num_synapses: Union[int, List[int]], 
            refractory_period: Union[float, List[float]], 
            firing_threshold: Union[float, List[float]],
            synaptic_decay: Union[float, List[float], List[List[float]]], # inverse synaptic time constant c_k, e.g., 1/5 ms
            somatic_decay: Union[float, List[float]], # inverse somatic time constant c_0, e.g., 1/10 ms
            weights_lim: Union[Tuple[float, float], List[Tuple[float, float]], List[List[Tuple[float, float]]]],
            weights_lvl: Union[Optional[int], List[Optional[int]]],
            delays_lim: Union[Tuple[float, float], List[Tuple[float, float]], List[List[Tuple[float, float]]]],
            rng: Optional[np.random.Generator] = None
            ):
        
        self.num_neurons = num_neurons
        self.rng = rng or np.random.default_rng()

        if isinstance(num_synapses, int):
            num_synapses = [num_synapses for _ in range(self.num_neurons)]
        assert len(num_synapses) == self.num_neurons

        if isinstance(synaptic_decay, (int, float)):
            synaptic_decay = [synaptic_decay for _ in range(self.num_neurons)]
        assert len(synaptic_decay) == self.num_neurons
        
        if isinstance(somatic_decay, (int, float)):
            somatic_decay = [somatic_decay for _ in range(self.num_neurons)]
        assert len(somatic_decay) == self.num_neurons
        
        if isinstance(refractory_period, (int, float)):
            refractory_period = [refractory_period for _ in range(self.num_neurons)]
        assert len(refractory_period) == self.num_neurons

        if isinstance(firing_threshold, (int, float)):
            firing_threshold = [firing_threshold for _ in range(self.num_neurons)]
        assert len(firing_threshold) == self.num_neurons

        if isinstance(weights_lim, tuple):
            weights_lim = [weights_lim for _ in range(self.num_neurons)]
        assert len(weights_lim) == self.num_neurons
            
        if weights_lvl is None or isinstance(weights_lvl, int):
            weights_lvl = [weights_lvl for _ in range(self.num_neurons)]
        assert len(weights_lvl) == self.num_neurons

        if isinstance(delays_lim, tuple):
            delays_lim = [delays_lim for _ in range(self.num_neurons)]
        assert len(delays_lim) == self.num_neurons
        self.delays_max = np.max([delay_lim[1] for delay_lim in delays_lim])

        self.neurons = [
            Neuron(
                l, 
                num_synapses[l],
                refractory_period[l],
                firing_threshold[l],
                synaptic_decay[l],
                somatic_decay[l],
                weights_lim[l],
                weights_lvl[l],
                delays_lim[l],
                rng
                ) for l in range(self.num_neurons)]
                
    def __repr__(self):
        string = "Network:"
        for neuron in self.neurons:
            string += neuron.__repr__()
        return string

    def __hash__(self):
        return hashlib.md5(self.__repr__().encode('utf-8')).hexdigest()

    def connect(self):
        # set source neurons (with weights and delays)
        for neuron in self.neurons:
            sources = self.rng.integers(0, self.num_neurons, neuron.num_synapses)
            neuron.sources = [self.neurons[source] for source in sources]
            neuron.delays = self.rng.uniform(neuron.delays_min, neuron.delays_max)
            neuron.weights = np.zeros(neuron.num_synapses)
            for neuron_src in set(neuron.sources):
                neuron_src.targets.append(neuron)

    def clear_firing_times(self):
        for neuron in self.neurons:
            neuron.clear_firing_times()

    def clear_firing_patterns(self):
        for neuron in self.neurons:
            neuron.clear_firing_patterns()

    def set_firing_times(self, firing_times: List[List[float]]):
        for neuron in self.neurons:
            neuron.set_firing_times(firing_times[neuron.idx])

    def set_firing_patterns(self, firing_patterns: List[List[np.ndarray]]):
        self.num_firing_patterns = len(firing_patterns)
        for neuron in self.neurons:
            neuron.set_firing_patterns([firing_pattern[neuron.idx] for firing_pattern in firing_patterns])

    def get_stability_matrices(self, tol=1e-6):
        for neuron in self.neurons:
            neuron.set_num_active_spikes(tol)
        
        # set matrix dimension and indices
        dim = np.sum([neuron.num_active_spikes for neuron in self.neurons])
        indices = np.cumsum([0] + [neuron.num_active_spikes for neuron in self.neurons])

        # one stability matrix for each firing pattern
        stability_matrices = []
        for i in range(self.num_firing_patterns):
            # extract all firing times of the network (plus the associated neurons) and sort them by time
            firing_times = []
            firing_neurons = []
            for neuron in self.neurons:
                firing_times.append(neuron.firing_patterns[i])
                firing_neurons.append(np.full_like(neuron.firing_patterns[i], neuron.idx))
            firing_times = np.concatenate(firing_times)
            firing_neurons = np.concatenate(firing_neurons).astype(int)
            
            Phi = np.identity(dim)
            for t in np.unique(firing_times):
                # sorted according to indices
                mask = (firing_times == t)

                firing_neurons_indices = indices[firing_neurons[mask]]

                A = np.identity(dim)
                # for neurons that fire at time t
                for idx in firing_neurons_indices:
                    neuron = self.get_neuron(idx)
                    # compute influence of all active spikes of the sources on the new last spike
                    for source in set(neuron.sources):
                        for j, s in enumerate(source.active_spikes):
                            A[idx, indices[source.idx] + j] = neuron.spike_influence_deriv(t, source, s)
                    
                    # shift all last spikes of the neuron by one
                    for j in range(1, neuron.num_active_spikes):
                        A[idx + j] = np.roll(A[idx + j], 1)

                Phi = A @ Phi

            stability_matrices.append(Phi)
        return stability_matrices

    def memorize_firing_patterns(
            self, 
            duration: float, 
            time_step: float = 0.1,
            active_lim: Tuple[float, float] = (-1.0, 1.0),
            active_deriv_min: float = 1e-6,
            silent_max: float = 0.0
            ):
        
        for neuron in self.neurons:
            neuron.memorize_firing_patterns(duration, time_step, active_lim, active_deriv_min, silent_max)

    def run(
            self, 
            duration: float, 
            time_step: float, 
            noise_lim: Tuple[float, float],
            noise_std: float,
            tol: float = 1e-6,
            ):
        # only consider autonomous neurons, other neurons have a predetermined firing pattern
        autonomous_neurons = [neuron for neuron in self.neurons if neuron.num_synapses > 0]

        # create potential noises for each neuron
        for neuron in autonomous_neurons:
            neuron.set_noise(noise_std, noise_lim)

        for t in tqdm(np.arange(0, duration, time_step)):
            for neuron in autonomous_neurons:
                neuron.run(t, time_step, tol)