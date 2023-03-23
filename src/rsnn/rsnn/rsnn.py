import hashlib
from typing import List, Optional, Tuple, Union

import numpy as np
from tqdm.autonotebook import tqdm

from ..optim.optim import solve
from ..signals.spike_train import SpikeTrain
from ..signals.utils import sphere_intersection, sphere_intersection_complement


class Neuron:
    def __init__(
            self,
            idx: int,
            num_synapses: int, 
            firing_threshold: float,
            synaptic_decay: Union[float, List[float]], # synaptic time constant c_k, e.g., 5 ms
            somatic_decay: float, # somatic time constant c_0, e.g., 10 ms
            delays_lim: Tuple[float, float],
            hard_refractory_period: float,
            soft_refractory_period: float,
            soft_refractory_weight: float, # weight of the refractoriness feedback loop
            weights_lim: Tuple[float, float],
            weights_lvl: Optional[int] = None,
            rng: Optional[np.random.Generator] = None
    ):
        self.idx = idx
        self.rng = rng or np.random.default_rng()

        self.num_synapses = num_synapses
        self.firing_threshold = firing_threshold
        self.hard_refractory_period = hard_refractory_period
        self.soft_refractory_period = soft_refractory_period
        self.soft_refractory_weight = soft_refractory_weight

        self.weights_min, self.weights_max = weights_lim
        self.weights_lvl = weights_lvl
        self.delays_min, self.delays_max = delays_lim

        self.synaptic_decay = synaptic_decay
        self.somatic_decay = somatic_decay

        if self.somatic_decay == self.synaptic_decay:
            self.gamma = np.exp(1) / self.somatic_decay
        else:
            tmax = (np.log(self.synaptic_decay) - np.log(self.somatic_decay))/(1/self.somatic_decay - 1/self.synaptic_decay)
            self.gamma = 1 / (np.exp(- tmax / self.somatic_decay) - np.exp(- tmax / self.synaptic_decay))

        assert self.weights_min <= self.weights_max
        assert self.delays_min <= self.delays_max

        self.firing_times = np.array([])
        self.firing_patterns = [] # a list of firing patterns to memorize   

        self.sources = []
        self.targets = []

    def refractory_resp(self, t):
        # () just one time point with one firing time, (n,) one time point with n firing times, (n, m) n time points with m firing times
        if (t.ndim < 2):
            if np.any((0 < t) * (t <= self.hard_refractory_period)):
                return -np.inf
            return -self.soft_refractory_weight * np.sum(((t - self.hard_refractory_period) > 0) * np.exp(- (t - self.hard_refractory_period) / self.soft_refractory_period))

        res = -self.soft_refractory_weight * np.sum(((t - self.hard_refractory_period) > 0) * np.exp(- (t - self.hard_refractory_period) / self.soft_refractory_period), axis=-1)
        res[np.any((0 < t) * (t <= self.hard_refractory_period), axis=-1)] = -np.inf
        return res

    # def refractory_resp_deriv(self, t):
    #     return - self.refractory_weight * self.somatic_decay * np.sum((t > 0) * np.exp(- t * self.somatic_decay), axis=-1)

    def impulse_resp(self, t):
        if self.somatic_decay == self.synaptic_decay:
            return self.gamma * np.sum((t > 0) * t * np.exp(1 - t / self.somatic_decay), axis=-1)
        else:
            return self.gamma * np.sum((t > 0) * (np.exp(- t / self.somatic_decay) - np.exp(- t / self.synaptic_decay)), axis=-1)

    def impulse_resp_deriv(self, t):
        if self.somatic_decay == self.synaptic_decay:
            return self.gamma * np.sum((t > 0) * (np.exp(-t / self.somatic_decay) -  t / self.somatic_decay * np.exp(-t/self.somatic_decay)), axis=-1)
        else:
            return self.gamma * np.sum((t > 0) * (- np.exp(- t / self.somatic_decay) / self.somatic_decay + np.exp(- t / self.synaptic_decay) / self.synaptic_decay), axis=-1)

    def potential(self, t):
        # z = - self.refractory_resp((t - self.firing_times))
        # or is it faster to first create input and then inner product?
        # for k in range(self.num_synapses):
        #     # firing times must be of shape (1, n) while t must be of shape (m, 1) (sum over axis=1 is faster)
        #     z += self.weights[k] * self.impulse_resp(t[:,None] - self.sources[k].firing_times[None,:] - self.delays[k])
        # return z
        if self.num_synapses == 0:
            return np.zeros_like(t)

        if t.ndim == 0:
            y = np.array([self.impulse_resp(t - self.sources[k].firing_times - self.delays[k]) for k in range(self.num_synapses)])
            return self.weights @ y + self.refractory_resp((t - self.firing_times))

        y = np.array([self.impulse_resp(t[:,None] - self.sources[k].firing_times[None,:] - self.delays[k]) for k in range(self.num_synapses)])
        return self.weights @ y + self.refractory_resp((t[:,None] - self.firing_times[None,:]))
    
    # def potential_deriv(self, t):
    #     # z = - self.refractory_resp_deriv((t - self.firing_times))
    #     # # or is it faster to first create input and then inner product?
    #     # for k in range(self.num_synapses):
    #     #     # firing times must be of shape (1, n) while t must be of shape (m, 1) (sum over axis=1 is faster)
    #     #     z += self.weights[k] * self.impulse_resp_deriv(t[:,None] - self.sources[k].firing_times[None,:] - self.delays[k])
    #     # return z
    
    #     y = np.array([self.impulse_resp_deriv(t[:,None] - self.sources[k].firing_times[None,:] - self.delays[k]) for k in range(self.num_synapses)])
    #     return np.inner(y, self.weights) - self.refractory_resp_deriv((t[:,None] - self.firing_times[None,:]))
    
    def step(self, t, dt, noise_std, tol=1e-6):
        z = self.potential(t)
        if z > self.noisy_firing_threshold:
            # find the firing time by bisection
            ft = t - dt/2
            dt /= 2
            z = self.potential(ft)
            while np.abs(z - self.noisy_firing_threshold) > tol and dt > tol:
                if z > self.noisy_firing_threshold:
                    ft -= dt/2
                else:
                    ft += dt/2
                dt /= 2
                z = self.potential(ft)
                # if dt < 1e-9:
                #     raise ValueError("No convergence in bisection method.")
            self.firing_times = np.append(self.firing_times, ft)
            self.noisy_firing_threshold = self.firing_threshold + self.rng.normal(0, noise_std)


    # def add_firing_time(self, t_left, noise_left, t_right, noise_right, tol):
    #     # cut interval in half
    #     t_mid = (t_left + t_right) / 2
        
    #     # interpolate noise at t_mid
    #     mean = (noise_left * (t_right - t_mid) + noise_right * (t_mid - t_left)) / (t_right - t_left)
    #     std = self.noise_std * math.sqrt((t_mid - t_left) * (t_right - t_mid) / (t_right - t_left))
    #     noise_mid = self.rng.normal(mean, std)
    #     while noise_mid < self.noise_min or noise_mid > self.noise_max:
    #         noise_mid = self.rng.normal(mean, std)

    #     # get noisy potential at t_mid
    #     z_mid = self.potential(t_mid) + noise_mid

    #     if abs(z_mid - self.firing_threshold) < tol or abs(t_right - t_left) < tol:
    #         self.firing_times = np.append(self.firing_times, t_mid)
    #         return

    #     # recursive call on left or right interval
    #     if z_mid > self.firing_threshold:
    #         self.add_firing_time(t_left, noise_left, t_mid, noise_mid, tol)
    #     else:
    #         self.add_firing_time(t_mid, noise_mid, t_right, noise_right, tol)

    # def clear_firing_times(self):
    #     self.firing_times = np.array([])

    # def clear_firing_patterns(self):
    #     self.firing_patterns = []
    #     self.durations_patterns = []

    # def set_firing_patterns(self, firing_patterns, durations_patterns):
    #     self.firing_patterns = [np.array(firing_pattern) for firing_pattern in firing_patterns]
    #     self.durations_patterns = durations_patterns

    # def set_firing_times(self, firing_times):
    #     self.firing_times = np.array(firing_times)

    # def set_num_active_spikes(self, tol):
    #     self.num_active_spikes = 0
    #     for neuron in self.targets:
    #         synapses = [k for k in range(neuron.num_synapses) if neuron.sources[k] is self]
    #         tmp = np.min(neuron.delays[synapses]) / self.refractory_period + 1
    #         while sum([neuron.weights[k] * neuron.impulse_resp[k](tmp * self.refractory_period - neuron.delays[k]) for k in synapses]) > tol:
    #             tmp += 1
    #         if tmp > self.num_active_spikes:
    #             self.num_active_spikes = tmp     
    #     return self.num_active_spikes   

    # def active_spikes(self, t, idx):
    #     last_idx = np.searchsorted((t - self.firing_patterns[idx]) % self.duration_patterns[idx], 0, side="right")
    #     for i in range(self.num_active_spikes):
    #         yield self.firing_patterns[idx][(last_idx - i) % self.firing_patterns[idx].size]

    # def source_spike_influence(self, t, source, s):
    #     return np.nansum([self.weights[k] * self.impulse_resp(t - s - self.delays[k], k) for k in range(self.num_synapses) if self.sources[k] is source])

    # def source_spike_influence_deriv(self, t, source, s):
    #     return np.nansum([self.weights[k] * self.impulse_resp_deriv(t - s - self.delays[k], k) for k in range(self.num_synapses) if self.sources[k] is source])

    def memorize(
            self, 
            spike_trains: Union[List[SpikeTrain], SpikeTrain],
            slope: float,
            level: float,
            eps: float,
            dt: float = 0.1,
            ):

        assert (slope > 0) and (level > 0)
        assert dt < eps

        # set time constraints
        # C, a, b = np.empty((0, self.num_synapses)), np.empty(0), np.empty(0)

        if isinstance(spike_trains, SpikeTrain):
            spike_trains = [spike_trains]

        a, b, C = np.empty(0), np.empty(0), np.empty((0, self.num_synapses))

        for spike_train in spike_trains:
            silent_times = sphere_intersection_complement(spike_train.firing_times[self.idx], eps, spike_train.period, dt)
            active_times = sphere_intersection(spike_train.firing_times[self.idx], eps, spike_train.period, dt)
            
            firing_min = self.firing_threshold - self.refractory_resp((spike_train.firing_times[self.idx][:,None] - spike_train.firing_times[self.idx][None,:]) % spike_train.period)
            silent_min = np.full_like(silent_times, -np.inf)
            active_min = np.full_like(active_times, slope)
            a = np.concatenate((a, firing_min.T, silent_min.T, active_min.T), axis=0)

            firing_max = self.firing_threshold - self.refractory_resp((spike_train.firing_times[self.idx][:,None] - spike_train.firing_times[self.idx][None,:]) % spike_train.period)
            silent_max = self.firing_threshold - level - self.refractory_resp((silent_times[:,None] - spike_train.firing_times[self.idx][None,:]) % spike_train.period)
            active_max = np.full_like(active_times, np.inf)
            b = np.concatenate((b, firing_max.T, silent_max.T, active_max.T), axis=0)

            firing_y = np.array([self.impulse_resp((spike_train.firing_times[self.idx][:,None] - spike_train.firing_times[self.sources[k].idx][None,:] - self.delays[k]) % spike_train.period) for k in range(self.num_synapses)])
            silent_y = np.array([self.impulse_resp((silent_times[:,None] - spike_train.firing_times[self.sources[k].idx][None,:] - self.delays[k]) % spike_train.period) for k in range(self.num_synapses)])
            active_y = np.array([self.impulse_resp_deriv((active_times[:,None] - spike_train.firing_times[self.sources[k].idx][None,:] - self.delays[k]) % spike_train.period) for k in range(self.num_synapses)])
            C = np.concatenate((C, firing_y.T, silent_y.T, active_y.T), axis=0)

            # if spike_train.firing_times[self.idx].size > 0:
            #     # firing time itself
            #     for t in spike_train.firing_times[self.idx]:
            #         y = np.array([self.impulse_resp((t - spike_train.firing_times[self.sources[k].idx] - self.delays[k]) % spike_train.period) for k in range(self.num_synapses)])[None,:]
            #         C = np.append(C, y, axis=0)
            #         a = np.append(a, self.firing_threshold + self.refractory_resp((t - spike_train.firing_times[self.idx]) % spike_train.period))
            #         b = np.append(b, self.firing_threshold + self.refractory_resp((t - spike_train.firing_times[self.idx]) % spike_train.period))

            #     for t in np.arange(0, spike_train.period, dt):
            #         diff = np.abs(t - spike_train.firing_times[self.idx])
            #         dist = np.minimum(diff, spike_train.period - diff)
            #         ft_idx = np.argmin(dist)

            #         # eps-close to a firing time
            #         if dist[ft_idx] < eps:
            #             y = np.array([self.impulse_resp_deriv((t - spike_train.firing_times[self.sources[k].idx] - self.delays[k]) % spike_train.period) for k in range(self.num_synapses)])[None,:]
            #             C = np.append(C, y, axis=0)
            #             a = np.append(a, slope)
            #             b = np.append(b, np.inf)
                    
            #         # complementary case
            #         else:
            #             y = np.array([self.impulse_resp((t - spike_train.firing_times[self.sources[k].idx] - self.delays[k]) % spike_train.period) for k in range(self.num_synapses)])[None,:]
            #             C = np.append(C, y, axis=0)
            #             a = np.append(a, -np.inf)
            #             b = np.append(b, self.firing_threshold + self.refractory_resp((t - spike_train.firing_times[self.idx]) % spike_train.period) - level)

            # else:
            #     for t in np.arange(0, spike_train.period, dt):
            #         y = np.array([self.impulse_resp((t - spike_train.firing_times[self.sources[k].idx] - self.delays[k]) % spike_train.period) for k in range(self.num_synapses)])[None,:]
            #         C = np.append(C, y, axis=0)
            #         a = np.append(a, -np.inf)
            #         b = np.append(b, self.firing_threshold + self.refractory_resp((t - spike_train.firing_times[self.idx]) % spike_train.period) - level)
        

            # # for t in times_generator(spike_train.firing_times[self.idx], spike_train.period):
            # #     y = np.array([self.impulse_resp((t - spike_train.firing_times[self.sources[k].idx] - self.delays[k]) % spike_train.period) for k in range(self.num_synapses)])[None,:]
            # #     C = np.append(C, y, axis=0)
            # #     a = np.append(a, self.firing_threshold + self.refractory_resp(t - spike_train.firing_times[self.idx]))
            #     b = np.append(b, self.firing_threshold + self.refractory_resp(t - spike_train.firing_times[self.idx]))
            #     print(t, self.refractory_resp(t - spike_train.firing_times[self.idx]))

            # for t in surrounding_times_generator(spike_train.firing_times[self.idx], -eps, eps, res, spike_train.period):
            #     y = np.array([self.impulse_resp_deriv((t - spike_train.firing_times[self.sources[k].idx] - self.delays[k]) % spike_train.period) for k in range(self.num_synapses)])[None,:]
            #     C = np.append(C, y, axis=0)
            #     a = np.append(a, slope)
            #     b = np.append(b, np.inf)

            # for t in complement_surrounding_times_generator(spike_train.firing_times[self.idx], -eps, eps, res, spike_train.period):
            #     y = np.array([self.impulse_resp((t - spike_train.firing_times[self.sources[k].idx] - self.delays[k]) % spike_train.period) for k in range(self.num_synapses)])[None,:]
            #     C = np.append(C, y, axis=0)
            #     a = np.append(a, -np.inf)
            #     b = np.append(b, self.firing_threshold + self.refractory_resp(t - spike_train.firing_times[self.idx]) - level)

        assert C.shape[0] == a.shape[0] == b.shape[0]
        assert C.shape[0] > 0        

        if self.weights_lvl is not None:
            raise NotImplementedError("Memorization is not implemented for discrete weights yet.")
        
        self.weights = solve(
            C,
            a,
            b,
            (np.full(self.num_synapses, self.weights_min), np.full(self.num_synapses, self.weights_max)),
            self.weights_lvl,
            1000, 
            1e-4,
            self.rng
            )
        
        # print(f"Optimization of neuron {self.idx} finished with status {status}")

    # def run(self, t, time_step, tol):
    #     # TODO: using derivative instead of analytical solution?
    #     self.eta[1] = forward_OU(self.eta[0], time_step)
    #     if self.potential(t) > self.firing_threshold:
    #         self.add_firing_time(t - time_step, self.noise_prev, t, noise, tol)

class Network:
    def __init__(
            self, 
            num_neurons: int, 
            num_synapses: int, 
            firing_threshold: float,
            synaptic_decay: Union[float, List[float]], # inverse synaptic time constant c_k, e.g., 1/5 ms
            somatic_decay: float, # inverse somatic time constant c_0, e.g., 1/10 ms
            delays_lim: Tuple[float, float],
            hard_refractory_period:float,
            soft_refractory_period: float,
            soft_refractory_weight: float, # weight of the refractoriness feedback loop
            weights_lim: Tuple[float, float],
            weights_lvl: Optional[int] = None,
            rng: Optional[np.random.Generator] = None,
            ):
        
        self.num_neurons = num_neurons
        self.rng = rng or np.random.default_rng()

        self.num_synapses = num_synapses
        self.firing_threshold = firing_threshold
        self.synaptic_decay = synaptic_decay
        self.somatic_decay = somatic_decay
        self.delays_lim = delays_lim
        self.hard_refractory_period = hard_refractory_period
        self.soft_refractory_period = soft_refractory_period
        self.soft_refractory_weight = soft_refractory_weight
        self.weights_lim = weights_lim
        self.weights_lvl = weights_lvl

        self.neurons = [
            Neuron(
                l, 
                num_synapses,
                firing_threshold,
                synaptic_decay,
                somatic_decay,
                delays_lim,
                hard_refractory_period,
                soft_refractory_period,
                soft_refractory_weight,
                weights_lim,
                weights_lvl,
                rng
                ) for l in range(self.num_neurons)]
        
        self.connect()
                
    def __repr__(self):
        string = "Network("
        string += f"num_neurons={self.num_neurons}, "
        string += f"num_synapses={self.num_synapses}, "
        string += f"firing_threshold={self.firing_threshold}, "
        string += f"synaptic_decay={self.synaptic_decay}, "
        string += f"somatic_decay={self.somatic_decay}, "
        string += f"delays_lim={self.delays_lim}, "
        string += f"hard_refractory_period={self.hard_refractory_period}, "
        string += f"soft_refractory_period={self.soft_refractory_period}, "
        string += f"soft_refractory_weight={self.soft_refractory_weight}, "
        string += f"weights_lim={self.weights_lim}, "
        string += f"weights_lvl={self.weights_lvl})"
        return string
        
    def __hash__(self):
        return hashlib.md5(self.__repr__().encode('utf-8')).hexdigest()

    def connect(self):
        # set source neurons (with weights and delays)
        for neuron in self.neurons:
            sources = self.rng.integers(0, self.num_neurons, neuron.num_synapses)
            neuron.sources = [self.neurons[source] for source in sources]
            neuron.delays = self.rng.uniform(neuron.delays_min, neuron.delays_max, neuron.num_synapses)
            neuron.weights = np.zeros(neuron.num_synapses)
            for neuron_src in set(neuron.sources):
                neuron_src.targets.append(neuron)

    # def clear_firing_times(self):
    #     for neuron in self.neurons:
    #         neuron.clear_firing_times()

    # def clear_firing_patterns(self):
    #     for neuron in self.neurons:
    #         neuron.clear_firing_patterns()

    # def set_firing_times(self, spike_train: List[List[float]]):
    #     for neuron in self.neurons:
    #         neuron.set_firing_times(firing_times[neuron.idx])

    # def set_firing_patterns(self, firing_patterns: List[List[np.ndarray]]):
    #     self.num_firing_patterns = len(firing_patterns)
    #     for neuron in self.neurons:
    #         neuron.set_firing_patterns([firing_pattern[neuron.idx] for firing_pattern in firing_patterns])

    # def get_stability_matrices(self, tol=1e-6):
    #     for neuron in self.neurons:
    #         neuron.set_num_active_spikes(tol)
        
    #     # set matrix dimension and indices
    #     dim = np.sum([neuron.num_active_spikes for neuron in self.neurons])
    #     indices = np.cumsum([0] + [neuron.num_active_spikes for neuron in self.neurons])

    #     # one stability matrix for each firing pattern
    #     stability_matrices = []
    #     for i in range(self.num_firing_patterns):
    #         # extract all firing times of the network (plus the associated neurons) and sort them by time
    #         firing_times = []
    #         firing_neurons = []
    #         for neuron in self.neurons:
    #             firing_times.append(neuron.firing_patterns[i])
    #             firing_neurons.append(np.full_like(neuron.firing_patterns[i], neuron.idx))
    #         firing_times = np.concatenate(firing_times)
    #         firing_neurons = np.concatenate(firing_neurons).astype(int)
            
    #         Phi = np.identity(dim)
    #         for t in np.unique(firing_times):
    #             # sorted according to indices
    #             mask = (firing_times == t)

    #             firing_neurons_indices = indices[firing_neurons[mask]]

    #             A = np.identity(dim)
    #             # for neurons that fire at time t
    #             for idx in firing_neurons_indices:
    #                 neuron = self.get_neuron(idx)
    #                 # compute influence of all active spikes of the sources on the new last spike
    #                 for source in set(neuron.sources):
    #                     for j, s in enumerate(source.active_spikes):
    #                         A[idx, indices[source.idx] + j] = neuron.spike_influence_deriv(t, source, s)
                    
    #                 # shift all last spikes of the neuron by one
    #                 for j in range(1, neuron.num_active_spikes):
    #                     A[idx + j] = np.roll(A[idx + j], 1)

    #             Phi = A @ Phi

    #         stability_matrices.append(Phi)
    #     return stability_matrices

    def memorize(
            self, 
            spike_trains: Union[List[SpikeTrain], SpikeTrain],
            slope: float,
            level: float,
            eps: tuple,
            res: float = 0.1,
            ):
        
        for neuron in tqdm(self.neurons):
            neuron.memorize(spike_trains, slope, level, eps, res)

    def run(
            self, 
            duration: float, 
            dt: float, 
            noise_std: float,
            ):
        # only consider autonomous neurons, other neurons have a predetermined firing pattern
        autonomous_neurons = [neuron for neuron in self.neurons if neuron.num_synapses > 0]

        for neuron in autonomous_neurons:
            neuron.noisy_firing_threshold = neuron.firing_threshold + neuron.rng.normal(0, noise_std)

        for t in tqdm(np.arange(0, duration+dt, dt)):
            for neuron in autonomous_neurons:
            # for neuron in tqdm(autonomous_neurons, leave=False):
                neuron.step(t, dt, noise_std)