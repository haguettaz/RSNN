import os
from typing import Optional, Union

import numpy as np

from ..utils.utils import load, save
from .utils import norm


class SpikeTrain:
    def __init__(self, num_channels, duration, firing_rate, hard_refractory_period, soft_refractory_period, rng=None):
        self.rng = rng or np.random.default_rng()

        self.num_channels = num_channels
        self.duration = duration

        self.firing_rate = firing_rate
        self.hard_refractory_period = hard_refractory_period
        self.soft_refractory_period = soft_refractory_period

        self.firing_times = [np.array([]) for _ in range(num_channels)]

    def hazard(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        if self.soft_refractory_period > 0:
            return (t >= self.hard_refractory_period) * self.firing_rate * (1 - np.exp(-(t - self.hard_refractory_period) / self.soft_refractory_period))
        
        return (t >= self.hard_refractory_period) * self.firing_rate
    
    def Hazard(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        if self.soft_refractory_period > 0:
            return (t >= self.hard_refractory_period) * self.firing_rate * self.soft_refractory_period * ((t - self.hard_refractory_period) / self.soft_refractory_period + np.exp(-(t - self.hard_refractory_period) / self.soft_refractory_period) - 1)
        
        return (t >= self.hard_refractory_period) * self.firing_rate * t
    
    def q(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return self.hazard(t) * np.exp(-self.Hazard(t))
    
    def Q(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return 1 - np.exp(- self.Hazard(t))
    
    def num_spikes(self, c=None):
        if c is None:
            return np.concatenate(self.firing_times).size
        return self.firing_times[c].size
    
    def num_unique_spikes(self, c=None):
        if c is None:
            return np.unique(np.concatenate(self.firing_times)).size
        return np.unique(self.firing_times[c]).size
    
    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        for c in range(self.num_channels):
            np.save(os.path.join(dirname, f"firing_times_{c}.npy"), self.firing_times[c])

    def load(self, dirname):
        for c in range(self.num_channels):
            self.firing_times[c] = np.load(os.path.join(dirname, f"firing_times_{c}.npy"))

    def random(self, res=1e-2, rmax=1):
        # 0. initialize
        self.firing_times = []

        # assume a spike occurs at time 0, create a spike train of duration 2*duration
        t = np.arange(0, rmax*self.duration, res)
        qt = norm(self.q(t))

        for _ in range(self.num_channels):  # for each channel
            # uniformly chose the window's origin s0 in between -rmax*duration and 0
            s = self.rng.uniform(-rmax*self.duration, 0, 1)
            while s[-1] < self.duration:
                s = np.append(s, s[-1] + self.rng.choice(t, p=qt))
            self.firing_times.append(s[(s>=0) & (s<self.duration)])

class PeriodicSpikeTrain(SpikeTrain):
    def __init__(self, num_channels, duration, firing_rate, hard_refractory_period, soft_refractory_period, rng=None):
        super().__init__(num_channels, duration, firing_rate, hard_refractory_period, soft_refractory_period, rng)
    
    def random(self, res=1e-2, rtol=1e-4, rmax=5):
        # 0. initialize
        self.firing_times = []

        # implicit zero padding with rmax
        t = (np.arange(0, self.duration, res)[None,:] + self.duration * np.arange(rmax)[:,None]).flatten()
        qt = norm(self.q(t))
        idx = np.ceil(self.duration/res).astype(int) # index in t corresponding to duration

        # 1. compute the forward messages 
        qf = np.fft.rfft(qt)
        tmp = np.copy(qf)
        msgf = [qt]
        while np.argmax(msgf[-1]) < idx or msgf[-1][idx] > rtol * np.max(msgf[-1]):
            tmp = tmp * qf
            msgf.append(norm(np.around(np.fft.irfft(tmp), 9)))
        msgf = np.stack(msgf, axis=0)

        # 2. compute the distribution of the number of firings
        pn = norm(msgf[:, idx])
        
        # 3. sample per channels
        for _ in range(self.num_channels):
            # 3.a. is there any spike?
            if self.rng.binomial(1, self.Q(self.duration)) == 0:
                self.firing_times.append(np.array([]))
                continue
                
            # 3.b. given there is at least one spike, sample the exact number of spikes in the sequence
            n = self.rng.choice(len(pn), p=pn) + 1
            
            # 3.c. given the exact number of spikes, sample sn, ..., s1 by backward sampling, starting from sn = self.duration
            s = np.empty(n)
            s[-1] = self.duration
            for m in range(n-2, -1, -1):
                psm = norm(msgf[m] * self.q(s[m+1] - t))
                s[m] = self.rng.choice(t, p=psm)
                
            # 3.d independently, sample the origin of the sequence s0, and cyclically shift the sequence
            s0 = self.rng.uniform(0, self.duration)
            s += s0
            self.firing_times.append(np.sort(s % self.duration))
