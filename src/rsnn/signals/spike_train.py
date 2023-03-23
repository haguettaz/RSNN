from typing import Optional, Union

import numpy as np
from scipy import signal
from tqdm.autonotebook import tqdm

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

    def __repr__(self):
        string = f"SpikeTrain(num_channels={self.num_channels}, duration={self.duration})"
        return string
    
    def generate(self, res=1e-2, rtol=1e-3):
        if self.soft_refractory_period > 0:
            hazard = lambda t_ : (t_ >= self.hard_refractory_period) * self.firing_rate * (1 - np.exp(-(t_ - self.hard_refractory_period) / self.soft_refractory_period )) # hazard function = conditional firing rate
            Hazard = lambda t_ : (t_ >= self.hard_refractory_period) * self.firing_rate * self.soft_refractory_period * ((t_ - self.hard_refractory_period) / self.soft_refractory_period + np.exp(-(t_ - self.hard_refractory_period) / soft_refractory_period) - 1)
        else:
            hazard = lambda t_ : (t_ >= self.hard_refractory_period) * self.firing_rate
            Hazard = lambda t_ : (t_ >= self.hard_refractory_period) * self.firing_rate * t_

        pdf = lambda t_ : hazard(t_) * np.exp(-Hazard(t_)) # pdf

        num_samples = int(self.duration // res)
        times = np.linspace(0, self.duration, num_samples)

        q = norm(pdf(times))

        self.firing_times = []
        for _ in range(self.num_channels):  # for each channel
            sn = np.array([0.0])
            while sn[-1] < self.duration:
                sn = np.append(sn, sn[-1] + self.rng.choice(times, p=q))
            self.firing_times.append(sn[1:-1])

class PeriodicSpikeTrain:
    def __init__(self, num_channels, period, firing_rate, hard_refractory_period, soft_refractory_period, rng=None):
        self.rng = rng or np.random.default_rng()

        self.num_channels = num_channels
        self.period = period
        self.firing_rate = firing_rate
        self.hard_refractory_period = hard_refractory_period
        self.soft_refractory_period = soft_refractory_period

        self.firing_times = [np.array([]) for _ in range(num_channels)]

    def __repr__(self):
        string = f"PeriodicSpikeTrain(num_channels={self.num_channels}, period={self.period})"
        return string
    
    def generate(self, res=1e-2, rtol=1e-3):

        if self.soft_refractory_period > 0:
            hazard = lambda t_ : (t_ >= self.hard_refractory_period) * self.firing_rate * (1 - np.exp(-(t_ - self.hard_refractory_period) / self.soft_refractory_period )) # hazard function = conditional firing rate
            Hazard = lambda t_ : (t_ >= self.hard_refractory_period) * self.firing_rate * self.soft_refractory_period * ((t_ - self.hard_refractory_period) / self.soft_refractory_period + np.exp(-(t_ - self.hard_refractory_period) / self.soft_refractory_period) - 1)
        else:
            hazard = lambda t_ : (t_ >= self.hard_refractory_period) * self.firing_rate
            Hazard = lambda t_ : (t_ >= self.hard_refractory_period) * self.firing_rate * t_

        pdf = lambda t_ : hazard(t_) * np.exp(-Hazard(t_)) # pdf
        survival = lambda t_ : np.exp(- Hazard(t_)) # survival
        no_survival = lambda t_ : 1 - np.exp(- Hazard(t_))

        num_samples = int(2*self.period // res)
        times = np.linspace(0, 2*self.period, num_samples)

        q = pdf(times)

        # 1. compute forward messages
        msgf = []
        msgf.append(signal.unit_impulse(num_samples)) # dirac delta
        msgf.append(norm(q * times))

        max_p = msgf[-1][num_samples // 2]
        while True:
            msgf.append(norm(np.convolve(q, msgf[-1])[:num_samples]))
            if msgf[-1][num_samples // 2] > max_p:
                max_p = msgf[-1][num_samples // 2]
                continue
            if msgf[-1][num_samples // 2] < rtol * max_p:
                break
        
        msgf = np.stack(msgf)
        n_max = msgf.shape[0]

        # compute marginal distribution of firing numbers
        pn = np.empty(n_max)
        pn[0] = survival(self.period) # no firing
        pn[1:] = norm(msgf[1:,num_samples // 2]) * no_survival(self.period) # one or more firings
            
        self.firing_times = []
        for _ in range(self.num_channels):
            # 1. sample the number of firings
            n = self.rng.choice(n_max, p=pn)
            if n < 1:
                self.firing_times.append(np.array([]))
                continue

            # 2. sample the interspike times
            sn = np.empty(n+1)
            sn[-1] = self.period
            for m in range(n-1, -1, -1):
                psm = norm(msgf[m] * pdf(sn[m+1] - times))
                sn[m] = self.rng.choice(times, p=psm)

            # 3. sample the exact positions of firing times
            xn = np.diff(sn)
            xn[0] = self.rng.uniform(0.0, xn[0])
            self.firing_times.append(np.cumsum(xn))