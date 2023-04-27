import math
import os
from typing import Optional, Union

import numpy as np
from tqdm.autonotebook import trange

from .utils import norm


class SpikeTrain:
    """A spike train is a sequence of firing times. It is defined by a firing rate, a hard and a soft refractory period, and a duration.
    """
    def __init__(
            self, 
            num_channels:int, 
            duration:float, 
            firing_rate:float, 
            abs_ref:float, 
            rel_ref:float, 
            rng:np.random.Generator=None
            ) -> None:
        """Initialize a spike train.

        Args:
            num_channels (int): the number of channels / neurons.
            duration (float): the duration of the spike train.
            firing_rate (float): the firing rate of the spike train.
            abs_ref (float): the absolute refractory period.
            rel_ref (float): the relative refractory period, i.e., the time constant of the refractoriness to the nominal firing rate.
            rng (np.random.Generator, optional): the random number generator. Defaults to None.
        """

        self.num_channels = num_channels
        self.duration = duration

        self.firing_rate = firing_rate

        if abs_ref <= 0:
            raise ValueError("Absolute refractory period must be positive.")
        self.abs_ref = abs_ref

        if rel_ref <= 0:
            raise ValueError("Relative refractory period must be positive.")
        self.rel_ref = rel_ref

        self.rng = rng or np.random.default_rng()

        self.firing_times = [np.array([]) for _ in range(num_channels)]

    def hazard(
            self, 
            t: Union[np.ndarray, float]
            ) -> Union[np.ndarray, float]:
        """Hazard function, i.e. the firing rate evolution after a spike at the origin.

        Args:
            t (Union[np.ndarray, float]): the time(s) to evaluate the hazard function at.

        Returns:
            Union[np.ndarray, float]: the hazard function evaluated at t.
        """
        if isinstance(t, np.ndarray):
            h = np.zeros_like(t)
            mask = t >= self.abs_ref
            h[mask] = self.firing_rate * (1 - np.exp(-(t[mask] - self.abs_ref) / self.rel_ref))
            return h
        
        if t < self.abs_ref:
            return 0
        return self.firing_rate * (1 - np.exp(-(t - self.abs_ref) / self.rel_ref))
    
    def Hazard(
            self, 
            t: Union[np.ndarray, float]
            ) -> Union[np.ndarray, float]:
        """Integrated hazard function, i.e. the integral of the firing rate evolution after a spike at the origin.

        Args:
            t (Union[np.ndarray, float]): the time(s) to evaluate the integrated hazard function at.

        Returns:
            Union[np.ndarray, float]: the integrated hazard function evaluated at t.
        """   
        if isinstance(t, np.ndarray):
            H = np.zeros_like(t)
            mask = (t >= self.abs_ref)
            H[mask] = self.firing_rate * (t[mask] - self.abs_ref + self.rel_ref * (np.exp(-(t[mask] - self.abs_ref)) - 1))
            return H    
        
        if t < self.abs_ref:
            return 0
        return self.firing_rate * (t - self.abs_ref + self.rel_ref * (np.exp(-(t - self.abs_ref)) - 1))

    def q(
            self, 
            t: Union[np.ndarray, float]
            ) -> Union[np.ndarray, float]:
        """The interspike probability density function.

        Args:
            t (Union[np.ndarray, float]): the interspike time(s) to evaluate the pdf at.

        Returns:
            Union[np.ndarray, float]: the pdf evaluated at t.
        """   
        return self.hazard(t) * np.exp(-self.Hazard(t))
    
    def Q(
            self, 
            t: Union[np.ndarray, float]
            ) -> Union[np.ndarray, float]:
        """The interspike cumulative density function.

        Args:
            t (Union[np.ndarray, float]): the interspike time(s) to evaluate the cdf at.

        Returns:
            Union[np.ndarray, float]: the cdf evaluated at t.
        """   
        return 1 - np.exp(- self.Hazard(t))
    
    def num_spikes(
            self, 
            c:int=None
            ) -> int:
        """Returns the number of spikes in the spike train, in total (if channel is None) or in the specified channel.

        Args:
            c (int, optional): the channel. Defaults to None.

        Returns:
            int: the number of spikes.
        """
        if c is None:
            return np.concatenate(self.firing_times).size
        return self.firing_times[c].size
    
    def num_unique_spikes(
            self, 
            c:Optional[int]=None
            )->int:
        """Returns the number of unique spikes in the spike train, in total (if channel is None) or in the specified channel.

        Args:
            c (int, optional): the channel. Defaults to None.

        Returns:
            int: the number of unique spikes.
        """
        if c is None:
            return np.unique(np.concatenate(self.firing_times)).size
        return np.unique(self.firing_times[c]).size
    
    def save(
            self, 
            dirname:str
            ) -> None:
        """Saves the firing times of the spike train.

        Args:
            dirname (str): the directory to save the firing times to.
        """
        os.makedirs(dirname, exist_ok=True)
        np.savez_compressed(os.path.join(dirname, "firing_times.npz"), **{f"firing_times_{c}": self.firing_times[c] for c in range(self.num_channels)})

    def load(
            self, 
            dirname:str
            ) -> None:
        """Loads the firing times of the spike train.

        Args:
            dirname (str): the directory to load the firing times from.
        """
        firing_times = np.load(os.path.join(dirname, "firing_times.npz"))
        self.firing_times = [firing_times[f"firing_times_{c}"] for c in range(self.num_channels)]

    def random(
            self, 
            res:Optional[float]=1e-2, 
            ) -> None:
        """Generates a random spike train.

        Args:
            res (Optional[float], optional): the time resolution. Defaults to 1e-2.
        """
        # 0. initialize
        self.firing_times = []

        # assume a spike occurs at time t0
        t = np.arange(0, 5*self.duration, res)
        qt = norm(self.q(t))

        for _ in trange(self.num_channels, desc="Spike Train Sampling"):
            # uniformly chose the window's origin s0 in between -4*duration and 0
            s = self.rng.uniform(-4*self.duration, 0, 1)
            while s[-1] < self.duration:
                s = np.append(s, s[-1] + self.rng.choice(t, p=qt))
            self.firing_times.append(s[(s>=0) & (s<self.duration)])

class PeriodicSpikeTrain(SpikeTrain):
    """A periodic spike train is a sequence of firing times, that repeats with a given period. It is defined by a firing rate, a hard and a soft refractory period.
    """

    def __init__(
            self, 
            num_channels:int, 
            period:float, 
            firing_rate:float, 
            abs_ref:float, 
            rel_ref:float, 
            rng:np.random.Generator=None
            ) -> None:
        """Initialize a periodic spike train.

        Args:
            num_channels (int): the number of channels / neurons.
            period (float): the period of the spike train.
            firing_rate (float): the firing rate of the spike train.
            abs_ref (float): the absolute refractory period.
            rel_ref (float): the relative refractory period, i.e., the time constant of the refractoriness to the nominal firing rate.
            rng (np.random.Generator, optional): the random number generator. Defaults to None.
        """
        super().__init__(num_channels, period, firing_rate, abs_ref, rel_ref, rng)
        self.period = period
        if self.period < self.abs_ref + self.rel_ref:
            raise ValueError("The period must be larger than the total refractory period.")
    
    def random(
            self,
            res:float=1e-2, 
            ) -> None:
        """Generates a random periodic spike train.

        Args:
            res (float, optional): the time resolution. Defaults to 1e-2.
            rmax (int, optional): the relative maximum to compute pdfs. Defaults to 5.
        """        
        # 0. initialize
        self.firing_times = []

        # implicit zero padding with rmax
        nmax = math.floor(self.duration/self.abs_ref)
        tmax = 10*self.duration
        times = np.linspace(0, tmax, math.ceil(tmax/res))
        idx = np.argmin(np.abs(times - self.duration)) # index in t corresponding to duration

        # 1. compute the forward messages 
        p = norm(np.around(self.q(times), 12))
        P = np.fft.rfft(p)
        msgf = []
        tmp = np.ones_like(P)
        for n in range(nmax):
            tmp = tmp * P
            msgf.append(norm(np.clip(np.around(np.fft.irfft(tmp), 12), 0, None)))
        msgf = np.stack(msgf, axis=0)

        # 2. compute the distribution of the number of firings
        pn = norm(msgf[:, idx])

        # 3. sample per channels
        for _ in trange(self.num_channels, desc="Periodic Spike Train Sampling"):
            # 3.a. is there any spike?
            if self.rng.binomial(1, self.Q(self.duration)) == 0:
                self.firing_times.append(np.array([]))
                continue
                
            # 3.b. given there is at least one spike, sample the exact number of spikes in the sequence
            n = self.rng.choice(nmax, p=pn) + 1
            
            # 3.c. given the exact number of spikes, sample sn, ..., s1 by backward sampling, starting from sn = self.duration
            s = np.empty(n)
            s[-1] = self.duration
            for m in range(n-2, -1, -1):
                psm = norm(msgf[m] * self.q(s[m+1] - times))
                s[m] = self.rng.choice(times, p=psm)
                
            # 3.d independently, sample the origin of the sequence s0, and cyclically shift the sequence
            s0 = self.rng.uniform(0, self.duration)
            s += s0
            self.firing_times.append(np.sort(s % self.duration))
