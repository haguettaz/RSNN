import math
import os
from typing import Optional, Union

import numpy as np
from tqdm.autonotebook import trange

from .utils import norm


class SpikeTrain:
    """
    Spike train class.
    """

    def __init__(
        self,
        num_channels: int,
    ):
        """
        Args:
            num_channels (int): the number of channels / neurons.
        """

        self.num_channels = num_channels
        # self.duration = duration

        # self.firing_rate = firing_rate

        # self.abs_refractory_time = abs_refractory_time
        # self.rel_refractory_time = rel_refractory_time

        self.firing_times = [np.array([]) for _ in range(num_channels)]

    # def hazard(
    #         self,
    #         t: Union[np.ndarray, float]
    #         ) -> Union[np.ndarray, float]:
    #     """Hazard function, i.e. the firing rate evolution after a spike at the origin.

    #     Args:
    #         t (Union[np.ndarray, float]): the time(s) to evaluate the hazard function at.

    #     Returns:
    #         Union[np.ndarray, float]: the hazard function evaluated at t.
    #     """
    #     if isinstance(t, np.ndarray):
    #         h = np.zeros_like(t)
    #         mask = (t >= self.abs_refractory_time)
    #         h[mask] = self.firing_rate * (1 - np.exp(-(t[mask] - self.abs_refractory_time)/ self.rel_refractory_time))
    #         return h

    #     if t < self.abs_refractory_time:
    #         return 0
    #     return self.firing_rate * (1 - np.exp(-(t - self.abs_refractory_time) / self.rel_refractory_time))

    # def Hazard(
    #         self,
    #         t: Union[np.ndarray, float]
    #         ) -> Union[np.ndarray, float]:
    #     """Integrated hazard function, i.e. the integral of the firing rate evolution after a spike at the origin.

    #     Args:
    #         t (Union[np.ndarray, float]): the time(s) to evaluate the integrated hazard function at.

    #     Returns:
    #         Union[np.ndarray, float]: the integrated hazard function evaluated at t.
    #     """
    #     if isinstance(t, np.ndarray):
    #         H = np.zeros_like(t)
    #         mask = (t >= self.abs_refractory_time)
    #         H[mask] = self.firing_rate * (t[mask] - self.abs_refractory_time + self.rel_refractory_time * (np.exp(-(t[mask] - self.abs_refractory_time) / self.rel_refractory_time) - 1))
    #         return H

    #     if t < self.abs_refractory_time:
    #         return 0
    #     return self.firing_rate * (t - self.abs_refractory_time + self.rel_refractory_time * (np.exp(-(t - self.abs_refractory_time) / self.rel_refractory_time) - 1))

    # def f(
    #         self,
    #         t: Union[np.ndarray, float]
    #         ) -> Union[np.ndarray, float]:
    #     """The interspike probability density function.

    #     Args:
    #         t (Union[np.ndarray, float]): the interspike time(s) to evaluate the pdf at.

    #     Returns:
    #         Union[np.ndarray, float]: the pdf evaluated at t.
    #     """
    #     return self.hazard(t) * np.exp(-self.Hazard(t))

    # def F(
    #         self,
    #         t: Union[np.ndarray, float]
    #         ) -> Union[np.ndarray, float]:
    #     """The interspike cumulative density function.

    #     Args:
    #         t (Union[np.ndarray, float]): the interspike time(s) to evaluate the cdf at.

    #     Returns:
    #         Union[np.ndarray, float]: the cdf evaluated at t.
    #     """
    #     return 1 - np.exp(- self.Hazard(t))

    def num_spikes(self, c: Optional[int] = None) -> int:
        """
        Returns the number of spikes in the spike train, in total (if channel is None) or in the given channel.

        Args:
            c (int, optional): the channel. Defaults to None.

        Returns:
            int: the number of spikes.
        """
        if c is None:
            return np.concatenate(self.firing_times).size
        return self.firing_times[c].size

    def num_unique_spikes(self, c: Optional[int] = None) -> int:
        """
        Returns the number of unique spikes in the spike train, in total (if channel is None) or in the specified channel.

        Args:
            c (int, optional): the channel. Defaults to None.

        Returns:
            int: the number of unique spikes.
        """
        if c is None:
            return np.unique(np.concatenate(self.firing_times)).size
        return np.unique(self.firing_times[c]).size

    def save(self, dirname: str) -> None:
        """
        Saves the spike train firing times.

        Args:
            dirname (str): the directory to save to.
        """
        os.makedirs(dirname, exist_ok=True)
        np.savez_compressed(
            os.path.join(dirname, "firing_times.npz"),
            **{f"firing_times_{c}": self.firing_times[c] for c in range(self.num_channels)},
        )

    def load(self, dirname: str) -> None:
        """
        Loads the spike train firing times.

        Args:
            dirname (str): the directory to load from.
        """
        firing_times = np.load(os.path.join(dirname, "firing_times.npz"))
        self.firing_times = [firing_times[f"firing_times_{c}"] for c in range(self.num_channels)]

    def sample_at_random(
        self,
        duration: float,
        firing_rate: float,
        abs_refractory_time: float,
        rel_refractory_time: float,
        sampling_rate: Optional[float] = 100,
    ) -> None:
        """
        Randomly sample a spike train.

        Args:
            duration (float): the duration in [ms].
            firing_rate (float): the firing rate in [kHz].
            abs_refractory_time (float): the absolute refractory time in [ms].
            rel_refractory_time (float): the relative refractory time in [ms].
            sampling_rate (float, optional): the sampling rate in [kHz]. Defaults to 100.
        """
        self.duration = duration
        self.firing_rate = firing_rate
        self.abs_refractory_time = abs_refractory_time
        self.rel_refractory_time = rel_refractory_time
        self.sampling_rate = sampling_rate

        # 0. initialize
        self.firing_times = []

        # assume a spike occurs at time t0
        hazard = lambda t_: (t_ > 0) * self.firing_rate * (1 - np.exp(-t_ / self.rel_refractory_time))
        Hazard = (
            lambda t_: (t_ > 0)
            * self.firing_rate
            * (t_ + self.rel_refractory_time * (np.exp(-t_ / self.rel_refractory_time)))
        )
        pdf = lambda t_: hazard(t_ - self.abs_refractory_time) * np.exp(-Hazard(t_ - self.abs_refractory_time))

        intervals = np.arange(0, 2 * self.duration, 1 / self.sampling_rate)
        p = norm(pdf(intervals))

        for _ in trange(self.num_channels, desc="sample spike train"):
            # uniformly chose the window's origin s0 in between -duration and 0
            s = np.random.uniform(-self.duration, 0, 1)
            while s[-1] < self.duration:
                s = np.append(s, s[-1] + np.random.choice(intervals, p=p))
            self.firing_times.append(s[(s >= 0) & (s < self.duration)])


class PeriodicSpikeTrain(SpikeTrain):
    """A periodic spike train is a sequence of firing times, that repeats with a given period. It is defined by a firing rate, a hard and a soft refractory period."""

    def sample_at_random(
        self,
        period: float,
        firing_rate: float,
        abs_refractory_time: float,
        rel_refractory_time: float,
        sampling_rate: float = 100,  # in kHz
        # rmax: float = 10,  # might be adapted according to the period
    ) -> None:
        """
        Randomly sample a (periodic) spike train.

        Args:
            period (float): the period in [ms].
            firing_rate (float): the firing rate in [kHz].
            abs_refractory_time (float): the absolute refractory time in [ms].
            rel_refractory_time (float): the relative refractory time in [ms].
            sampling_rate (float, optional): the sampling rate in [kHz]. Defaults to 100.
        """
        self.period = period
        self.firing_rate = firing_rate
        self.abs_refractory_time = abs_refractory_time
        self.rel_refractory_time = rel_refractory_time
        self.sampling_rate = sampling_rate

        nmax = self.period // self.abs_refractory_time

        # 0. initialize
        self.firing_times = []

        hazard = lambda t_: (t_ > 0) * self.firing_rate * (1 - np.exp(-t_ / self.rel_refractory_time))
        Hazard = (
            lambda t_: (t_ > 0)
            * self.firing_rate
            * (t_ + self.rel_refractory_time * (np.exp(-t_ / self.rel_refractory_time)))
        )
        pdf = lambda t_: hazard(t_ - self.abs_refractory_time) * np.exp(-Hazard(t_ - self.abs_refractory_time))
        cdf = lambda t_: 1 - np.exp(-Hazard(t_ - self.abs_refractory_time))

        intervals = np.arange(0, 10 * self.period, 1 / self.sampling_rate)
        idx = np.argmin(np.abs(intervals - self.period))  # index in intervals corresponding to duration
        # p = norm(pdf(intervals))
        # m = (p * intervals).sum()
        # v = (p * (intervals - m) ** 2).sum()
        # print("m", m, "v", v)
        # pg = norm(np.exp(-0.5 * np.square(intervals - m) / v))
        # print("p vs g", np.sum(np.abs(p - pg)))
        # pft = np.fft.rfft(p)
        # pnft = np.power(pft, 20)
        # pn = np.around(np.fft.irfft(pnft), decimals=12)
        # mn = (pn * intervals).sum()
        # vn = (pn * (intervals - mn) ** 2).sum()
        # print("mn", mn, "vn", vn)
        # png = norm(np.exp(-0.5 * np.square(intervals - mn) / vn))
        # print("pn vs gn", np.sum(np.abs(pn - png)))

        # p = norm(pdf(intervals))

        # times = np.arange(0, rmax*self.duration, res)

        # 1. compute the forward messages
        p1 = norm(pdf(intervals))  # p1 should never change
        p1ft = np.fft.rfft(p1)
        pn = np.copy(p1)
        pnft = np.copy(p1ft)
        msgf = [p1]
        # iterate over n while P(Sn <= t) > atol
        for _ in range(nmax):
            pnft = pnft * p1ft
            pn = np.around(np.fft.irfft(pnft), decimals=12)
            msgf.append(pn)
        msgf = np.vstack(msgf)

        # 2. compute the distribution of the number of firings
        pn = norm(msgf[:, idx])
        nmax = pn.shape[0]

        # 3. sample per channels
        for _ in trange(self.num_channels, desc="sample (periodic) spike train"):
            # 3.a. is there any spike?
            if np.random.binomial(1, 1 - cdf(self.period)):
                self.firing_times.append(np.array([]))
                continue

            # 3.b. given there is at least one spike, sample the exact number of spikes in the sequence
            n = np.random.choice(np.arange(1, nmax + 1), p=pn)

            # 3.c. given the exact number of spikes, sample sn, ..., s1 by backward sampling, starting from sn = self.duration
            s = np.empty(n)
            s[-1] = self.period
            for m in range(n - 2, -1, -1):
                psm = norm(msgf[m] * pdf(s[m + 1] - intervals))
                s[m] = np.random.choice(intervals, p=psm)

            # 3.d independently, sample the origin of the sequence s0, and cyclically shift the sequence
            s0 = np.random.uniform(0, self.period)
            s += s0
            self.firing_times.append(np.sort(s % self.period))
