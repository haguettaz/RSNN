from typing import Optional, Union, List

import numpy as np
from tqdm.autonotebook import trange

from ..utils.math import norm
from .distribution import SurvivalDistribution


class SpikeTrainGenerator:
    def __init__(self, firing_rate: float, absolute_refractory: float, relative_refractory: float):
        """
        Initialize the spike train generator.
        Setup the survival distribution.

        Args:
            firing_rate (float): The firing rate in [kHz].
            absolute_refractory (float): The absolute refractory time in [ms].
            relative_refractory (float): The relative refractory time in [ms].
        """
        self.survival_dist = SurvivalDistribution(firing_rate, absolute_refractory, relative_refractory)

    def rand(self, duration: float, num_channels: Optional[int] = None) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Returns a single- or multi-channel spike train with duration `duration`.

        Args:
            duration (float): The duration in [ms].
            num_channels (int, optional): The number of channels / neurons. If None, returns a single-channel spike train, otherwise a multi-channel spike train. Defaults to None.

        Raises:
            ValueError: If the duration is not positive.

        Returns:
            (np.ndarray or List[np.ndarray]): a spike train or a multi-channel spike train.
        """
        if duration <= 0:
            raise ValueError("The duration must be positive.")

        # Single-channel spike train by default
        if num_channels is None:
            # Uniformly chose the first spike in between -duration and 0
            s = np.random.uniform(-duration, 0.0, 1)

            # Sample firing times until the end of the window is reached
            while s[-1] < duration:
                s = np.append(s, s[-1] + self.survival_dist.rvs(1))
            firing_times = s[(s >= 0) & (s < duration)]
            return firing_times

        # Multi-channel spike train
        firing_times = []
        for _ in trange(num_channels, desc="Multi-channel spike train sampling"):
            # Uniformly chose the first spike in between -duration and 0
            s = np.random.uniform(-duration, 0.0, 1)

            # Sample firing times until the end of the window is reached
            while s[-1] < duration:
                s = np.append(s, s[-1] + self.survival_dist.rvs(1))
            firing_times.append(s[(s >= 0) & (s < duration)])
        return firing_times


class PeriodicSpikeTrainGenerator:
    def __init__(self, firing_rate: float, absolute_refractory: float, relative_refractory: float):
        """
        Initialize the spike train generator.
        Setup the survival distribution and the linear distribution for the approximation.

        Args:
            firing_rate (float): The firing rate in [kHz].
            absolute_refractory (float): The absolute refractory time in [ms].
            relative_refractory (float): The relative refractory time in [ms].
        """
        self.firing_rate = firing_rate
        self.absolute_refractory = absolute_refractory
        self.relative_refractory = relative_refractory

        self.survival_dist = SurvivalDistribution(firing_rate, absolute_refractory, relative_refractory)
        # self.linear_dist = LinearDistribution()

    def rand(self, period, num_channels=None) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Returns a single- or multi-channel periodic spike train with period `period`.

        Args:
            period (float): The period in [ms].
            num_channels (int, optional): The number of channels / neurons. If None, returns a single-channel spike train, otherwise a multi-channel spike train. Defaults to None.

        Raises:
            ValueError: If the period is not positive.

        Returns:
            (np.ndarray or List[np.ndarray]): a single- or multi-channel periodic spike train.
        """
        if period <= 0:
            raise ValueError("The period must be positive.")

        # Approximate the maximum number of spikes
        nmax = int(period / self.survival_dist.ppf(1e-9))

        # Approximate the step size for linear approximation of the pdfs
        step = self.survival_dist.ppf(0.5) / 1000

        # Create the time vector, should be even for the FFT-based convolution
        # Note: The maximum time is set to 5 standard deviations above the mean, using CLT for the sum of nmax iid random variables
        z = np.arange(0, nmax * self.survival_dist.mean() + 5 * self.survival_dist.std() * np.sqrt(nmax), step)
        if z.size % 2:
            z.resize(z.size - 1)  # Remove the last element of z in place

        # Get the index of the period
        idx = np.argmin(np.abs(z - period))

        # Init DFT of the pdf for forward filtering in the Fourier domain
        # Note: implicit zero-padding for linear convolution
        pft = np.fft.rfft(norm(self.survival_dist.pdf(z)))

        # Compute all necessary forward messages (up to nmax)
        fmus = []
        pnft = np.ones_like(pft)
        for n in range(1, nmax + 1):  # Note: The number of spikes is upper bounded
            pnft = pnft * pft
            pn = np.clip(np.fft.irfft(pnft), 0, None)
            fmus.append(pn)
        fmus = np.vstack(fmus)

        # Keep only the relevant parts, i.e. the parts before the period (included)
        z = z[: idx + 1]
        fmus = fmus[:, : idx + 1]

        # Single-channel spike train by default
        if num_channels is None:
            # Sample the number of spikes
            pn = np.full(nmax + 1, 1 - self.survival_dist.cdf(period))
            pn[1:] = norm(fmus[:, -1]) * self.survival_dist.cdf(period)
            n = np.random.choice(pn.size, p=pn)

            if n == 0:
                return np.array([])

            # Sample the firing times by backward sampling
            # Note: 0 and period are assumed to be a (unique) firing time
            s = np.full(n, period, dtype=float)
            for m in range(n - 2, -1, -1):
                # Note: The first and last elements of psm are always zero
                psm = norm(fmus[m] * self.survival_dist.pdf(s[m + 1] - z))
                s[m] = np.random.choice(z, p=psm)

            # Ramdomly shift the firing times cyclically
            firing_times = np.sort((s + np.random.uniform(0, period)) % period)
            return firing_times

        # Multi-channel spike train
        firing_times = []
        for _ in trange(num_channels, desc="Multi-channel periodic spike train sampling"):
            # Sample the number of spikes
            pn = np.full(nmax + 1, 1 - self.survival_dist.cdf(period))
            pn[1:] = norm(fmus[:, -1]) * self.survival_dist.cdf(period)
            n = np.random.choice(pn.size, p=pn)

            if n == 0:
                firing_times.append(np.array([]))
                continue

            # Sample the firing times by backward sampling
            # Note: 0 and period are assumed to be a (unique) firing time
            s = np.full(n, period, dtype=float)
            for m in range(n - 2, -1, -1):
                # Note: The first and last elements of psm are always zero
                psm = norm(fmus[m] * self.survival_dist.pdf(s[m + 1] - z))
                s[m] = np.random.choice(z, p=psm)
            
            # Ramdomly shift the firing times cyclically
            firing_times.append(np.sort((s + np.random.uniform(0, period)) % period))

        return firing_times
