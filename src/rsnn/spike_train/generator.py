from typing import Optional, Union

import numpy as np
from tqdm.autonotebook import trange

from ..utils.utils import norm
from .distribution import LinearDistribution, SurvivalDistribution
from .periodic_spike_train import MultiChannelPeriodicSpikeTrain, PeriodicSpikeTrain
from .spike_train import MultiChannelSpikeTrain, SpikeTrain


class SpikeTrainGenerator:
    def __init__(self, firing_rate: float, abs_refractory_time: float, rel_refractory_time: float):
        """
        Initialize the spike train generator.
        Setup the survival distribution.

        Args:
            firing_rate (float): The firing rate in [kHz].
            abs_refractory_time (float): The absolute refractory time in [ms].
            rel_refractory_time (float): The relative refractory time in [ms].
        """
        self.survival_dist = SurvivalDistribution(firing_rate, abs_refractory_time, rel_refractory_time)

    def rand(self, duration: float, num_channels: Optional[int] = None) -> Union[SpikeTrain, MultiChannelSpikeTrain]:
        """
        Returns a SpikeTrain or a MultiChannelSpikeTrain of duration `duration` [ms].

        Args:
            duration (float): The duration.
            num_channels (int, optional): The number of channels / neurons. If None, returns a `SpikeTrain`, otherwise a `MultiChannelSpikeTrain`. Defaults to None.

        Raises:
            ValueError: If the duration is not positive.

        Returns:
            (SpikeTrain or MultiChannelSpikeTrain): a spike train or a multi-channel spike train.
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
            return SpikeTrain(firing_times)

        # Multi-channel spike train
        firing_times = []
        for _ in trange(num_channels, desc="Sampling multi-channel spike train"):
            # Uniformly chose the first spike in between -duration and 0
            s = np.random.uniform(-duration, 0.0, 1)

            # Sample firing times until the end of the window is reached
            while s[-1] < duration:
                s = np.append(s, s[-1] + self.survival_dist.rvs(1))
            firing_times.append(s[(s >= 0) & (s < duration)])
        return MultiChannelSpikeTrain(num_channels, firing_times)


class PeriodicSpikeTrainGenerator:
    def __init__(self, firing_rate: float, abs_refractory_time: float, rel_refractory_time: float):
        """
        Initialize the spike train generator.
        Setup the survival distribution and the linear distribution for the approximation.

        Args:
            firing_rate (float): The firing rate in [kHz].
            abs_refractory_time (float): The absolute refractory time in [ms].
            rel_refractory_time (float): The relative refractory time in [ms].
        """
        self.survival_dist = SurvivalDistribution(firing_rate, abs_refractory_time, rel_refractory_time)
        self.linear_dist = LinearDistribution()

    def rand(self, period, num_channels=None):
        """
        Returns a PeriodicSpikeTrain or a MultiChannelPeriodicSpikeTrain of period `period` [ms].

        Args:
            period (float): The period.
            num_channels (int, optional): The number of channels / neurons. If None, returns a `PeriodicSpikeTrain`, otherwise a `MultiChannelPeriodicSpikeTrain`. Defaults to None.

        Raises:
            ValueError: If the period is not positive.

        Returns:
            (PeriodicSpikeTrain or MultiChannelPeriodicSpikeTrain): a spike train or a multi-channel spike train.
        """
        if period <= 0:
            raise ValueError("The period must be positive.")

        # Approximate the maximum number of spikes
        nmax = int(period / self.survival_dist.ppf(1e-9))

        # Approximate the step size for linear approximation of the pdfs
        step = self.survival_dist.ppf(0.5) / 100

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
                return PeriodicSpikeTrain(period)

            # Sample the firing times by backward sampling
            # Note: 0 and period are assumed to be a (unique) firing time
            s = np.full(n, period, dtype=float)
            for m in range(n - 2, -1, -1):
                # Note: The first and last elements of psm are always zero
                psm = norm(fmus[m] * self.survival_dist.pdf(s[m + 1] - z))

                # Sample the discrete bin containing the current firing time
                bin = np.random.choice(psm.size, p=psm)

                # Sample in the continuous interval using linear interpolation
                # Note: The resolution is good enough to neglect higher order terms
                self.linear_dist.a = z[bin - 1]
                self.linear_dist.b = z[bin + 1]
                self.linear_dist.slope = (psm[bin + 1] - psm[bin - 1]) / (2 * step)
                s[m] = self.linear_dist.rvs()

            # Ramdomly shift the firing times cyclically
            firing_times = np.sort((s + np.random.uniform(0, period)) % period)
            return PeriodicSpikeTrain(period, firing_times)

        # Multi-channel spike train
        firing_times = []
        for _ in trange(num_channels, desc="Sampling multi-channel spike train"):
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

                # Sample the discrete bin containing the current firing time
                bin = np.random.choice(psm.size, p=psm)

                # Sample in the continuous interval using linear interpolation
                # Note: The resolution is good enough to neglect higher order terms
                self.linear_dist.a = z[bin - 1]
                self.linear_dist.b = z[bin + 1]
                self.linear_dist.slope = (psm[bin + 1] - psm[bin - 1]) / (2 * step)
                s[m] = self.linear_dist.rvs()

            # Ramdomly shift the firing times cyclically
            firing_times.append(np.sort((s + np.random.uniform(0, period)) % period))

        return MultiChannelPeriodicSpikeTrain(period, num_channels, firing_times)
