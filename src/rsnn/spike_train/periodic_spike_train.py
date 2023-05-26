from typing import List, Optional

import numpy as np

from .spike_train import MultiChannelSpikeTrain, SpikeTrain


class PeriodicSpikeTrain(SpikeTrain):
    """
    Periodic spike train class.
    """

    def __init__(
        self,
        period: float,
        firing_times: Optional[np.ndarray] = None,
    ):
        """
        Initialize a periodic spike train.

        Args:
            period (float): The period of the spike train in [ms].
            firing_times (Optional[np.ndarray]): The firing times of the spike train in [ms].
        """
        if period <= 0:
            raise ValueError("The period must be positive.")
        
        self.period = period
        
        if firing_times is None:
            self.firing_times = np.array([])
        else:
            self.firing_times = firing_times

    @property
    def num_spikes(self) -> int:
        """
        Returns the number of spikes in the spike train.
        
        Returns:
            (int): The number of spikes in the spike train.
        """
        return super().num_spikes

class MultiChannelPeriodicSpikeTrain(MultiChannelSpikeTrain):
    def __init__(
        self,
        period:float,
        num_channels: int,
        firing_times: Optional[List[np.ndarray]] = None,
    ):
        """
        Initialize a multi-channel periodic spike train.

        Args:
            period (float): The period in [ms].
            num_channels (int): The number of channels / neurons.
            firing_times (list): a list of arrays of firing times in [ms]. Defaults to None.

        Raises:
            ValueError: If number of channels is not positive.
            ValueError: If period is not positive.
            ValueError: If number of firing times does not match number of channels.
        """
        if period <= 0:
            raise ValueError("The period must be positive.")
        self.period = period
        
        if num_channels <= 0:
            raise ValueError("Number of channels must be positive.")
        self.num_channels = num_channels

        if firing_times is None:
            self.spike_trains = [PeriodicSpikeTrain(period) for _ in range(num_channels)]
        else:
            if len(firing_times) != num_channels:
                raise ValueError("Number of firing times must match number of channels.")
            self.spike_trains = [PeriodicSpikeTrain(period, firing_times[i]) for i in range(num_channels)]

    @property
    def num_spikes(self) -> int:
        """
        Return the number of spikes in the multi-channel spike train.

        Returns:
            (int): The number of spikes.
        """
        return super().num_spikes