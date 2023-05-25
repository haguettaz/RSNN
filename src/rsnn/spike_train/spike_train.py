import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class SpikeTrain:
    """
    Spike train class.
    """

    def __init__(
        self,
        firing_times: Optional[np.ndarray] = None,
    ):
        """
        Initialize a spike train.

        Args:
            firing_times (Optional[np.ndarray]): the firing times of the spike train in [ms]. Default to None.
        """
        if firing_times is None:
            self.firing_times = np.array([])
        else:
            self.firing_times = firing_times

    @property
    def config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the spike train.

        Returns:
            (Dict[str, np.ndarray]): the spike train configuration.
        """
        return {"firing_times": self.firing_times}

    def reset(self):
        """
        Reset the spike train.
        """
        self.firing_times = np.array([])

    def save_to_file(self, filename: str):
        """
        Save the spike train configuration to a file.

        Args:
            filename (str): the name of the file to save to.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the configuration to a file
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.config, f)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error saving network configuration: {e}")

    def load_from_file(self, filename: str):
        """
        Load the spike train from a file.

        Args:
            filename (str): the name of the file to load from.

        Raises:
            FileNotFoundError: if the file does not exist
            ValueError: if number of neurons does not match
            ValueError: if error loading the file
        """

        # Check if the file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")

        # Load the configuration from the file
        try:
            with open(filename, "rb") as f:
                config = pickle.load(f)
        except (FileNotFoundError, PermissionError) as e:
            raise ValueError(f"Error loading network configuration: {e}")

        # Set the spike train firing times
        self.firing_times = config["firing_times"]

    @property
    def num_spikes(self) -> int:
        """
        Returns the number of spikes in the spike train.

        Returns:
            (int): the number of spikes.
        """
        return self.firing_times.size

    def append(self, t: float):
        """
        Add a firing time to the spike train.

        Args:
            t (float): the firing time in [ms].
        """
        self.firing_times = np.append(self.firing_times, t)

    def jitter(self, std:float):
        """
        Add jitter to the spike train.

        Args:
            std (float): the standard deviation of the jitter in [ms].
        """
        self.firing_times += np.random.normal(0, std, self.firing_times.shape)

    def copy(self):
        """
        Returns a copy of the spike train.

        Returns:
            (SpikeTrain): a copy of the spike train.
        """
        return SpikeTrain(self.firing_times.copy())
        

class MultiChannelSpikeTrain:
    """
    Multi-channel spike train class.
    """

    def __init__(
        self,
        num_channels: int,
        firing_times: Optional[List[np.ndarray]] = None,
    ):
        """
        Initialize a multi-channel spike train.

        Args:
            num_channels (int): the number of channels / neurons.
            firing_times (Optional[List[np.ndarray]]): the firing times of the spike train in [ms]. Default to None.

        Raises:
            ValueError: if number of channels is not positive.
            ValueError: if number of firing times does not match number of channels.
        """
        if num_channels <= 0:
            raise ValueError("Number of channels must be positive.")
        self.num_channels = num_channels

        if firing_times is None:
            self.spike_trains = [SpikeTrain() for _ in range(num_channels)]
        else:
            if len(firing_times) != num_channels:
                raise ValueError("Number of firing times must match number of channels.")
            self.spike_trains = [SpikeTrain(firing_times[i]) for i in range(num_channels)]

    def reset(self):
        """
        Reset the spike train.
        """
        for spike_train in self.spike_trains:
            spike_train.reset()

    @property
    def num_spikes(self) -> int:
        """
        Returns the number of spikes.

        Returns:
            (int): the number of spikes.
        """
        return sum([spike_train.num_spikes for spike_train in self.spike_trains])

    def save_to_file(self, filename: str):
        """
        Save the multi-channel spike train configuration to a file.

        Args:
            filename (str): the name of the file to save to.
        """
        config = [spike_train.config for spike_train in self.spike_trains]

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the configuration to a file
        try:
            with open(filename, "wb") as f:
                pickle.dump(config, f)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error saving network configuration: {e}")

    def load_from_file(self, filename: str):
        """
        Load the multi-channel spike train from a file.

        Args:
            filename (str): the name of the file to load from.

        Raises:
            FileNotFoundError: if the file does not exist
            ValueError: if number of neurons does not match
            ValueError: if error loading the file
        """

        # Check if the file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")

        # Load the configuration from the file
        try:
            with open(filename, "rb") as f:
                configs = pickle.load(f)
        except (FileNotFoundError, PermissionError) as e:
            raise ValueError(f"Error loading network configuration: {e}")

        if len(configs) != self.num_channels:
            raise ValueError(f"Number of channels does not match: {len(configs)} != {self.num_channels}")

        # Set the spike train firing times
        for spike_train, spike_train_config in zip(self.spike_trains, configs):
            spike_train.firing_times = spike_train_config["firing_times"]
