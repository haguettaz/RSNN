import numpy as np

from rsnn.spike_train.sampler import *


def test_get_phi():
    """
    unitest for get_phi function
    """
    
def test_sample_spike_trains():
    """
    unitest for sample_spike_trains function
    """
    period = 1000
    firing_rate = 10
    num_channels = 100
    spike_trains = sample_spike_trains(period=1000, firing_rate=10, num_channels=1000)
    
    assert len(spike_trains) == num_channels
    assert spike_trains[0].shape[0] == 10
    assert spike_trains[0].dtype == np.float32