import os

import numpy as np

from rsnn.spike_train.generator import SpikeTrainGenerator
from rsnn.utils.utils import load_object_from_file, save_object_to_file

NUM_NEURONS = 1000

NUM_DRIVEN_NEURONS = 200
NUM_NOISY_NEURONS = 10

NOMINAL_THRESHOLD = 1.0
ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

NUM_INPUTS = 500

PERIOD, FIRING_RATE = 200, 10

DT = 0.1
STD_THRESHOLD, STD_JITTER = 0.2, 1.0

# load neurons and spike trains
ref_firing_times = load_object_from_file(os.path.join("spike_trains", f"firing_times_{PERIOD}_{FIRING_RATE}.pkl"))
network = load_object_from_file(os.path.join("network", f"network.pkl"))
network.reset_firing_times()

# inject noise via noisy neurons
spike_train_generator = SpikeTrainGenerator(FIRING_RATE, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY) # for random noise
for neuron in network.neurons[-NUM_NOISY_NEURONS:]:
    neuron.firing_times = spike_train_generator.rand(3*PERIOD)

# phase 1: driven neurons are jittered ideal spike trains
for neuron in network.neurons[:NUM_DRIVEN_NEURONS]:
    neuron.firing_times = np.random.normal(ref_firing_times[neuron.idx], STD_JITTER)
network.sim(PERIOD, DT, STD_THRESHOLD, range(NUM_DRIVEN_NEURONS, NUM_NEURONS-NUM_NOISY_NEURONS))

# phase 2: stop neuron driving
network.sim(3*PERIOD, DT, STD_THRESHOLD, range(NUM_NEURONS-NUM_NOISY_NEURONS))

save_object_to_file(network, os.path.join("network", f"network_sim.pkl"))