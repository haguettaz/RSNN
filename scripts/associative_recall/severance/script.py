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

PERIOD, FIRING_RATE = 100, 10

DT = 0.1
STD_THRESHOLD, STD_JITTER = 0.2, 1.0

# load neurons and spike trains
ref_firing_times = load_object_from_file(os.path.join("spike_trains", f"firing_times_{PERIOD}_{FIRING_RATE}.pkl"))
network = load_object_from_file(os.path.join("network", f"network.pkl"))
network.reset_firing_times()

# inject noise via noisy neurons
spike_train_generator = SpikeTrainGenerator(FIRING_RATE, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY) # for random noise
for neuron in network.neurons[-NUM_NOISY_NEURONS:]:
    neuron.firing_times = spike_train_generator.rand(9*PERIOD)

# phase 1: driven neurons are jittered ideal spike trains 1 for 1 period
for neuron in network.neurons[:NUM_DRIVEN_NEURONS]:
    neuron.firing_times = np.concatenate([neuron.firing_times, np.random.normal(ref_firing_times[neuron.idx], STD_JITTER)])
network.sim(PERIOD, DT, STD_THRESHOLD, range(NUM_DRIVEN_NEURONS, NUM_NEURONS-NUM_NOISY_NEURONS))

# phase 2: stop neuron driving for 2 periods
network.sim(3*PERIOD, DT, STD_THRESHOLD, range(NUM_NEURONS-NUM_NOISY_NEURONS))

# phase 3: driven neurons are jittered ideal spike trains 2 for 1 period
for neuron in network.neurons[:NUM_DRIVEN_NEURONS]:
    neuron.firing_times = np.concatenate([neuron.firing_times, np.random.normal(ref_firing_times[neuron.idx + NUM_NEURONS] + 3*PERIOD, STD_JITTER)])
network.sim(4*PERIOD, DT, STD_THRESHOLD, range(NUM_DRIVEN_NEURONS, NUM_NEURONS-NUM_NOISY_NEURONS))

# phase 4: stop neuron driving for 2 periods
network.sim(6*PERIOD, DT, STD_THRESHOLD, range(NUM_NEURONS-NUM_NOISY_NEURONS))

# phase 5: driven neurons are jittered ideal spike trains 3 for 1 period
for neuron in network.neurons[:NUM_DRIVEN_NEURONS]:
    neuron.firing_times = np.concatenate([neuron.firing_times, np.random.normal(ref_firing_times[neuron.idx + 2*NUM_NEURONS] + 6*PERIOD, STD_JITTER)])
network.sim(7*PERIOD, DT, STD_THRESHOLD, range(NUM_DRIVEN_NEURONS, NUM_NEURONS-NUM_NOISY_NEURONS))

# phase 6: stop neuron driving for 2 periods
network.sim(9*PERIOD, DT, STD_THRESHOLD, range(NUM_NEURONS-NUM_NOISY_NEURONS))

save_object_to_file(network, os.path.join("network", f"network_sim.pkl"))