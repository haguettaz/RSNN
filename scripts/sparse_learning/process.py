import os
import random

import matplotlib
# import cvxpy as cp
# import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import trange

from rsnn.network.network import Network
from rsnn.neuron.neuron_old import Synapse
from rsnn.spike_train.generator import PeriodicSpikeTrainGenerator
from rsnn.spike_train.periodic_spike_train import PeriodicSpikeTrain

NUM_NEURONS = 2000
NOMINAL_THRESHOLD = 1.0
ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

NUM_SYNAPSES = 1000
SYNAPSE_BETA = 5.0
SYNAPSE_DELAY_LIM = (1.0, 60.0)

FIRING_RATE = 0.1

EPS, GAP, SLOPE = 1.0, 1.0, 0.5
WEIGHTS_LIM = (-0.1, 0.1)

network = Network(NUM_NEURONS, NOMINAL_THRESHOLD, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY)

list_of_dict = []
for period in range(100, 600, 100):
    for i in range(100):
        for num_unique_neurons in [NUM_SYNAPSES // 100, NUM_SYNAPSES // 10, NUM_SYNAPSES]:
            if not os.path.exists(f"networks/{period}_{num_unique_neurons}_{i}.pkl"):
                print(f"networks/{period}_{num_unique_neurons}_{i}.pkl does not exist")
                continue
            network.load_from_file(f"networks/{period}_{num_unique_neurons}_{i}.pkl")
            for neuron in network.neurons:
                neuron.spike_train = PeriodicSpikeTrain(period, neuron.spike_train.firing_times)
            network.save_to_file(f"networks/{period}_{num_unique_neurons}_{i}.pkl")