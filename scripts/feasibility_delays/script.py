import os
import random

from tqdm import trange

from rsnn.network.network import Network
from rsnn.neuron.neuron_old import Synapse
from rsnn.spike_train.generator import PeriodicSpikeTrainGenerator

NUM_NEURONS = 5000
NOMINAL_THRESHOLD = 1.0
ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

NUM_SYNAPSES = 500
SYNAPSE_BETA = 5.0
SYNAPSE_DELAY_LIM = (1.0, 60.0)

FIRING_RATE = 0.1

EPS, GAP, SLOPE = 1.0, 1.0, 0.5
WEIGHTS_LIM = (-0.2, 0.2)

network = Network(NUM_NEURONS, NOMINAL_THRESHOLD, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY)
spike_train_generator = PeriodicSpikeTrainGenerator(FIRING_RATE, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY)

for period in range(100, 1200, 100):
    spike_trains = spike_train_generator.rand(period, NUM_NEURONS)  
    for n_ in network.neurons:
        n_.spike_train = spike_trains.spike_trains[n_.idx]

    for i in trange(100):
        for num_unique_neurons in [NUM_SYNAPSES // 100, NUM_SYNAPSES // 10, NUM_SYNAPSES]:
            if os.path.exists(f"networks/{period}_{num_unique_neurons}_{i}.pkl"):
                continue
            neuron = random.choice(network.neurons)
            neuron_pool = random.sample([n_ for n_ in network.neurons if n_ is not neuron], num_unique_neurons)
            neuron.synapses =[Synapse(
                idx,
                neuron_pool[idx % num_unique_neurons],
                SYNAPSE_BETA,
                random.uniform(*SYNAPSE_DELAY_LIM))
                for idx in range(NUM_SYNAPSES)]
            neuron.init_template_single(EPS, GAP, SLOPE)
            neuron.memorize(WEIGHTS_LIM)
            # print("period:", period, "num unique neurons:", num_unique_neurons, "status:", neuron.opt_status)
            network.save_to_file(f"networks/{period}_{num_unique_neurons}_{i}.pkl")
            neuron.synapses = []