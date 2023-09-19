import os
import random

from tqdm import trange

from rsnn.network.network import Network
from rsnn.neuron.neuron_old import Synapse
from rsnn.spike_train.generator import PeriodicSpikeTrainGenerator

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
spike_train_generator = PeriodicSpikeTrainGenerator(FIRING_RATE, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY)

for period in range(100, 600, 100):
    spike_trains = spike_train_generator.rand(period, NUM_NEURONS)  
    for n_ in network.neurons:
        n_.spike_train = spike_trains.spike_trains[n_.idx]
        
    for i in range(100, 250):
        for num_unique_neurons in [NUM_SYNAPSES // 100, NUM_SYNAPSES // 10, NUM_SYNAPSES]:
            # if os.path.exists(f"networks/{period}_{num_unique_neurons}_{i}.pkl"):
            #     continue
            neuron = random.choice(network.neurons)
            neuron_pool = random.sample([n_ for n_ in network.neurons if n_ is not neuron], num_unique_neurons - 1) + [neuron]
            neuron.synapses =[Synapse(
                idx,
                neuron_pool[idx % num_unique_neurons],
                SYNAPSE_BETA,
                random.uniform(*SYNAPSE_DELAY_LIM))
                for idx in range(NUM_SYNAPSES)]
            neuron.init_template_single(EPS, GAP, SLOPE)
            neuron.memorize(WEIGHTS_LIM, "l1")
            network.save_to_file(f"networks/{period}_{num_unique_neurons}_{i}.pkl")
            neuron.synapses = []

# import argparse
# import pickle

# import numpy as np

# from rsnn.rsnn.generator import NetworkGenerator
# from rsnn.spike_train.generator import PeriodicSpikeTrainGenerator

# FIRING_RATE = 0.1
# ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY = 5.0, 5.0

# NUM_SYNAPSES = 1000  # NUMBER OF SYNAPSES
# SYNAPSE_DELAYS_LIM = (1.0, 60.0)
# SYNAPSE_WEIGHTS_LIM = (-0.1, 0.1)
# SYNAPSE_BETA = 5.0
# NOMINAL_THRESHOLD = 1.0

# EPS, GAP, SLOPE = 1.0, 1.0, 0.5

# # TODO: 100 realizations, for each realization, pick one neuron at random, optimize its weights with sparsity regularization. Do it for various combinations of num neurons and period


# # num synapses: 1000, but try 500
# # num neurons (or expected influence from one neuron to another 1 / num neurons): 1, 10, 100
# # period: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
# # expected number of connections between two neurons 

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Memorization Simulation")
#     parser.add_argument("--num_neurons", type=int, default=100)
#     parser.add_argument("--period", type=int, default=100, help="in ms")
#     args = parser.parse_args()

#     print("Parameters", flush=True)
#     print(f"  num_neurons: {args.num_neurons}", flush=True)
#     print(f"  period in [ms]: {args.period}", flush=True)
#     print()

#     network_generator = NetworkGenerator(
#         args.num_neurons,
#         NOMINAL_THRESHOLD,
#         ABSOLUTE_REFRACTORY,
#         RELATIVE_REFRACTORY,
#         NUM_SYNAPSES,
#         SYNAPSE_BETA,
#         SYNAPSE_DELAYS_LIM
#     )
#     spike_train_generator = PeriodicSpikeTrainGenerator(FIRING_RATE, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY)

#     results = []
#     while len(results) < 1000:
#         network = network_generator.rand()
#         multi_channel_spike_train = spike_train_generator.rand(args.period, args.num_neurons)
#         network.memorize(multi_channel_spike_train, SYNAPSE_WEIGHTS_LIM, EPS, GAP, SLOPE, "l1")
#         for neuron in network.neurons:
#             results.append({
#                 "num_active_synapses": neuron.num_active_synapses,
#                 "status": neuron.prob.status
#                 })

#     with open(f"results/sparse_learning_{args.num_neurons}_{args.period}.pkl", "wb") as f:
#         pickle.dump(results, f)