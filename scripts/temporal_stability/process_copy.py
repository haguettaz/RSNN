import os

import numpy as np
import pandas as pd
from tqdm import trange

from rsnn.network.network import Network
from rsnn.spike_train.generator import SpikeTrainGenerator
from rsnn.spike_train.measure import multi_channel_correlation, single_channel_correlation
from rsnn.utils.math import dist_mod, mod
from rsnn.utils.utils import load_object_from_file, save_object_to_file

list_of_dicts = []

num_neurons = 500
num_inputs = 500
period = 200
wb = 20
slope_min = 50

# spike_trains = load_object_from_file(os.path.join(f"{num_neurons}_{num_inputs}_{period}_{wb}_{slope_min}", "spike_trains", f"experiment_0_spike_trains.pkl"))
# print(spike_trains[0])

for exp_idx in range(100):
    for std_threshold in [5, 10, 15, 20]:
        try:
            network = load_object_from_file(os.path.join(f"{num_neurons}_{num_inputs}_{period}_{wb}_{slope_min}", "sim", "networks", f"experiment_{exp_idx}_noise_{std_threshold}_network.pkl"))
            for neuron in network.neurons:
                if neuron.status != "optimal":
                    print(f"Error in optimization: {exp_idx}")
                    continue
            
            for cycle, (precision, recall) in enumerate(zip(network.precision, network.recall)):
                list_of_dicts.append(
                    {
                        "num_neurons": num_neurons,
                        "period": period,
                        "num_inputs": num_inputs,
                        "wb": wb,
                        "slope_min": slope_min,
                        "exp_idx": exp_idx,
                        "std_threshold": std_threshold,
                        "cycle": cycle,
                        "precision": precision,
                        "recall": recall,
                    }
                )
        except:
            print(f"Error: {exp_idx}, {std_threshold}")

df = pd.DataFrame(list_of_dicts)
df.to_csv(os.path.join(f"{num_neurons}_{num_inputs}_{period}_{wb}_{slope_min}", "sim", "results.csv"), index=False)
