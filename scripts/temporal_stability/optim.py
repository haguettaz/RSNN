import argparse
import os
from random import choices

import numpy as np
from tqdm import trange

from rsnn.network.network import Network
from rsnn.neuron.neuron import Neuron
from rsnn.utils.utils import load_object_from_file, save_object_to_file

# args.num_neurons = 500

NOMINAL_THRESHOLD = 1.0
ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

FIRING_RATE = 10 # previously 150, 10
INPUT_BETA = 5.0
INPUT_DELAYS_LIM = (1.0, 50.0) # previously (1.0, 60.0)

# TEMPLATE_DICT = {"eps":1.0, "margin_min":1.0, "slope_min":0.5}
# WEIGHT_DICT = {"b":(-0.2, 0.2)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Stability Simulation")
    parser.add_argument("--num_neurons", type=int, default=500)
    parser.add_argument("--num_inputs", type=int, default=500)
    parser.add_argument("--period", type=int, default=100)
    parser.add_argument("--wb", type=int, default=20)
    parser.add_argument("--l_reg", type=int, default=0)
    parser.add_argument("--slope_min", type=int, default=2)
    parser.add_argument("--exp_idx", type=int)
    args = parser.parse_args()

    print("Parameters", flush=True)
    print("num_neurons:", args.num_neurons, flush=True)
    print("num_inputs:", args.num_inputs, flush=True)
    print("period:", args.period, flush=True)
    print("wb:", args.wb, flush=True)
    print("l_reg:", args.l_reg)
    print("slope_min:", args.slope_min, flush=True)
    print("exp_idx:", args.exp_idx, flush=True)
    print()

    all_spike_trains = load_object_from_file(os.path.join("spike_trains", f"spike_trains_{args.period}_{FIRING_RATE}.pkl"))
    template_dict = {"eps":1.0, "margin_min":1.0, "slope_min":args.slope_min}
    weight_dict = {"b":(-args.wb/100, args.wb/100)}
    if args.l_reg > 0:
        weight_dict["l"] = args.l_reg

    spike_trains = choices(all_spike_trains, k=args.num_neurons)
    save_object_to_file(spike_trains, os.path.join(f"{args.num_neurons}_{args.num_inputs}_{args.period}_{args.wb}_{args.slope_min}", "spike_trains", f"experiment_{args.exp_idx}_spike_trains.pkl"))

    neurons = [Neuron(
            neuron_idx, args.num_inputs, INPUT_BETA, NOMINAL_THRESHOLD, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY
        ) for neuron_idx in range(args.num_neurons)]
    for neuron in neurons:        
        neuron.sources = np.random.choice(args.num_neurons, size=args.num_inputs, replace=True)
        neuron.delays = np.random.uniform(*INPUT_DELAYS_LIM, size=args.num_inputs)
        firing_times = spike_trains[neuron.idx].copy()
        input_firing_times = [(spike_trains[s] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)]
        neuron.optimize_weights(firing_times, input_firing_times, args.period, template_dict, weight_dict)
        if neuron.status != "optimal":
            print(f"Neuron {neuron.idx} failed to optimize weights")
        # save_object_to_file(neuron, os.path.join(f"{args.num_neurons}_{args.num_inputs}_{args.period}_{args.wb}_{args.slope_min}", "optim", "neurons", f"experiment_{args.exp_idx}_neuron_{neuron_idx}.pkl"))
    
    # neurons = [load_object_from_file(os.path.join(f"{args.num_neurons}_{args.num_inputs}_{args.period}_{args.wb}_{args.slope_min}", "optim", "neurons", f"experiment_{args.exp_idx}_neuron_{neuron_idx}.pkl")) for neuron_idx in range(args.num_neurons)]
    network = Network(neurons)
    if args.l_reg > 0:
        dir_path = f"{args.num_neurons}_{args.num_inputs}_{args.period}_{args.wb}_{args.slope_min}_{args.l_reg}"
    else:
        dir_path = f"{args.num_neurons}_{args.num_inputs}_{args.period}_{args.wb}_{args.slope_min}"
    save_object_to_file(network, os.path.join(dir_path, "optim", "networks", f"experiment_{args.exp_idx}_network.pkl"))


# num_inputs = 300
# network_idx = 4
# spike_trains = choices(REF_FIRING_TIMES, k=args.num_neurons)
# save_object_to_file(spike_trains, os.path.join("spike_trains", f"spike_trains_{num_inputs}_{network_idx}.pkl"))

# for neuron_idx in trange(args.num_neurons):        
#     neuron = Neuron(
#         neuron_idx, num_inputs, INPUT_BETA, NOMINAL_THRESHOLD, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY
#     )
#     neuron.sources = np.random.choice(args.num_neurons, size=num_inputs, replace=True)
#     neuron.delays = np.random.uniform(*INPUT_DELAYS_LIM, size=num_inputs)
#     firing_times = spike_trains[neuron_idx].copy()
#     input_firing_times = [(spike_trains[s] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)]
#     neuron.optimize_weights(firing_times, input_firing_times, PERIOD, TEMPLATE_DICT, WEIGHT_DICT)
#     if neuron.status != "optimal":
#         print(f"Neuron {neuron_idx} of network {network_idx} with {num_inputs} inputs failed to optimize weights")
#     save_object_to_file(neuron, os.path.join("neurons", f"network_{num_inputs}_{network_idx}_neuron_{neuron_idx}.pkl"))

# neurons = [load_object_from_file(os.path.join("neurons", f"network_{num_inputs}_{network_idx}_neuron_{neuron_idx}.pkl")) for neuron_idx in range(args.num_neurons)]
# network = Network(neurons)
# save_object_to_file(network, os.path.join("networks", f"network_{num_inputs}_{network_idx}.pkl"))