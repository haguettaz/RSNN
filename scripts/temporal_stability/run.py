import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from rsnn.network.network import Network
from rsnn.neuron.neuron import Neuron
from rsnn.spike_train.measure import multi_channel_correlation
from rsnn.spike_train.sampler import sample_spike_trains
from rsnn.utils.analysis import get_phis
from rsnn.utils.utils import load_object_from_file, save_object_to_file

NUM_INPUTS = 500

PERIOD = 50 # in tau_0
FIRING_RATE = 0.2  # in number of spikes / tau_0 (outside guard period)

DELAY_MIN = 0.1  # in tau_0

NUM_CYCLES = 10
DT = 0.01

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Stability Simulation")
    parser.add_argument("--exp_idx", type=int)
    parser.add_argument("--num_neurons", type=int, default=200)
    parser.add_argument("--delay_max", type=int, default=10)
    parser.add_argument("--slope_min", type=int, default=2)
    parser.add_argument("--weight_bound", type=int, default=20) # in %
    parser.add_argument("--l1", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--l2", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    print(args)

    # if not args.similarity in {"low", "high"}:
    #     raise ValueError("similarity should be one of low or high")

    if args.l1:
        weight_regularization = "l1"
        exp_dir = os.path.join("data", f"{args.num_neurons}_{args.delay_max}_{args.slope_min}_{args.weight_bound}_l1",f"exp_{args.exp_idx}")
    elif args.l2:
        weight_regularization = "l2"
        exp_dir = os.path.join("data", f"{args.num_neurons}_{args.delay_max}_{args.slope_min}_{args.weight_bound}_l2",f"exp_{args.exp_idx}")
    else:
        weight_regularization = None
        exp_dir = os.path.join("data", f"{args.num_neurons}_{args.delay_max}_{args.slope_min}_{args.weight_bound}",f"exp_{args.exp_idx}")
        
    if os.path.exists(os.path.join(exp_dir, "results.csv")):
        print(f"Experiment already exists at",  os.path.join(exp_dir, "results.csv"), flush=True)
        exit(0)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print(f"Created directory {exp_dir}", flush=True)

    rng = np.random.default_rng(seed=args.exp_idx)

    if os.path.exists(os.path.join(exp_dir, "spike_trains.pkl")):
        spike_trains = load_object_from_file(os.path.join(exp_dir, "spike_trains.pkl"))
        print(f"Loaded spike trains from", os.path.join(exp_dir, "spike_trains.pkl"), flush=True)
    else:
        spike_trains = sample_spike_trains(PERIOD, FIRING_RATE, args.num_neurons, rng)
        save_object_to_file(spike_trains, os.path.join(exp_dir, "spike_trains.pkl"))
        print(f"Spike trains saved at", os.path.join(exp_dir, "spike_trains.pkl"), flush=True)

    # Learning phase
    if os.path.exists(os.path.join(exp_dir, f"network.pkl")):
        network = load_object_from_file(os.path.join(exp_dir, "network.pkl"))
        for neuron in network.neurons:
            if neuron.status != "optimal":
                raise ValueError(f"Problem infeasible for neuron {neuron.idx}!")
        print(f"Loaded network from", os.path.join(exp_dir, "network.pkl"), flush=True)
    else:
        network = Network(
            [
                Neuron(
                    neuron_idx,
                    NUM_INPUTS,
                    sources=rng.choice(args.num_neurons, NUM_INPUTS),
                    delays=rng.uniform(low=DELAY_MIN, high=args.delay_max, size=NUM_INPUTS),
                )
                for neuron_idx in range(args.num_neurons)
            ]
        )

        for neuron in tqdm(network.neurons):
            neuron.optimize_weights(
                spike_trains[neuron.idx], 
                [(spike_trains[s] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)], 
                PERIOD, 
                slope_min=args.slope_min,
                weight_bound=args.weight_bound/100,
                weight_regularization=weight_regularization
                )

            if neuron.status != "optimal":
                raise ValueError(f"Problem infeasible for neuron {neuron.idx}!")

        save_object_to_file(network, os.path.join(exp_dir, "network.pkl"))
        print(f"Network saved at", os.path.join(exp_dir, "network.pkl"), flush=True)

    # Analytical temporal stability
    mod_phis = np.sort(np.abs(get_phis(network.neurons, spike_trains, PERIOD)))
    print("top 5 mod phis:", mod_phis[-5:])
    # save_object_to_file(network, os.path.join(exp_dir, f"network.pkl"))
    # print(f"Network saved at", os.path.join(exp_dir, f"network.pkl"), flush=True)

    # Empirical temporal stability
    list_of_dict = []
    for std_threshold in [0.05, 0.1, 0.15, 0.2]:
        print()
        print(f"std_threshold: {std_threshold}", flush=True)
        # if os.path.exists(os.path.join(exp_dir, f"network_{std_threshold}.pkl")):
        #     print(f"Experiment already exists at",  os.path.join(exp_dir, f"network_{std_threshold}.pkl"), flush=True)
        #     continue

        for neuron in network.neurons:
            # perfect initialization
            neuron.firing_times = spike_trains[neuron.idx] - PERIOD
            neuron.firing_threshold = None

        precision, recall = multi_channel_correlation(
                            [spike_trains[neuron.idx] for neuron in network.neurons],
                            [neuron.firing_times for neuron in network.neurons],
                            -PERIOD,
                            PERIOD,
                        )
        
        print(f"cycle: -1, precision: {precision}, recall: {recall}", flush=True)
        
        list_of_dict.append(
            {
                "exp_idx": args.exp_idx,
                "std_threshold": std_threshold,
                "cycle": -1,
                "precision": precision,
                "recall": recall,
                "phi0": mod_phis[-1],
                "phi1": mod_phis[-2],
            }
        )

        for i in range(NUM_CYCLES):
            network.sim(i*PERIOD, PERIOD, DT, std_threshold, rng=rng)
        
            precision, recall = multi_channel_correlation(
                            [spike_trains[neuron.idx] for neuron in network.neurons],
                            [neuron.firing_times for neuron in network.neurons],
                            i * PERIOD,
                            PERIOD,
                        )
            print(f"cycle: {i}, precision: {precision}, recall: {recall}", flush=True)
            
            list_of_dict.append(
                {
                    "exp_idx": args.exp_idx,
                    "std_threshold": std_threshold,
                    "cycle": i,
                    "precision": precision,
                    "recall": recall,
                    "phi0": mod_phis[-1],
                    "phi1": mod_phis[-2],
                }
            )

    df = pd.DataFrame(list_of_dict)
    df.to_csv(os.path.join(exp_dir, "results.csv"), index=False)
    print(f"Experiment saved at", os.path.join(exp_dir, "results.csv"), flush=True)
