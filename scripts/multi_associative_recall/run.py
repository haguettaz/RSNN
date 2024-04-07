import argparse
import os
from random import choices

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from rsnn.network.network import Network
from rsnn.neuron.neuron import Neuron
from rsnn.spike_train.measure import (multi_channel_correlation,
                                      single_channel_correlation)
from rsnn.spike_train.sampler import sample_spike_trains
from rsnn.utils.analysis import get_phis
from rsnn.utils.utils import load_object_from_file, save_object_to_file

FIRING_RATE = 0.2  # in number of spikes / tau_min (outside guard period)
DELAY_MIN = 0.1  # in tau_min

PERIOD = 30  # in tau_min
FIRING_RATE = 0.2  # in number of spikes / tau_min (outside guard period)

NUM_PATTERNS = 2
SCHEDULE = [
    {"pattern_ctrl":0, "num_ctrl_neurons":50, "std_ctrl":0.1, "std_threshold": 0.1},
    {"pattern_ctrl":0, "num_ctrl_neurons":50, "std_ctrl":0.1, "std_threshold": 0.1},
    {"num_ctrl_neurons":0, "std_threshold": 0.1},
    {"num_ctrl_neurons":0, "std_threshold": 0.1},
    {"num_ctrl_neurons":0, "std_threshold": 0.1},
    {"pattern_ctrl":1, "num_ctrl_neurons":50, "std_ctrl":0.1, "std_threshold": 0.1},
    {"pattern_ctrl":1, "num_ctrl_neurons":50, "std_ctrl":0.1, "std_threshold": 0.1},
    {"num_ctrl_neurons":0, "std_threshold": 0.1},
    {"num_ctrl_neurons":0, "std_threshold": 0.1},
    {"num_ctrl_neurons":0, "std_threshold": 0.1},
    {"pattern_ctrl":1, "num_ctrl_neurons":50, "std_ctrl":0.1, "std_threshold": 0.1},
    {"pattern_ctrl":1, "num_ctrl_neurons":50, "std_ctrl":0.1, "std_threshold": 0.1},
    {"pattern_ctrl":1, "num_ctrl_neurons":50, "std_ctrl":0.5, "std_threshold": 0.1},
    {"pattern_ctrl":1, "num_ctrl_neurons":50, "std_ctrl":0.5, "std_threshold": 0.1},
    {"pattern_ctrl":1, "num_ctrl_neurons":50, "std_ctrl":0.1, "std_threshold": 0.1},
    {"pattern_ctrl":1, "num_ctrl_neurons":50, "std_ctrl":0.1, "std_threshold": 0.1},
    {"pattern_ctrl":0, "num_ctrl_neurons":50, "std_ctrl":0.1, "std_threshold": 0.1},
    {"pattern_ctrl":0, "num_ctrl_neurons":50, "std_ctrl":0.1, "std_threshold": 0.1},
    {"pattern_ctrl":0, "num_ctrl_neurons":50, "std_ctrl":0.1, "std_threshold": 0.1},
    {"num_ctrl_neurons":0, "std_threshold": 0.1},
    {"num_ctrl_neurons":0, "std_threshold": 0.1},
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Stability Simulation")
    parser.add_argument("--exp_idx", type=int)
    parser.add_argument("--num_inputs", type=int, default=400)
    parser.add_argument("--num_neurons", type=int, default=100)
    parser.add_argument("--delay_max", type=int, default=10)
    parser.add_argument("--slope_min", type=int, default=2)
    parser.add_argument("--weight_bound", type=int, default=20)  # in %
    parser.add_argument("--l1", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--l2", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    print(args)

    if args.l1:
        weight_regularization = "l1"
        exp_dir = os.path.join(
            "data",
            f"{args.num_neurons}_{args.num_inputs}_{args.delay_max}_{args.slope_min}_{args.weight_bound}_l1",
            f"exp_{args.exp_idx}",
        )
    elif args.l2:
        weight_regularization = "l2"
        exp_dir = os.path.join(
            "data",
            f"{args.num_neurons}_{args.num_inputs}_{args.delay_max}_{args.slope_min}_{args.weight_bound}_l2",
            f"exp_{args.exp_idx}",
        )
    else:
        weight_regularization = None
        exp_dir = os.path.join(
            "data",
            f"{args.num_neurons}_{args.num_inputs}_{args.delay_max}_{args.slope_min}_{args.weight_bound}",
            f"exp_{args.exp_idx}",
        )

    if os.path.exists(os.path.join(exp_dir, "results.csv")):
        print(f"Experiment already exists at", os.path.join(exp_dir, "results.csv"), flush=True)
        exit(0)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print(f"Created directory {exp_dir}", flush=True)

    if os.path.exists(os.path.join(exp_dir, "spike_trains.pkl")):
        spike_trains = load_object_from_file(os.path.join(exp_dir, "spike_trains.pkl"))
        print(f"Loaded spike trains from", os.path.join(exp_dir, "spike_trains.pkl"), flush=True)
    else:
        spike_trains = [sample_spike_trains(PERIOD, FIRING_RATE, args.num_neurons) for _ in range(NUM_PATTERNS)]
        save_object_to_file(spike_trains, os.path.join(exp_dir, "spike_trains.pkl"))
        print(f"Spike trains saved at", os.path.join(exp_dir, "spike_trains.pkl"), flush=True)

    # Learning phase
    if os.path.exists(os.path.join(exp_dir, "network.pkl")):
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
                    args.num_inputs,
                    sources=np.random.choice(args.num_neurons, args.num_inputs),
                    delays=np.random.uniform(low=DELAY_MIN, high=args.delay_max, size=args.num_inputs),
                )
                for neuron_idx in range(args.num_neurons)
            ]
        )

        for neuron in tqdm(network.neurons):
            neuron.optimize_weights_many(
                [fts[neuron.idx] for fts in spike_trains],
                [[(fts[s] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)] for fts in spike_trains],
                PERIOD,
                slope_min=args.slope_min,
                weight_bound=args.weight_bound / 100,
                weight_regularization=weight_regularization,
            )

            if neuron.status != "optimal":
                raise ValueError(f"Problem infeasible for neuron {neuron.idx}!")

        save_object_to_file(network, os.path.join(exp_dir, "network.pkl"))
        print(f"Network saved at", os.path.join(exp_dir, "network.pkl"), flush=True)

    # Empirical associative recall
    list_of_dict = []
    init_spike_trains = sample_spike_trains(PERIOD, FIRING_RATE, args.num_neurons)

    for neuron in network.neurons:
        neuron.firing_times = init_spike_trains[neuron.idx] - PERIOD

    for i, prog in enumerate(SCHEDULE): 

        # set controllable neurons
        if neuron.idx >= args.num_neurons - prog["num_ctrl_neurons"]:
            neuron.firing_times = np.concatenate(
                [neuron.firing_times, np.random.normal(spike_trains[prog["pattern_ctrl"]][neuron.idx], prog["std_ctrl"] / 100)]
            )
        # set uncontrollable neurons
        else:
            neuron.firing_threshold = None


        network.sim(i * PERIOD, PERIOD, 0.01, prog["std_threshold"], range(args.num_neurons - prog["num_ctrl_neurons"]))

        for j in range(NUM_PATTERNS):
            precision, recall = multi_channel_correlation(
                [spike_trains[j][neuron.idx] for neuron in network.neurons],
                [neuron.firing_times for neuron in network.neurons],
                i * PERIOD,
                PERIOD,
            )
            list_of_dict.append(
                {
                    "exp_idx": args.exp_idx,
                    "num_inputs": args.num_inputs,
                    "num_neurons": args.num_neurons,
                    "delay_max": args.delay_max,
                    "slope_min": args.slope_min,
                    "weight_bound": args.weight_bound / 100,
                    "weight_regularization": weight_regularization,
                    "cycle": i,
                    "pattern":j,
                    "precision": precision,
                    "recall": recall,
                    "type": "all",
                }
            )

            precision, recall = multi_channel_correlation(
                [spike_trains[j][neuron.idx] for neuron in network.neurons[-args.num_ctrl_neurons :]],
                [neuron.firing_times for neuron in network.neurons[-args.num_ctrl_neurons :]],
                i * PERIOD,
                PERIOD,
            )
            list_of_dict.append(
                {
                    "exp_idx": args.exp_idx,
                    "num_inputs": args.num_inputs,
                    "num_neurons": args.num_neurons,
                    "delay_max": args.delay_max,
                    "slope_min": args.slope_min,
                    "weight_bound": args.weight_bound / 100,
                    "weight_regularization": weight_regularization,
                    "cycle": i,
                    "pattern":j,
                    "precision": precision,
                    "recall": recall,
                    "type": "ctrl",
                }
            )

            precision, recall = multi_channel_correlation(
                [spike_trains[j][neuron.idx] for neuron in network.neurons[: -args.num_ctrl_neurons]],
                [neuron.firing_times for neuron in network.neurons[: -args.num_ctrl_neurons]],
                i * PERIOD,
                PERIOD,
            )
            list_of_dict.append(
                {
                    "exp_idx": args.exp_idx,
                    "num_inputs": args.num_inputs,
                    "num_neurons": args.num_neurons,
                    "delay_max": args.delay_max,
                    "slope_min": args.slope_min,
                    "weight_bound": args.weight_bound / 100,
                    "weight_regularization": weight_regularization,
                    "cycle": i,
                    "pattern":j,
                    "precision": precision,
                    "recall": recall,
                    "type": "non-ctrl",
                }
            )

        # for neuron in network.neurons:
        #     precision, recall = single_channel_correlation(
        #         spike_trains[neuron.idx],
        #         neuron.firing_times,
        #         i * PERIOD,
        #         PERIOD,
        #     )
        #     list_of_neuron_dict.append(
        #         {
        #             "exp_idx": args.exp_idx,
        #             "num_inputs": args.num_inputs,
        #             "num_neurons": args.num_neurons,
        #             "delay_max": args.delay_max,
        #             "slope_min": args.slope_min,
        #             "weight_bound": args.weight_bound / 100,
        #             "weight_regularization": weight_regularization,
        #             "init": args.init,
        #             "num_ctrl_neurons": args.num_ctrl_neurons,
        #             "std_ctrl": args.std_ctrl / 100,
        #             "std_threshold": std_threshold,
        #             "std_threshold": std_threshold,
        #             "cycle": i,
        #             "precision": precision,
        #             "recall": recall,
        #             "neuron_idx": neuron.idx,
        #             "neuron_type": "ctrl" if neuron.idx >= args.num_neurons - args.num_ctrl_neurons else "non-ctrl",
        #         }
        #     )

    pd.DataFrame(list_of_dict).to_csv(os.path.join(exp_dir, "results.csv"), index=False)
    print(f"Experiment saved at", os.path.join(exp_dir, "results.csv"), flush=True)
    # pd.DataFrame(list_of_neuron_dict).to_csv(os.path.join(exp_dir, "neuron_results.csv"), index=False)
    # print(f"Experiment saved at", os.path.join(exp_dir, "neuron_results.csv"), flush=True)
