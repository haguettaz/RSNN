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

FIRING_RATE = 0.2  # in number of spikes / tau_0 (outside guard period)
DELAY_MIN = 0.1  # in tau_0

PERIOD = 50  # in tau_0
FIRING_RATE = 0.2  # in number of spikes / tau_0 (outside guard period)
NUM_CYCLES = 20

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
    parser.add_argument("--init", type=str, default="random")  # in {"silent", "random", "memory"}
    parser.add_argument("--num_ctrl_neurons", type=int, default=50)
    parser.add_argument("--std_ctrl", type=int, default=10)  # in % of tau_0

    args = parser.parse_args()

    print(args)

    if args.l1:
        weight_regularization = "l1"
        net_dir = os.path.join(
            "..", "temporal_stability", "data", f"{args.num_neurons}_{args.num_inputs}_{args.delay_max}_{args.slope_min}_{args.weight_bound}_l1", f"exp_{args.exp_idx}"
        )
        exp_dir = os.path.join(
            "data",
            f"{args.num_neurons}_{args.num_inputs}_{args.delay_max}_{args.slope_min}_{args.weight_bound}_{args.num_ctrl_neurons}_{args.std_ctrl}_{args.init}_l1",
            f"exp_{args.exp_idx}",
        )
    elif args.l2:
        weight_regularization = "l2"
        net_dir = os.path.join(
            "..", "temporal_stability", "data", f"{args.num_neurons}_{args.num_inputs}_{args.delay_max}_{args.slope_min}_{args.weight_bound}_l2", f"exp_{args.exp_idx}"
        )
        exp_dir = os.path.join(
            "data",
            f"{args.num_neurons}_{args.num_inputs}_{args.delay_max}_{args.slope_min}_{args.weight_bound}_{args.num_ctrl_neurons}_{args.std_ctrl}_{args.init}_l2",
            f"exp_{args.exp_idx}",
        )
    else:
        weight_regularization = None
        net_dir = os.path.join(
            "..", "temporal_stability", "data", f"{args.num_neurons}_{args.num_inputs}_{args.delay_max}_{args.slope_min}_{args.weight_bound}", f"exp_{args.exp_idx}"
        )
        exp_dir = os.path.join(
            "data",
            f"{args.num_neurons}_{args.num_inputs}_{args.delay_max}_{args.slope_min}_{args.weight_bound}_{args.num_ctrl_neurons}_{args.std_ctrl}_{args.init}",
            f"exp_{args.exp_idx}",
        )

    if os.path.exists(os.path.join(exp_dir, "results.csv")):
        print(f"Experiment already exists at", os.path.join(exp_dir, "results.csv"), flush=True)
        exit(0)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print(f"Created directory {exp_dir}", flush=True)

    if os.path.exists(os.path.join(net_dir, "spike_trains.pkl")):
        spike_trains = load_object_from_file(os.path.join(net_dir, "spike_trains.pkl"))
        print(f"Loaded spike trains from", os.path.join(net_dir, "spike_trains.pkl"), flush=True)
    else:
        spike_trains = sample_spike_trains(PERIOD, FIRING_RATE, args.num_neurons)
        save_object_to_file(spike_trains, os.path.join(net_dir, "spike_trains.pkl"))
        print(f"Spike trains saved at", os.path.join(net_dir, "spike_trains.pkl"), flush=True)

    # Learning phase
    if os.path.exists(os.path.join(net_dir, "network.pkl")):
        network = load_object_from_file(os.path.join(net_dir, "network.pkl"))
        for neuron in network.neurons:
            if neuron.status != "optimal":
                raise ValueError(f"Problem infeasible for neuron {neuron.idx}!")
        print(f"Loaded network from", os.path.join(net_dir, "network.pkl"), flush=True)
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
            neuron.optimize_weights(
                spike_trains[neuron.idx],
                [(spike_trains[s] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)],
                PERIOD,
                slope_min=args.slope_min,
                weight_bound=args.weight_bound / 100,
                weight_regularization=weight_regularization,
            )

            if neuron.status != "optimal":
                raise ValueError(f"Problem infeasible for neuron {neuron.idx}!")

        save_object_to_file(network, os.path.join(net_dir, "network.pkl"))
        print(f"Network saved at", os.path.join(net_dir, "network.pkl"), flush=True)

    # # Analytical temporal stability
    # mod_phis = np.sort(np.abs(get_phis(network.neurons, spike_trains, PERIOD)))
    # print("top 5 mod phis:", mod_phis[-5:])
    # save_object_to_file(network, os.path.join(net_dir, f"network.pkl"))
    # print(f"Network saved at", os.path.join(net_dir, f"network.pkl"), flush=True)

    # Empirical associative recall
    # list_of_neuron_dict = []
    list_of_dict = []

    if args.init == "silent":
        init_spike_trains = [np.array([])] * args.num_neurons
    elif args.init == "random":
        init_spike_trains = sample_spike_trains(PERIOD, FIRING_RATE, args.num_neurons)
    elif args.init == "memory":
        init_spike_trains = spike_trains

    for std_threshold in [0.05, 0.1, 0.15]:
        for neuron in network.neurons:
            # set controllable neurons
            if neuron.idx >= args.num_neurons - args.num_ctrl_neurons:
                neuron.firing_times = np.concatenate(
                    [init_spike_trains[neuron.idx] - PERIOD, np.random.normal(spike_trains[neuron.idx][None, :] + PERIOD * np.arange(NUM_CYCLES)[:, None], args.std_ctrl / 100).flatten()]
                )
            # set uncontrollable neurons
            else:
                neuron.firing_times = init_spike_trains[neuron.idx] - PERIOD
                neuron.firing_threshold = None

        for i in range(NUM_CYCLES): # 10
            network.sim(i * PERIOD, PERIOD, 0.01, std_threshold, range(args.num_neurons - args.num_ctrl_neurons))

            precision, recall = multi_channel_correlation(
                [spike_trains[neuron.idx] for neuron in network.neurons],
                [neuron.firing_times for neuron in network.neurons],
                i * PERIOD,
                PERIOD,
            )
            print(f"std_threshold: {std_threshold}, cycle: {i}, precision: {precision}, recall: {recall}", flush=True)
            list_of_dict.append(
                {
                    "exp_idx": args.exp_idx,
                    "num_inputs": args.num_inputs,
                    "num_neurons": args.num_neurons,
                    "delay_max": args.delay_max,
                    "slope_min": args.slope_min,
                    "weight_bound": args.weight_bound / 100,
                    "weight_regularization": weight_regularization,
                    "init": args.init,
                    "num_ctrl_neurons": args.num_ctrl_neurons,
                    "std_ctrl": args.std_ctrl / 100,
                    "std_threshold": std_threshold,
                    "cycle": i,
                    "precision": precision,
                    "recall": recall,
                    "type": "all",
                }
            )

            precision, recall = multi_channel_correlation(
                [spike_trains[neuron.idx] for neuron in network.neurons[-args.num_ctrl_neurons :]],
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
                    "init": args.init,
                    "num_ctrl_neurons": args.num_ctrl_neurons,
                    "std_ctrl": args.std_ctrl / 100,
                    "std_threshold": std_threshold,
                    "cycle": i,
                    "precision": precision,
                    "recall": recall,
                    "type": "ctrl",
                }
            )

            precision, recall = multi_channel_correlation(
                [spike_trains[neuron.idx] for neuron in network.neurons[: -args.num_ctrl_neurons]],
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
                    "init": args.init,
                    "num_ctrl_neurons": args.num_ctrl_neurons,
                    "std_ctrl": args.std_ctrl / 100,
                    "std_threshold": std_threshold,
                    "cycle": i,
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
