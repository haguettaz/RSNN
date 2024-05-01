import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from rsnn.network.network import Network
from rsnn.neuron.neuron import Neuron
from rsnn.spike_train.measure import (multi_channel_correlation,
                                      single_channel_correlation)
from rsnn.spike_train.sampler import (sample_jittered_spike_train,
                                      sample_spike_trains)
from rsnn.utils.utils import load_object_from_file, save_object_to_file

NUM_NEURONS = 200
NUM_INPUTS = 500

PERIOD = 50  # in tau_0
FIRING_RATE = 0.2  # in number of spikes / tau_0 (outside guard period)

DELAY_MIN = 0.1  # in tau_0
DELAY_MAX = 10  # in tau_0

SLOPE_MIN = 2
WEIGHT_BOUND = 20
WEIGHT_REGULARIZATION = "l2"

DT = 0.01
NUM_CYCLES = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Stability Simulation")
    parser.add_argument("--exp_idx", type=int)
    parser.add_argument(
        "--init", type=str, default="silent"
    )  # in {"silent", "random", "memory"}
    parser.add_argument("--num_ctrl_neurons", type=int, default=100)
    parser.add_argument("--std_ctrl", type=int, default=10)  # in % of tau_0

    args = parser.parse_args()

    print(args)

    rsc_dir = os.path.join(
        "..",
        "temporal_stability",
        "data",
        f"{NUM_NEURONS}_{DELAY_MAX}_{SLOPE_MIN}_{WEIGHT_BOUND}_l2",
        f"exp_{args.exp_idx}",
    )
    exp_dir = os.path.join(
        "data",
        f"{args.num_ctrl_neurons}_{args.std_ctrl}_{args.init}",
        f"exp_{args.exp_idx}",
    )

    if os.path.exists(os.path.join(exp_dir, "results.csv")):
        print(
            f"Experiment already exists at",
            os.path.join(exp_dir, "results.csv"),
            flush=True,
        )
        exit(0)

    rng = np.random.default_rng(seed=args.exp_idx)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print(f"Created directory {exp_dir}", flush=True)

    if os.path.exists(os.path.join(rsc_dir, "spike_trains.pkl")):
        spike_trains = load_object_from_file(os.path.join(rsc_dir, "spike_trains.pkl"))
        print(
            f"Loaded spike trains from",
            os.path.join(rsc_dir, "spike_trains.pkl"),
            flush=True,
        )
    else:
        spike_trains = sample_spike_trains(PERIOD, FIRING_RATE, NUM_NEURONS)
        save_object_to_file(spike_trains, os.path.join(rsc_dir, "spike_trains.pkl"))
        print(
            f"Spike trains saved at",
            os.path.join(rsc_dir, "spike_trains.pkl"),
            flush=True,
        )

    # Learning phase
    if os.path.exists(os.path.join(rsc_dir, "network.pkl")):
        network = load_object_from_file(os.path.join(rsc_dir, "network.pkl"))
        for neuron in network.neurons:
            if neuron.status != "optimal":
                raise ValueError(f"Problem infeasible for neuron {neuron.idx}!")
        print(f"Loaded network from", os.path.join(rsc_dir, "network.pkl"), flush=True)
    else:
        network = Network(
            [
                Neuron(
                    neuron_idx,
                    NUM_INPUTS,
                    sources=rng.choice(NUM_NEURONS, NUM_INPUTS),
                    delays=rng.uniform(low=DELAY_MIN, high=DELAY_MAX, size=NUM_INPUTS),
                )
                for neuron_idx in range(NUM_NEURONS)
            ]
        )

        for neuron in tqdm(network.neurons):
            neuron.optimize_weights(
                spike_trains[neuron.idx],
                [
                    (spike_trains[s] + d).reshape(-1)
                    for s, d in zip(neuron.sources, neuron.delays)
                ],
                PERIOD,
                slope_min=SLOPE_MIN,
                weight_bound=WEIGHT_BOUND / 100,
                weight_regularization=WEIGHT_REGULARIZATION,
            )

            if neuron.status != "optimal":
                raise ValueError(f"Problem infeasible for neuron {neuron.idx}!")

        save_object_to_file(network, os.path.join(rsc_dir, "network.pkl"))
        print(f"Network saved at", os.path.join(rsc_dir, "network.pkl"), flush=True)

    # Empirical associative recall
    list_of_dict = []

    if args.init == "silent":
        init_spike_trains = [np.array([])] * NUM_NEURONS
    elif args.init == "random":
        init_spike_trains = sample_spike_trains(
            PERIOD, FIRING_RATE, NUM_NEURONS, rng=rng
        )
    elif args.init == "memory":
        init_spike_trains = spike_trains

    # use the same noisy prompts for every experiment
    for neuron in network.neurons[-args.num_ctrl_neurons :]:
        neuron.firing_threshold = None
        neuron.firing_times = sample_jittered_spike_train(
            (
                spike_trains[neuron.idx][None, :]
                + (np.arange(-1, NUM_CYCLES) * PERIOD)[:, None]
            ).flatten(),
            args.std_ctrl / 100,
            rng=rng,
        )

    for std_threshold in [0.05, 0.1, 0.15, 0.2]:
        print()
        print(f"std_threshold: {std_threshold}", flush=True)

        for neuron in network.neurons[: -args.num_ctrl_neurons]:
            neuron.firing_threshold = None
            neuron.firing_times = init_spike_trains[neuron.idx] - PERIOD

        precision, recall = multi_channel_correlation(
            [spike_trains[neuron.idx] for neuron in network.neurons],
            [neuron.firing_times for neuron in network.neurons],
            -PERIOD,
            PERIOD,
        )

        print(f"cycle: -1, precision: {precision}, recall: {recall}", flush=True)

        for i in range(NUM_CYCLES):  # 10
            # set controllable neurons
            network.sim(
                i * PERIOD,
                PERIOD,
                DT,
                std_threshold,
                range(NUM_NEURONS - args.num_ctrl_neurons),
                rng=rng,
            )

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
                [
                    spike_trains[neuron.idx]
                    for neuron in network.neurons[-args.num_ctrl_neurons :]
                ],
                [
                    neuron.firing_times
                    for neuron in network.neurons[-args.num_ctrl_neurons :]
                ],
                i * PERIOD,
                PERIOD,
            )
            list_of_dict.append(
                {
                    "exp_idx": args.exp_idx,
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
                [
                    spike_trains[neuron.idx]
                    for neuron in network.neurons[: -args.num_ctrl_neurons]
                ],
                [
                    neuron.firing_times
                    for neuron in network.neurons[: -args.num_ctrl_neurons]
                ],
                i * PERIOD,
                PERIOD,
            )
            list_of_dict.append(
                {
                    "exp_idx": args.exp_idx,
                    "init": args.init,
                    "num_ctrl_neurons": args.num_ctrl_neurons,
                    "std_ctrl": args.std_ctrl / 100,
                    "std_threshold": std_threshold,
                    "cycle": i,
                    "precision": precision,
                    "recall": recall,
                    "type": "auto",
                }
            )

            for neuron in network.neurons:
                precision, recall = single_channel_correlation(
                    spike_trains[neuron.idx],
                    neuron.firing_times,
                    i * PERIOD,
                    PERIOD,
                )
                list_of_dict.append(
                    {
                        "exp_idx": args.exp_idx,
                        "cycle": i,
                        "precision": precision,
                        "recall": recall,
                        "idx": neuron.idx,
                        "type": (
                            "ctrl"
                            if neuron.idx >= NUM_NEURONS - args.num_ctrl_neurons
                            else "auto"
                        ),
                    }
                )

    pd.DataFrame(list_of_dict).to_csv(os.path.join(exp_dir, "results.csv"), index=False)
    print(f"Experiment saved at", os.path.join(exp_dir, "results.csv"), flush=True)
