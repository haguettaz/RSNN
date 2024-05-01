import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from rsnn.network.network import Network
from rsnn.neuron.neuron import Neuron
from rsnn.spike_train.measure import (multi_channel_correlation,
                                      single_channel_correlation)
from rsnn.spike_train.sampler import (sample_jittered_spike_train,
                                      sample_spike_trains)
from rsnn.spike_train.utils import (check_refractoriness,
                                    check_refractoriness_periodicity)
from rsnn.utils.utils import load_object_from_file, save_object_to_file

NUM_NEURONS = 200
NUM_INPUTS = 500

PERIOD = 50  # in tau_0
FIRING_RATE = 0.2  # in number of spikes / tau_0 (outside guard period)

DELAY_MIN = 0.1  # in tau_0
DELAY_MAX = 10  # in tau_0

SLOPE_MIN = 2
WEIGHT_BOUND = 0.2
WEIGHT_REGULARIZATION = "l2"

DT = 0.01

NUM_PATTERNS = 2
SCHEDULE = [
    {"pattern_ctrl": 0, "num_ctrl_neurons": 150, "std_ctrl": 0.1, "std_threshold": 0.1},
    # {"pattern_ctrl": 0, "num_ctrl_neurons": 150, "std_ctrl": 0.1, "std_threshold": 0.1},
    # {"pattern_ctrl": 0, "num_ctrl_neurons": 150, "std_ctrl": 0.1, "std_threshold": 0.1},
    {"num_ctrl_neurons": 0, "std_threshold": 0.1},
    {"num_ctrl_neurons": 0, "std_threshold": 0.1},
    {"num_ctrl_neurons": 0, "std_threshold": 0.1},
    {"pattern_ctrl": 1, "num_ctrl_neurons": 150, "std_ctrl": 0.1, "std_threshold": 0.1},
    # {"pattern_ctrl": 1, "num_ctrl_neurons": 150, "std_ctrl": 0.1, "std_threshold": 0.1},
    # {"pattern_ctrl": 1, "num_ctrl_neurons": 150, "std_ctrl": 0.1, "std_threshold": 0.1},
    {"num_ctrl_neurons": 0, "std_threshold": 0.1},
    {"num_ctrl_neurons": 0, "std_threshold": 0.1},
    {"num_ctrl_neurons": 0, "std_threshold": 0.1},
    {"pattern_ctrl": 0, "num_ctrl_neurons": 150, "std_ctrl": 0.1, "std_threshold": 0.1},
    # {"pattern_ctrl": 0, "num_ctrl_neurons": 150, "std_ctrl": 0.1, "std_threshold": 0.1},
    # {"pattern_ctrl": 0, "num_ctrl_neurons": 150, "std_ctrl": 0.1, "std_threshold": 0.1},
    {"num_ctrl_neurons": 0, "std_threshold": 0.1},
    {"num_ctrl_neurons": 0, "std_threshold": 0.1},
    {"num_ctrl_neurons": 0, "std_threshold": 0.1},
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Stability Simulation")
    parser.add_argument("--exp_idx", type=int)

    args = parser.parse_args()

    print(args)

    exp_dir = os.path.join("data", f"exp_{args.exp_idx}")

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

    if os.path.exists(os.path.join(exp_dir, "spike_trains.pkl")):
        spike_trains = load_object_from_file(os.path.join(exp_dir, "spike_trains.pkl"))
        print(
            f"Loaded spike trains from",
            os.path.join(exp_dir, "spike_trains.pkl"),
            flush=True,
        )
    else:
        spike_trains = [
            sample_spike_trains(PERIOD, FIRING_RATE, NUM_NEURONS, rng=rng)
            for _ in range(NUM_PATTERNS)
        ]
        save_object_to_file(spike_trains, os.path.join(exp_dir, "spike_trains.pkl"))
        print(
            f"Spike trains saved at",
            os.path.join(exp_dir, "spike_trains.pkl"),
            flush=True,
        )

    for spike_train in spike_trains:
        for firing_times in spike_train:
            if not check_refractoriness_periodicity(firing_times, PERIOD):
                print(firing_times)
                raise ValueError(f"Refractory condition violated for some neuron!")

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
                    NUM_INPUTS,
                    sources=rng.choice(NUM_NEURONS, NUM_INPUTS),
                    delays=rng.uniform(low=DELAY_MIN, high=DELAY_MAX, size=NUM_INPUTS),
                )
                for neuron_idx in range(NUM_NEURONS)
            ]
        )

        for neuron in tqdm(network.neurons):
            neuron.optimize_weights_many(
                [fts[neuron.idx] for fts in spike_trains],
                [
                    [
                        (fts[s] + d).reshape(-1)
                        for s, d in zip(neuron.sources, neuron.delays)
                    ]
                    for fts in spike_trains
                ],
                PERIOD,
                slope_min=SLOPE_MIN,
                weight_bound=WEIGHT_BOUND,
                weight_regularization=WEIGHT_REGULARIZATION,
            )

            if neuron.status != "optimal":
                raise ValueError(f"Problem infeasible for neuron {neuron.idx}!")

        save_object_to_file(network, os.path.join(exp_dir, "network.pkl"))
        print(f"Network saved at", os.path.join(exp_dir, "network.pkl"), flush=True)

    # Empirical associative recall
    list_of_dict = []
    init_spike_trains = [np.array([])] * NUM_NEURONS

    for neuron in network.neurons:
        neuron.firing_times = init_spike_trains[neuron.idx] - PERIOD
        neuron.firing_threshold = None

    for i, prog in enumerate(SCHEDULE):
        # set controllable neurons
        if prog["num_ctrl_neurons"] > 0:
            for neuron in network.neurons[-prog["num_ctrl_neurons"] :]:
                # sample jittered spike trains
                try:
                    jittered_spike_train = sample_jittered_spike_train(
                        spike_trains[prog["pattern_ctrl"]][neuron.idx] + i * PERIOD,
                        prog["std_ctrl"],
                        # i*PERIOD,
                        np.max(neuron.firing_times, initial=i * PERIOD - 1) + 1,
                        (i + 1) * PERIOD,
                        rng=rng,
                    )
                except ValueError:
                    print(f"tmin: {np.max(neuron.firing_times, initial=i * PERIOD - 1) + 1} and tmax: {(i + 1) * PERIOD}")
                    print(f"spike_train: {spike_trains[prog['pattern_ctrl']][neuron.idx] + i * PERIOD}")
                    raise ValueError
                neuron.firing_times = np.concatenate(
                    [neuron.firing_times, jittered_spike_train]
                )
                neuron.firing_threshold = None

        network.sim(
            i * PERIOD,
            PERIOD,
            DT,
            prog["std_threshold"],
            range(NUM_NEURONS - prog["num_ctrl_neurons"]),
            rng=rng,
        )

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
                    "num_inputs": NUM_INPUTS,
                    "num_neurons": NUM_NEURONS,
                    "delay_max": DELAY_MAX,
                    "slope_min": SLOPE_MIN,
                    "weight_bound": WEIGHT_BOUND,
                    "weight_regularization": WEIGHT_REGULARIZATION,
                    "cycle": i,
                    "pattern": j,
                    "precision": precision,
                    "recall": recall,
                    "type": "all",
                }
            )
            print(f"Cycle {i}, Pattern {j}, Precision {precision}, Recall {recall}", flush=True)

            # precision, recall = multi_channel_correlation(
            #     [spike_trains[j][neuron.idx] for neuron in network.neurons[-prog["num_ctrl_neurons"] :]],
            #     [neuron.firing_times for neuron in network.neurons[-prog["num_ctrl_neurons"] :]],
            #     i * PERIOD,
            #     PERIOD,
            # )
            # list_of_dict.append(
            #     {
            #         "exp_idx": args.exp_idx,
            #         "num_inputs": NUM_INPUTS,
            #         "num_neurons": NUM_NEURONS,
            #         "delay_max": DELAY_MAX,
            #         "slope_min": SLOPE_MIN,
            #         "weight_bound": WEIGHT_BOUND,
            #         "weight_regularization": weight_regularization,
            #         "cycle": i,
            #         "pattern":j,
            #         "precision": precision,
            #         "recall": recall,
            #         "type": "ctrl",
            #     }
            # )

            # precision, recall = multi_channel_correlation(
            #     [spike_trains[j][neuron.idx] for neuron in network.neurons[: -prog["num_ctrl_neurons"]]],
            #     [neuron.firing_times for neuron in network.neurons[: -prog["num_ctrl_neurons"]]],
            #     i * PERIOD,
            #     PERIOD,
            # )
            # list_of_dict.append(
            #     {
            #         "exp_idx": args.exp_idx,
            #         "num_inputs": NUM_INPUTS,
            #         "num_neurons": NUM_NEURONS,
            #         "delay_max": DELAY_MAX,
            #         "slope_min": SLOPE_MIN,
            #         "weight_bound": WEIGHT_BOUND,
            #         "weight_regularization": weight_regularization,
            #         "cycle": i,
            #         "pattern":j,
            #         "precision": precision,
            #         "recall": recall,
            #         "type": "auto",
            #     }
            # )

            for neuron in network.neurons:
                if not check_refractoriness(neuron.firing_times):
                    raise ValueError(
                        f"Refractory condition violated for neuron {neuron.idx}!"
                    )

                precision, recall = single_channel_correlation(
                    spike_trains[j][neuron.idx],
                    neuron.firing_times,
                    i * PERIOD,
                    PERIOD,
                )
                list_of_dict.append(
                    {
                        "exp_idx": args.exp_idx,
                        "num_inputs": NUM_INPUTS,
                        "num_neurons": NUM_NEURONS,
                        "delay_max": DELAY_MAX,
                        "slope_min": SLOPE_MIN,
                        "weight_bound": WEIGHT_BOUND,
                        "weight_regularization": WEIGHT_REGULARIZATION,
                        "cycle": i,
                        "pattern": j,
                        "precision": precision,
                        "recall": recall,
                        "idx": neuron.idx,
                        "type": (
                            "ctrl"
                            if neuron.idx >= NUM_NEURONS - prog["num_ctrl_neurons"]
                            else "auto"
                        ),
                    }
                )

    pd.DataFrame(list_of_dict).to_csv(os.path.join(exp_dir, "results.csv"), index=False)
    print(f"Experiment saved at", os.path.join(exp_dir, "results.csv"), flush=True)
