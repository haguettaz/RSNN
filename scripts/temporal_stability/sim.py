import argparse
import os

from rsnn.spike_train.measure import multi_channel_correlation
from rsnn.utils.utils import load_object_from_file, save_object_to_file

# NUM_NEURONS = 500

NOMINAL_THRESHOLD = 1.0
ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

NUM_INPUTS = 500

FIRING_RATE = 10

DT = 0.1
EPS = 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Stability Simulation")
    parser.add_argument("--num_neurons", type=int, default=500)
    parser.add_argument("--num_inputs", type=int, default=500)
    parser.add_argument("--period", type=int, default=100)
    parser.add_argument("--wb", type=int, default=20)
    parser.add_argument("--l_reg", type=int, default=0)
    parser.add_argument("--slope_min", type=int, default=50)
    parser.add_argument("--exp_idx", type=int)
    parser.add_argument("--std_threshold", type=int, default=0)

    args = parser.parse_args()

    print("Parameters", flush=True)
    print("num_neurons:", args.num_neurons, flush=True)
    print("num_inputs:", args.num_inputs, flush=True)
    print("period:", args.period, flush=True)
    print("wb:", args.wb, flush=True)
    print("slope_min:", args.slope_min, flush=True)
    # print("std_threshold:", arg.std_threshold/100, flush=True)
    print("exp_idx:", args.exp_idx, flush=True)
    print()

    list_std_threshold = [5, 10, 15, 20] if args.std_threshold == 0 else [args.std_threshold]

    if args.l_reg > 0:
        dir_path = f"{args.num_neurons}_{args.num_inputs}_{args.period}_{args.wb}_{args.slope_min}_{args.l_reg}"
    else:
        dir_path = f"{args.num_neurons}_{args.num_inputs}_{args.period}_{args.wb}_{args.slope_min}"

    for std_threshold in list_std_threshold:
        if not os.path.exists(os.path.join(dir_path, "sim", "networks", f"experiment_{args.exp_idx}_noise_{std_threshold}_network.pkl")):
            spike_trains = load_object_from_file(os.path.join(dir_path, "spike_trains", f"experiment_{args.exp_idx}_spike_trains.pkl"))
            network = load_object_from_file(os.path.join(dir_path, "optim", "networks", f"experiment_{args.exp_idx}_network.pkl"))

            for neuron in network.neurons:
                # perfect initialization
                neuron.firing_times = spike_trains[neuron.idx] - args.period
                neuron.firing_threshold = None

            network.precision = []
            network.recall = []

            for i in range(8):
                network.sim(i*args.period, args.period, DT, std_threshold/100, range(args.num_neurons))
            
                precision, recall = multi_channel_correlation(
                                [spike_trains[neuron.idx] for neuron in network.neurons],
                                [neuron.firing_times for neuron in network.neurons],
                                i * args.period,
                                args.period,
                                ABSOLUTE_REFRACTORY,
                                EPS,
                            )

                network.precision.append(precision)
                network.recall.append(recall)

                print(f"cycle {i}: precision={precision}, recall={recall}", flush=True)

            save_object_to_file(network, os.path.join(dir_path, "sim", "networks", f"experiment_{args.exp_idx}_noise_{std_threshold}_network.pkl"))
