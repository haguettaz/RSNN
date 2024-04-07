import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import trange

from rsnn.neuron.neuron import Neuron
from rsnn.spike_train.sampler import sample_spike_trains

# Remark: tau_0 is the minimum distance between two consecutive spikes, i.e., the absolute refractory period.

FIRING_RATE = 0.2 # in number of spikes / tau_0 (outside guard period)
DELAY_MIN = 0.1  # in tau_0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Storage Capacity Simulation")
    parser.add_argument("--num_exp", type=int, default=1000)
    parser.add_argument("--num_neurons", type=int, default=0) # 0 => num_neurons = num_inputs
    parser.add_argument("--num_inputs", type=int)
    # parser.add_argument("--similarity", type=str) # low or high
    parser.add_argument("--delay_max", type=int, default=10)
    parser.add_argument("--weight_bound", type=int, default=20)  # [in %]
    parser.add_argument("--period_min", type=int, default=20)

    args = parser.parse_args()
    print(args, flush=True)

    # if not args.similarity in {"low", "high"}:
    #     raise ValueError("similarity should be one of low or high")

    exp_dir = os.path.join("data", f"{args.num_neurons}_{args.num_inputs}_{args.delay_max}_{args.weight_bound}")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print(f"Created directory {exp_dir}", flush=True)
    elif os.path.exists(os.path.join(exp_dir, "results.csv")):
        print("Experiment already done", flush=True)
        sys.exit(0)
    
    list_of_dict = []

    neuron = Neuron(0, args.num_inputs)

    period = args.period_min
    max_num_errors_99 = 0.01 * args.num_exp

    while True:
        num_errors = 0
        for exp_idx in trange(args.num_exp):
            if args.num_neurons < 1:
                neuron.sources = np.arange(args.num_inputs)
            else:
                neuron.sources = np.random.choice(args.num_neurons, args.num_inputs)
            neuron.delays = np.random.uniform(low=DELAY_MIN, high=args.delay_max, size=args.num_inputs)

            firing_times = sample_spike_trains(period, FIRING_RATE, 1)[0]

            spike_trains = sample_spike_trains(period, FIRING_RATE, args.num_inputs if args.num_neurons < 1 else args.num_neurons)
            input_firing_times = [(spike_trains[l] + d).reshape(-1) for l, d in zip(neuron.sources, neuron.delays)]
            
            neuron.optimize_weights(firing_times, input_firing_times, period, weight_bound=args.weight_bound/100)
            
            if neuron.status != "optimal":
                num_errors += 1
                if num_errors > max_num_errors_99:
                    pd.DataFrame(list_of_dict).to_csv(os.path.join(exp_dir, "results.csv"), index=False)
                    print(f"Results saved to {exp_dir}", flush=True)
                    sys.exit(0)

        list_of_dict.append({
                            "num_neurons": np.inf if args.num_neurons < 1 else args.num_neurons,
                            "num_inputs": args.num_inputs,
                            "period": period,
                            "delay_max": args.delay_max,
                            "weight_bound": args.weight_bound/100,
                            "num_errors": num_errors
                        })
        period += 5
        