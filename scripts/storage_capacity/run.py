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

WEIGHT_BOUND = 0.2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Storage Capacity Simulation")
    parser.add_argument("--num_exp", type=int, default=1000)
    parser.add_argument("--num_neurons", type=int, default=0) # 0 => num_neurons = num_inputs
    parser.add_argument("--num_inputs", type=int)
    parser.add_argument("--delay_max", type=int, default=10)
    parser.add_argument("--period_min", type=int, default=20) # in %


    args = parser.parse_args()
    print(args, flush=True)

    exp_dir = os.path.join("data", f"{args.num_neurons}_{args.num_inputs}_{args.delay_max}")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print(f"Created directory {exp_dir}", flush=True)
    elif os.path.exists(os.path.join(exp_dir, "results.csv")):
        print("Experiment already done", flush=True)
        sys.exit(0)

    rng = np.random.default_rng()
    
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
                neuron.sources = rng.choice(args.num_neurons, args.num_inputs)
            neuron.delays = rng.uniform(low=DELAY_MIN, high=args.delay_max, size=args.num_inputs)

            spike_train = sample_spike_trains(period, FIRING_RATE)
            spike_trains = sample_spike_trains(period, FIRING_RATE, args.num_inputs if args.num_neurons < 1 else args.num_neurons)
            input_spike_trains = [(spike_trains[l] + d).reshape(-1) for l, d in zip(neuron.sources, neuron.delays)]
            
            neuron.optimize_weights(spike_train, input_spike_trains, period, weight_bound=WEIGHT_BOUND)
            
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
                            "num_errors": num_errors
                        })
        period += 5
        