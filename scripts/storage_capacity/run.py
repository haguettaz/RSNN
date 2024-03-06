import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import trange

from rsnn.neuron.neuron import Neuron
from rsnn.spike_train.sampler import forward_sampling

# Remark: tau_min is the minimum distance between two consecutive spikes, i.e., the absolute refractory period.

FIRING_RATE = 0.2 # in number of spikes / tau_min (outside guard period)
DELAY_MIN = 0.1  # in tau_min

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Storage Capacity Simulation")
#     parser.add_argument("--num_exp", type=int)
#     parser.add_argument("--num_inputs", type=int)
#     parser.add_argument("--period", type=int)
#     parser.add_argument("--similarity", type=str) # low or high
#     parser.add_argument("--delay_max", type=int, default=10)
#     parser.add_argument("--weight_bound", type=int, default=20)  # [in %]

#     args = parser.parse_args()
    
#     if not args.similarity in {"low", "high"}:
#         raise ValueError("similarity should be one of low or high")

#     exp_dir = os.path.join("data", f"{args.num_inputs}_{args.period}_{args.similarity}_{args.delay_max}_{args.weight_bound}")
#     if not os.path.exists(exp_dir):
#         os.makedirs(exp_dir)
#         print(f"Created directory {exp_dir}", flush=True)
#     elif os.path.exists(os.path.join(exp_dir, "results.csv")):
#         print("Experiment already done", flush=True)
#         sys.exit(0)
    
#     if args.similarity == "low":
#         num_distinct_sources = args.num_inputs
#         neuron = Neuron(0, args.num_inputs, sources=np.arange(args.num_inputs))
#     else:
#         num_distinct_sources = 20
#         neuron = Neuron(0, args.num_inputs, sources=np.arange(args.num_inputs) % 20)
    
#     list_of_dict = []

#     for exp_idx in trange(args.num_exp):

#         spike_trains = forward_sampling(args.period, FIRING_RATE, num_distinct_sources)

#         np.random.shuffle(neuron.sources)
#         neuron.delays = np.random.uniform(low=DELAY_MIN, high=args.delay_max, size=args.num_inputs)

#         firing_times = (spike_trains[0]).copy()
#         input_firing_times = [(spike_trains[l] + d).reshape(-1) for l, d in zip(neuron.sources, neuron.delays)]
        
#         neuron.optimize_weights(firing_times, input_firing_times, args.period, weight_bound=args.weight_bound/100)
        
#         list_of_dict.append(
#             {
#                 "exp_idx": exp_idx,
#                 "num_inputs": args.num_inputs,
#                 "period": args.period,
#                 "similarity": args.similarity,
#                 "delay_max": args.delay_max,
#                 "weight_bound": args.weight_bound/100,
#                 "status":neuron.status
#             }
#         )

#     df = pd.DataFrame(list_of_dict)
#     df.to_csv(os.path.join(exp_dir, "results.csv"), index=False)
#     print(f"Results saved to {exp_dir}", flush=True)
#     sys.exit(0)

# 1st batch: to show the effect of the number of inputs on the storage capacity, in the limit L >> K
# num_neurons: 2000
# delay_max: 10
# num_inputs: 200:100:500
# periods: 25:25:200
        
# 2nd batch: to show the effect of reducing the number of neurons, in the limit K << L
# num_neurons: 10
# delay_max: 5
# num_inputs: 200:100:500
# periods: 25:25:200
        
# 3rd batch: to show how increasing the maximum delay improves the storage capacity in the limit K << L
# num_neurons: 10
# delay_max: 10
# num_inputs: 200:100:500
# periods: 25:25:200
    

# over 1000 experiments, for each number of inputs
# what is the longest spike train that is feasible more than 995 times?
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Storage Capacity Simulation")
    parser.add_argument("--num_exp", type=int, default=1000)
    parser.add_argument("--num_inputs", type=int)
    parser.add_argument("--similarity", type=str) # low or high
    parser.add_argument("--delay_max", type=int, default=10)
    parser.add_argument("--weight_bound", type=int, default=20)  # [in %]

    args = parser.parse_args()
    print(args, flush=True)
    
    if not args.similarity in {"low", "high"}:
        raise ValueError("similarity should be one of low or high")

    exp_dir = os.path.join("data", f"{args.num_inputs}_{args.similarity}_{args.delay_max}_{args.weight_bound}")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print(f"Created directory {exp_dir}", flush=True)
    elif os.path.exists(os.path.join(exp_dir, "results.csv")):
        print("Experiment already done", flush=True)
        sys.exit(0)
    
    if args.similarity == "low":
        num_distinct_sources = args.num_inputs
        neuron = Neuron(0, args.num_inputs, sources=np.arange(args.num_inputs))
    else:
        num_distinct_sources = 20
        neuron = Neuron(0, args.num_inputs, sources=np.arange(args.num_inputs) % 20)
    
    list_of_dict = []

    period = 20
    max_num_errors_95 = 0.05 * args.num_exp
    # max_num_errors_98 = 0.02 * args.num_exp
    # max_num_errors_99 = 0.01 * args.num_exp
    while True:
        num_errors = 0
        for exp_idx in trange(args.num_exp):

            spike_trains = forward_sampling(period, FIRING_RATE, num_distinct_sources)

            np.random.shuffle(neuron.sources)
            neuron.delays = np.random.uniform(low=DELAY_MIN, high=args.delay_max, size=args.num_inputs)

            firing_times = (spike_trains[0]).copy()
            input_firing_times = [(spike_trains[l] + d).reshape(-1) for l, d in zip(neuron.sources, neuron.delays)]
            
            neuron.optimize_weights(firing_times, input_firing_times, period, weight_bound=args.weight_bound/100)
            
            if neuron.status != "optimal":
                num_errors += 1
                if num_errors > max_num_errors_95:
                    pd.DataFrame(list_of_dict).to_csv(os.path.join(exp_dir, "results.csv"), index=False)
                    print(f"Results saved to {exp_dir}", flush=True)
                    sys.exit(0)

        list_of_dict.append({
                            "num_inputs": args.num_inputs,
                            "period": period,
                            "similarity": args.similarity,
                            "delay_max": args.delay_max,
                            "weight_bound": args.weight_bound/100,
                            "num_errors": num_errors
                        })
        period += 10
        