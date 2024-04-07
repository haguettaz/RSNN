import os

import pandas as pd

FIRING_RATE = 0.2 # in number of spikes / tau_0 (outside guard period)
DELAY_MIN = 0.1  # in tau_0

PERIOD = 50 # in tau_0
FIRING_RATE = 0.2  # in number of spikes / tau_0 (outside guard period)

if __name__ == "__main__":
    list_of_df = []

    for batch_dir in os.listdir("data"):
        if not os.path.isdir(os.path.join("data", batch_dir)):
            continue

        for exp_dir in os.listdir(os.path.join("data", batch_dir)):
            if not os.path.isdir(os.path.join("data", batch_dir, exp_dir)):
                continue
            
            if not os.path.exists(os.path.join("data", batch_dir, exp_dir, "results.csv")):
                print(f"Results not found at {os.path.join('data', batch_dir, exp_dir, 'results.csv')}", flush=True)
                continue

            df = pd.read_csv(os.path.join("data", batch_dir, exp_dir, "results.csv"))
            print(f"Loaded {os.path.join('data', batch_dir, exp_dir, 'results.csv')}", flush=True)
            list_of_df.append(df)

            # if not os.path.exists(os.path.join("data", batch_dir, exp_dir, "neuron_results.csv")):
            #     print(f"Results not found at {os.path.join('data', batch_dir, exp_dir, 'neuron_results.csv')}", flush=True)
            #     continue

            # df = pd.read_csv(os.path.join("data", batch_dir, exp_dir, "neuron_results.csv"))
            # print(f"Loaded {os.path.join('data', batch_dir, exp_dir, 'neuron_results.csv')}", flush=True)
            # list_of_neuron_df.append(df)
        
    pd.concat(list_of_df).to_csv(os.path.join("data", "results.csv"), index=False)
    print(f"Results saved to {os.path.join('data', 'results.csv')}", flush=True)

    # pd.concat(list_of_neuron_df).to_csv(os.path.join("data", "neuron_results.csv"), index=False)
    # print(f"Results saved to {os.path.join('data', 'neuron_results.csv')}", flush=True)
