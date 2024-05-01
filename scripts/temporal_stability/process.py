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

            params = batch_dir.split("_")
            df["period"] = int(params[0])
            df["num_neurons"] = int(params[1])
            df["delay_max"] = int(params[2])
            df["slope_min"] = int(params[3])
            df["weight_bound"] = int(params[4])
            if "l1" in params:
                df["weight_regularization"] = "l1"
            elif "l2" in params:
                df["weight_regularization"] = "l2"
            else:
                df["weight_regularization"] = None
            df["exp_idx"] = int(exp_dir.split("_")[-1])
            
            list_of_df.append(df)
        
    pd.concat(list_of_df).to_csv(os.path.join("data", "results.csv"), index=False)
    print(f"Results saved to {os.path.join('data', 'results.csv')}", flush=True)
