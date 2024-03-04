import os

import pandas as pd

if __name__ == "__main__":
    list_of_df = []

    for exp_dir in os.listdir("data"):
        if not os.path.isdir(os.path.join("data", exp_dir)):
            continue

        df = pd.read_csv(os.path.join("data", exp_dir, "results.csv"))
        list_of_df.append(df)
        
    pd.concat(list_of_df).to_csv(os.path.join("data", "results.csv"), index=False)
    print(f"Results saved to {os.path.join('data', 'results.csv')}", flush=True)