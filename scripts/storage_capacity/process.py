import os

import pandas as pd

if __name__ == "__main__":
    list_of_df = []

    for exp_dir in os.listdir("data"):
        if not os.path.isdir(os.path.join("data", exp_dir)):
            continue
        
        fpath = os.path.join("data", exp_dir, "results.csv")
        try:
            df = pd.read_csv(fpath)
            list_of_df.append(df)
            print(f"Loaded {fpath}", flush=True)
        except pd.errors.EmptyDataError:
            print(f"Empty file found at {fpath}", flush=True)
        
    pd.concat(list_of_df).to_csv(os.path.join("data", "results.csv"), index=False)
    print(f"Results saved to {os.path.join('data', 'results.csv')}", flush=True)