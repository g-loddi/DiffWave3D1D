import pickle


import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance_nd
from tqdm import tqdm
import re
import glob
import os
import pickle

folder_path = '3_services_data'


def conditional_wasserstein_distance(df1, df2):
    """
    Compute an average Wasserstein distance between two distributions
    of time series, conditioned on (DoW, position_0, position_1).

    Parameters:
        df1, df2 : pd.DataFrame
            Must contain columns 'DoW', 'position_0', 'position_1',
            plus 96 numeric columns named "0" through "95" for time steps.

    Returns:
        avg_distance : float
            The average group-wise Wasserstein distance, averaged across
            all groups found in df1 that also exist in df2.
    """
    # Convert 'date' to datetime and add a day-of-week column.
    for df in [df1, df2]:
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df["DoW"] = df["date"].dt.dayofweek.apply(lambda x: 0 if x <= 4 else 1)


    # Identify group columns
    group_cols = ["service","DoW", "position_0", "position_1"]

    # Identify the 96 time series columns (as strings "0".."95")
    ts_cols = [str(i) for i in range(96)]

    # Group each DataFrame by (DoW, position_0, position_1)
    df1_groups = df1.groupby(group_cols)
    df2_groups = df2.groupby(group_cols)

    rows = []
    
    # For each group key in df1, try to find the matching group in df2
    for key, group1 in tqdm(df1_groups, desc="Computing distances"):
            if key not in df2_groups.groups:
                continue  # skip if df2 doesn't have this group
            group2 = df2_groups.get_group(key)

            # key is a tuple like (service, DoW, position_0, position_1)
            service, dow_val, pos0, pos1 = key

            # Flatten the time series into 1D arrays
            arr1 = group1[ts_cols].values
            arr2 = group2[ts_cols].values

            # Compute Wasserstein distance
            dist = wasserstein_distance_nd(arr1, arr2)

            # Build one output row
            rows.append({
                "service": service,
                "DoW": dow_val,
                "position_0": pos0,
                "position_1": pos1,
                "wasserstein_dist": dist
            })

    # Convert rows to a DataFrame
    if not rows:
        return pd.DataFrame(columns=["service", "date", "DoW", "position_0", "position_1", "wasserstein_dist"])

    df_result = pd.DataFrame(rows)
    return df_result



seed_list = [42,21, 7]
model_name_list = ['diffwave1d', 'diffwave3d', 'diffwave3d1d_64', 'csdi']

for seed in seed_list:
    real_path = os.path.join(folder_path, f"real_data_seed{seed}.parquet")
    df_real = pd.read_parquet(real_path)

    for model_name in model_name_list:
        gen_file = f"generated_data_{model_name}_seed{seed}.parquet"
        gen_path = os.path.join(folder_path, gen_file)

        if not os.path.exists(gen_path):
            print(f"Missing generated file: {gen_file}, skipping...")
            continue

        df_generated = pd.read_parquet(gen_path)
        distances = conditional_wasserstein_distance(df_real, df_generated)

        avg_dist = np.mean(distances['wasserstein_dist'])

        # Save distances to file
        out_path = os.path.join(folder_path, f"wasserstein_distances_{model_name}_seed{seed}_weekendweekday.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(distances, f)

        # Print average distance
        print(f"Average group-wise Wasserstein distance Real vs {model_name} (seed {seed}): {avg_dist:.4f}")

    
        del df_generated, distances

    del df_real



results = {}

folder_path = '3_services_data'
seed_list = [42,21, 7]
model_name_list = ['diffwave1d', 'diffwave3d', 'diffwave3d1d_64', 'csdi']

for seed in seed_list:
    for model_name in model_name_list:
        filepath = os.path.join(folder_path,f"wasserstein_distances_{model_name}_seed{seed}_weekendweekday.pkl")
        with open(filepath, "rb") as f:
                distances = pickle.load(f)
        avg_dist = np.mean(distances['wasserstein_dist'])
        results.append({"model_name":model_name,
                        "seed":seed,
                        "w_dist":avg_dist})

df = pd.DataFrame(results)
out_csv = os.path.join(folder_path, "wasserstein_summary.csv")
df.to_csv(out_csv, index=False)

print(f"Saved summary to {out_csv}")

        