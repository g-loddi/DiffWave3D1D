import pandas as pd
import numpy as np
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import pickle



def get_ring(n, center, positions):
    """
    Return the set of positions in the n-th ring, defined as the set of positions
    in a square grid of side length (2*n + 1) centered at 'center' minus all previous rings.
    """
    c0, c1 = center

    if n == 0:
        return {center} if center in positions else set()
    
    # Compute the positions in the n-th square ring directly
    ring_positions = set()
    
    # Top side (horizontal line)
    for j in range(c1 - n, c1 + n + 1):
        ring_positions.add((c0 - n, j))
    
    # Bottom side (horizontal line)
    for j in range(c1 - n, c1 + n + 1):
        ring_positions.add((c0 + n, j))
    
    # Left side (vertical line)
    for i in range(c0 - n + 1, c0 + n):
        ring_positions.add((i, c1 - n))
    
    # Right side (vertical line)
    for i in range(c0 - n + 1, c0 + n):
        ring_positions.add((i, c1 + n))
    
    # Return the intersection of ring_positions and the positions set
    return ring_positions.intersection(positions)

def compute_ring_correlations(df):
    """
    Compute the average Pearson correlation for the 1-ring and 2-ring for each cell
    in the grid, grouped by ['city', 'service', 'date'], and using the MultiIndex for efficient lookup.
    """
    result = []
    
    # Ensure the dataframe is sorted by 'city', 'service', 'date', 'position_0', 'position_1'
    df = df.sort_values(by=['city', 'service', 'date', 'position_0', 'position_1'])

    # Set the MultiIndex with ['city', 'service', 'date', 'position_0', 'position_1']
    df.set_index(['city', 'service', 'date', 'position_0', 'position_1'], inplace=True)
    
    # Precompute the time series column names.
    time_columns = [str(i) for i in range(96)]

    # Iterate over each group of ('city', 'service', 'date')
    for (city, service, date), group in tqdm(df.groupby(['city', 'service', 'date'])):

        # Create a lookup for this group using only (position_0, position_1) as the index.
        group_lookup = group.copy()
        group_lookup.index = group_lookup.index.droplevel([0, 1, 2])
        group_positions = set(group_lookup.index)

        # Iterate over the positions in this group (i.e., over each ('position_0', 'position_1'))
        sampled_group = group.sample(n=min(100, len(group)), random_state=42)  # limit to 100 rows max
        for idx, row in tqdm(sampled_group.iterrows(), total=len(sampled_group), leave=False, desc="Processing rows"):
        # for idx, row in tqdm(group.iterrows(), total=len(group), leave=False, desc="Processing rows"):
            # Extract the current cell's position and time series.
            pos = (row.name[3], row.name[4])
            current_series = row[time_columns].values

            # Get the 1-ring and 2-ring neighbors (restricted to the current group positions).
            neighbors_1_ring = get_ring(1, pos, group_positions)
            neighbors_2_ring = get_ring(2, pos, group_positions)
            
            correlations_1_ring = []
            for neighbor in neighbors_1_ring:
                neighbor_series = group_lookup.loc[neighbor][time_columns].values
                # Use np.corrcoef for a fast Pearson correlation calculation.
                if np.std(current_series) == 0 or np.std(neighbor_series) == 0:
                    corr = np.nan
                else:
                    corr = np.corrcoef(current_series, neighbor_series)[0, 1]
                correlations_1_ring.append(corr)
                    
            correlations_2_ring = []
            for neighbor in neighbors_2_ring:
                if neighbor in group_lookup.index:
                    neighbor_series = group_lookup.loc[neighbor][time_columns].values
                    if np.std(current_series) == 0 or np.std(neighbor_series) == 0:
                        corr = np.nan
                    else:
                        corr = np.corrcoef(current_series, neighbor_series)[0, 1]
                    correlations_2_ring.append(corr)
            
            avg_1_ring_corr = np.mean(correlations_1_ring) if correlations_1_ring else np.nan
            avg_2_ring_corr = np.mean(correlations_2_ring) if correlations_2_ring else np.nan
            
            result.append([city, service, date, pos[0], pos[1], avg_1_ring_corr, avg_2_ring_corr])
    
    # Build the result DataFrame.
    correlation_df = pd.DataFrame(
        result,
        columns=['city', 'service', 'date', 'position_0', 'position_1', 'corr_ring1', 'corr_ring2']
    )
    return correlation_df




for seed in [21,7]:
    correlations_dict = {}
    paths = {
        "generated_3d": f"3_services_data/generated_data_diffwave3d_seed{seed}.parquet",
        "generated_1d": f"3_services_data/generated_data_diffwave1d_seed{seed}.parquet",
        "generated_csdi": f"3_services_data/generated_data_csdi_seed{seed}.parquet",
        "generated_3d1d_64": f"3_services_data/generated_data_diffwave3d1d_64_seed{seed}.parquet",
        "real": f"3_services_data/real_data_seed{seed}.parquet"
    }

    for name, path in paths.items():
        print("processing: ", name)
        dataset = pd.read_parquet(path)
        correlations_dict[name] = compute_ring_correlations(dataset)

    # # --- Save the correlations_dict to disk ---
    save_path = f"3_services_data/correlations_dict_seed{seed}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(correlations_dict, f)
    print(f"Saved correlations_dict to {save_path}")
