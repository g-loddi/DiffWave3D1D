import torch
import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import logging
import math
from datetime import datetime
import lmdb
from skimage.measure import block_reduce
import torch.distributed as dist
from torch.utils.data import DataLoader
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from config import MODEL_CONFIGS



def write_real_data(dataset, batch_size,seed):
    # Create a simple DataLoader without distributed sampling
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    filename = f'real_data_seed{seed}.parquet'
    writer = None  # ParquetWriter instance

    for metadata, x in tqdm(dl, desc="Generating samples"):
        # If the data is in patch form (4D tensor), flatten to individual cells.
        if len(x.shape) == 4:
            this_batch_size = x.shape[0]
            cells_in_patch = x.shape[2] * x.shape[3]
            x = x.permute(0, 2, 3, 1).reshape(this_batch_size * cells_in_patch, 96)
            metadata['position'] = metadata['position'].permute(0, 2, 3, 1).reshape(this_batch_size * cells_in_patch, 2)
            metadata['DoW'] = metadata['DoW'].repeat_interleave(cells_in_patch)
            metadata['city'] = [item for item in metadata['city'] for _ in range(cells_in_patch)]
            metadata['date'] = [item for item in metadata['date'] for _ in range(cells_in_patch)]
            metadata['service'] = [item for item in metadata['service'] for _ in range(cells_in_patch)]

        # Build rows for the current batch.
        rows = []
        # Loop over each sample in the batch.
        for j in range(x.shape[0]):
            city = str(metadata['city'][j])      # Ensure city is a string.
            service = str(metadata['service'][j])  # Ensure service is a string.
            date = str(metadata['date'][j])        # Ensure date is a string.

            # Convert the position to a tuple and then split it.
            position = metadata['position'][j]
            pos_tuple = tuple(map(int, position.tolist()))
            if len(pos_tuple) < 2:
                raise ValueError("Position tuple must contain at least 2 elements.")
            pos_0, pos_1 = pos_tuple[0], pos_tuple[1]

            # Get the generated data as a NumPy array of length 96.
            gen_data = x[j].cpu().numpy()
            row = {
                'city': city,
                'service': service,
                'date': date,
                'position_0': pos_0,
                'position_1': pos_1,
            }
            # Add each generated value as a separate column.
            for i in range(96):
                row[str(i)] = gen_data[i]
            rows.append(row)

        # Convert the batch rows to a DataFrame.
        df_batch = pd.DataFrame(rows)

        # Convert DataFrame to a PyArrow table and write.
        table = pa.Table.from_pandas(df_batch)
        if writer is None:
            writer = pq.ParquetWriter(filename, table.schema)
        writer.write_table(table)
        # Optionally, clear variables that are no longer needed
        del df_batch, table, rows

    if writer is not None:
        writer.close()

    return None

#torchvision ema implementation
#https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_sample(synthetic_time_series, real_samples, metadata, save_path, filename="time_series_plot.png"):
    """
    Plot synthetic and real time series with different line styles.

    Args:
        synthetic_time_series: A tensor or array representing the synthetic time series.
        real_samples: A list of real samples, each containing a time series.
        metadata: Dictionary containing position, DoW, and service information.
        save_path: The directory where the plot will be saved.
        filename: The name of the file to save the plot as.
    """
    # Extract the synthetic time series
    if len(synthetic_time_series[0].shape) == 3:
        synthetic_time_series = synthetic_time_series[0, :, 0, 0].cpu().numpy() 
    elif len(synthetic_time_series[0].shape) == 1:
        synthetic_time_series = synthetic_time_series[0].cpu().numpy()

    # Create a figure for the plot
    plt.figure(figsize=(12, 8))

    # Plot the real time series with thin lines
    for real_sample in real_samples:
        if len(real_sample.shape) == 3:
            real_time_series = real_sample[:, 0, 0]  # Extract the time series from each real_sample
        elif len(real_sample.shape) == 1:
            real_time_series = real_sample
        plt.plot(real_time_series, color='gray', linewidth=0.5, linestyle='-', label='_nolegend_')  # Thin gray lines

    # Plot the synthetic time series with a thicker line
    plt.plot(synthetic_time_series, color='blue', linewidth=2, linestyle='-', label='Synthetic Time Series')

    # Create title with metadata
    position_str = f"Position: {metadata['position']}"
    dow_str = f"DoW: {metadata['DoW']}"
    service_str = f"Service: {metadata['service']}"
    title = f"{position_str} | {dow_str} | {service_str}"

    # Add labels, title, legend, and grid
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save the plot to the specified folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Create the folder if it doesn't exist
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file)

    # Clear the plot after saving to avoid overlap in future plots
    plt.close()


def plot_denoised_sample(real_sample, noised_sample, denoised_sample, save_path="./debug_plots", filename="denoised_plot.png"):
        # Extract the real, noised, and denoised time series
        if len(real_sample.shape) == 3:
            real_sample_time_series = real_sample[:, 0, 0].cpu().numpy()
            noised_sample_time_series = noised_sample[:, 0, 0].cpu().numpy()
            denoised_sample_time_series = denoised_sample[:, 0, 0].cpu().numpy()
        else:
            real_sample_time_series = real_sample.cpu().numpy()
            noised_sample_time_series = noised_sample.cpu().numpy()
            denoised_sample_time_series = denoised_sample.cpu().numpy()

        # Plotting the time series
        plt.figure(figsize=(10, 6))
        plt.plot(real_sample_time_series, marker='o', linestyle='-', label='Real Time Series')
        plt.plot(noised_sample_time_series, marker='o', linestyle='-', label='Noised Time Series')
        plt.plot(denoised_sample_time_series, marker='o', linestyle='-', label='Denoised Time Series')

        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title('Time Series Plot')
        plt.legend()
        plt.grid(True)

        # Save the plot to the specified folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # Create the folder if it doesn't exist
        save_file = os.path.join(save_path, filename)
        plt.savefig(save_file)

        # Clear the plot after saving to avoid overlap in future plots
        plt.close()

class NetmobDatasetLMDBUnified(Dataset):
    def __init__(
        self,
        data_dir: str,
        scaling: str = None,
        log_transform: bool = False,
        model_name: str = 'diffwave3d1d',  # Default model name
        threshold: int = 5,
        db_dir: str = "lmdb_data",  # directory where the LMDB file will be stored
        use_precomputed: bool = True,
        map_size: int = 50 * 1024**3,    # LMDB map_size (50gb)
        build_index: bool = True
    ):
        """
        Unified dataset class for Netmob that stores all samples in a single LMDB file.
        
        - For single-cell patches (image_size == 1):
            The patch is of shape (96, 1, 1) and is later squeezed to (96,).
            Each sample's metadata contains a "cell_index" (the sample's index).
        - For multi-cell patches (image_size > 1):
            Each patch is stored with shape (96, image_size, image_size).
        
        All samples are stored in one LMDB file (e.g. "all_samples.lmdb") with keys "sample_0", "sample_1", etc.
        Precomputed metadata and stats (global_min, global_max) are stored in pickle files for faster reload.
        """
        self.data_dir = data_dir
        self.model_name = model_name
        # Get image_size from MODEL_CONFIGS
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_name}")
        self.image_size = MODEL_CONFIGS[model_name]['image_size']
        self.patch_rows = self.image_size
        self.patch_cols = self.image_size
        self.threshold = threshold
        self.log_transform = log_transform
        self.scaling = scaling
        self.use_precomputed = use_precomputed
        self.map_size = map_size
        self.build_index = build_index

        # Define a directory to hold our LMDB file and metadata.
        self.db_dir = 'lmdb_data_'+str(self.image_size)
        os.makedirs(self.db_dir, exist_ok=True)
        # Our single LMDB file:
        self.db_path = os.path.join(self.db_dir, "all_samples.lmdb")

        # These files hold precomputed metadata and stats.
        self.metadata_file = os.path.join(self.db_dir, "metadata.pkl")
        self.stats_file = os.path.join(self.db_dir, "stats.pkl")
        self.discarded_file = os.path.join(self.db_dir, "discarded_samples.pkl")

        # Global statistics.
        self.global_min = float('inf')
        self.global_max = float('-inf')
        # List of per-sample metadata dictionaries.
        self.samples = []
        self.discarded_samples = []

        # We also need city dimensions.
        self.city_dims = {
            'Bordeaux': (334, 342),
            'Clermont-Ferrand': (208, 268),
            'Dijon': (195, 234),
            'Grenoble': (409, 251),
            'Lille': (330, 342),
            'Lyon': (426, 287),
            'Mans': (228, 246),
            'Marseille': (211, 210),
            'Metz': (226, 269),
            'Montpellier': (334, 327),
            'Nancy': (151, 165),
            'Nantes': (277, 425),
            'Nice': (150, 214),
            'Orleans': (282, 256),
            'Paris': (409, 346),
            'Rennes': (423, 370),
            'Saint-Etienne': (305, 501),
            'Strasbourg': (296, 258),
            'Toulouse': (280, 347),
            'Tours': (251, 270)
        }

        self.SERVICE_MAPPING = {
        "Instagram": 0,
        "Netflix": 1,
        "WhatsApp": 2
    }


        # LMDB environment (will be lazily initialized per worker)
        self.env = None

        if self.use_precomputed and os.path.exists(self.db_path) and os.path.exists(self.metadata_file):
            # Load metadata and stats from pickle files.
            with open(self.metadata_file, "rb") as f:
                self.samples = pickle.load(f)
            with open(self.stats_file, "rb") as f:
                stats = pickle.load(f)
                self.global_min = stats["global_min"]
                self.global_max = stats["global_max"]
            print(f"Loaded {len(self.samples)} samples from LMDB with global_min={self.global_min} and global_max={self.global_max}")
        else:
            # Precompute: iterate over raw data and store all valid samples in one LMDB file.
            all_samples = []  # Each element is a dict: {'patch': patch, 'metadata': metadata}
            cities = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
            for city in cities:
                city_path = os.path.join(self.data_dir, city)
                geojson_file = os.path.join(self.data_dir, f"{city}.geojson")
                # Use _generate_anchors which handles both single-cell and multi-cell.
                anchors = self._generate_anchors(geojson_file, city)
                if not anchors:
                    continue
                services = [s for s in os.listdir(city_path) if os.path.isdir(os.path.join(city_path, s))]
                for service in services:
                    service_path = os.path.join(city_path, service)
                    dates = [d for d in os.listdir(service_path) if os.path.isdir(os.path.join(service_path, d))]
                    for date in dates:
                        print(f"Processing {city} - {service} - {date}")
                        txt_path = self._get_txt_path(city, service, date)
                        traffic_matrix = self._load_traffic_data(txt_path, city)
                        if traffic_matrix is None:
                            continue
                        for anchor in anchors:
                            i, j = anchor
                            patch = traffic_matrix[:, i:i+self.patch_rows, j:j+self.patch_cols]
                            metadata = {
                                'city': city,
                                'service': service,
                                'date': date,
                                'anchor': (i, j),  # store as a tuple
                                'position': self._create_patch_array(anchor),  # shape depends on patch size
                                'DoW': self._get_day_of_week(date)
                            }
                            # Validate patch shape.
                            if patch.shape != (96, self.patch_rows, self.patch_cols):
                                continue
                            if np.isnan(patch).any():
                                self.discarded_samples.append(metadata)
                                continue
                            # Check for "bad cells" based on zeros.
                            zero_count_patch = np.sum(patch == 0, axis=0)
                            summed_patch = np.sum(patch, axis=0)
                            binary_matrix = ((zero_count_patch >= self.threshold) & (summed_patch > 0)).astype(int)
                            if np.any(binary_matrix > 0):
                                self.discarded_samples.append(metadata)
                                continue
                            # For single-cell, record cell_index.
                            if self.patch_rows == 1 and self.patch_cols == 1:
                                metadata["cell_index"] = len(all_samples)
                            all_samples.append({'patch': patch, 'metadata': metadata})
                            # Update global statistics.
                            patch_min = np.nanmin(patch)
                            patch_max = np.nanmax(patch)
                            self.global_min = min(self.global_min, patch_min)
                            self.global_max = max(self.global_max, patch_max)
            # Write all samples into a single LMDB file.
            env = lmdb.open(self.db_path, map_size=self.map_size)
            with env.begin(write=True) as txn:
                for i, sample in enumerate(all_samples):
                    key = f"sample_{i}".encode('utf-8')
                    txn.put(key, pickle.dumps(sample))
                # Optionally store the total number of samples.
                txn.put(b"length", pickle.dumps(len(all_samples)))
            env.close()
            # Save metadata and stats separately for quick loading.
            self.samples = [sample['metadata'] for sample in all_samples]
            with open(self.metadata_file, "wb") as f:
                pickle.dump(self.samples, f)
            with open(self.stats_file, "wb") as f:
                pickle.dump({"global_min": self.global_min, "global_max": self.global_max}, f)
            print(f"Created LMDB with {len(self.samples)} samples; global_min={self.global_min}, global_max={self.global_max}")


        self.anchor_index = self.build_inverse_index('anchor') if build_index else None
        self.DoW_index = self.build_inverse_index('DoW') if build_index else None
        self.service_index = self.build_inverse_index('service') if build_index else None

        # Print dataset statistics after initialization
        # if build_index:
            # self.print_dataset_statistics()

    def __len__(self):
        return len(self.samples)

    def _init_env(self):
        # Each worker should open its own LMDB environment.
        if self.env is None:
            self.env = lmdb.open(self.db_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, idx):
        self._init_env()
        key = f"sample_{idx}".encode('utf-8')
        with self.env.begin() as txn:
            sample_data = txn.get(key)
        sample = pickle.loads(sample_data)
        patch = sample['patch']
        metadata = sample['metadata']

        # Convert 'service' to an integer index using SERVICE_MAPPING
        # metadata['service'] = torch.tensor([self.SERVICE_MAPPING.get(metadata['service'], -1)], dtype=torch.long)
        metadata['service'] = self.SERVICE_MAPPING.get(metadata['service'], -1)
        
        # Apply log transform and scaling if requested.
        if self.log_transform:
            patch = np.log1p(patch)
            transformed_min = math.log1p(self.global_min)
            transformed_max = math.log1p(self.global_max)
        else:
            transformed_min = self.global_min
            transformed_max = self.global_max
        if self.scaling == 'minmax':
            if not np.isclose(transformed_min, transformed_max):
                patch = 2 * (patch - transformed_min) / (transformed_max - transformed_min) - 1
            else:
                patch = np.zeros_like(patch)
        # For single-cell patches, squeeze the spatial dimensions.
        if self.patch_rows == 1 and self.patch_cols == 1:
            patch = np.squeeze(patch)  # now shape: (96,)
        tensor = torch.tensor(patch, dtype=torch.float)
        return metadata, tensor

    def _get_txt_path(self, city, service, date):
        return os.path.join(self.data_dir, city, service, date, f"{city}_{service}_{date}_DL.txt")

    def _get_day_of_week(self, date_str):
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        return date_obj.weekday()

    def _generate_anchors(self, geojson_file, city):
        """
        Generates anchor points by pooling the binary city mask instead of sliding window.
        A patch is valid if the average pooling result is 1 (i.e., fully filled).
        """
        if not os.path.isfile(geojson_file):
            logging.warning(f"GeoJSON file missing: {geojson_file}")
            return []

        with open(geojson_file, 'r') as f:
            try:
                geojson = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing GeoJSON {geojson_file}: {e}")
                return []

        tile_ids = []
        for feature in geojson.get('features', []):
            tile_id = feature.get('properties', {}).get('tile_id')
            if tile_id is not None:
                try:
                    tile_ids.append(int(tile_id))
                except ValueError:
                    continue

        if not tile_ids:
            logging.warning(f"No valid tile_ids in {geojson_file}")
            return []

        if city not in self.city_dims:
            logging.warning(f"City {city} not in city_dims.")
            return []

        n_rows, n_cols = self.city_dims[city]
        city_mask = np.zeros((n_rows, n_cols), dtype=np.uint8)
        for tid in tile_ids:
            row_idx = tid // n_cols
            col_idx = tid % n_cols
            if 0 <= row_idx < n_rows and 0 <= col_idx < n_cols:
                city_mask[row_idx, col_idx] = 1

        if self.patch_rows == 1 and self.patch_cols == 1:
            return [(r, c) for r in range(n_rows) for c in range(n_cols) if city_mask[r, c] == 1]

        # Truncate city_mask to make it divisible by patch size
        trunc_rows = (n_rows // self.patch_rows) * self.patch_rows
        trunc_cols = (n_cols // self.patch_cols) * self.patch_cols
        city_mask_cropped = city_mask[:trunc_rows, :trunc_cols]

        # Average pooling
        pooled = block_reduce(city_mask_cropped, block_size=(self.patch_rows, self.patch_cols), func=np.mean)

        # Find anchors (value == 1 means all ones in the block)
        
        anchors = []
        for i in range(pooled.shape[0]):
            for j in range(pooled.shape[1]):
                if pooled[i, j] == 1.0:
                    anchor_row = i * self.patch_rows
                    anchor_col = j * self.patch_cols
                    anchors.append((anchor_row, anchor_col))

        return anchors

    def _load_traffic_data(self, txt_path, city):
        if not os.path.isfile(txt_path):
            logging.warning(f"Traffic file missing: {txt_path}")
            return None
        try:
            with open(txt_path, 'r') as f:
                line = f.readline()
            num_cols = len(line.strip().split())
            if num_cols < 2:
                logging.warning(f"File {txt_path} has insufficient columns.")
                return None
            df = pd.read_csv(
                txt_path,
                sep=' ',
                header=None,
                names=['tile_id'] + [f"{h:02d}:{m:02d}" for h in range(24) for m in (0,15,30,45)]
            )
            if df.shape[1] < 97:
                logging.warning(f"File {txt_path} does not have 96 time intervals.")
                return None
            if city not in self.city_dims:
                logging.warning(f"City {city} not in city_dims.")
                return None
            n_rows, n_cols = self.city_dims[city]
            traffic_matrix = np.zeros((96, n_rows, n_cols), dtype=np.float32)
            for _, row in df.iterrows():
                try:
                    tile_id = int(row['tile_id'])
                except ValueError:
                    continue
                r_idx = tile_id // n_cols
                c_idx = tile_id % n_cols
                if 0 <= r_idx < n_rows and 0 <= c_idx < n_cols:
                    traffic_values = row[1:97].astype(float).values
                    traffic_matrix[:, r_idx, c_idx] = traffic_values
            return traffic_matrix
        except Exception as e:
            logging.error(f"Error loading data from {txt_path}: {e}")
            return None

    def build_inverse_index(self, metadata_key):
        inverse_index = defaultdict(list)
        for idx, metadata in enumerate(self.samples):
            value = metadata.get(metadata_key, None)
            if value is None:
                logging.warning(f"Metadata key '{metadata_key}' not found in sample index {idx}.")
                continue
            if isinstance(value, (torch.Tensor, np.ndarray)):
                value = tuple(value.flatten().tolist() if isinstance(value, torch.Tensor) else value.flatten())
            elif isinstance(value, list):
                value = tuple(value)
            inverse_index[value].append(idx)
        return dict(inverse_index)

    def _create_patch_array(self, anchor):
        i, j = anchor
        if self.patch_rows == 1 and self.patch_cols == 1:
            return np.array([i, j])
        else:
            row_indices = np.arange(i, i + self.patch_rows)[:, None]
            row_layer = np.tile(row_indices, (1, self.patch_cols))
            col_indices = np.arange(j, j + self.patch_cols)[None, :]
            col_layer = np.tile(col_indices, (self.patch_rows, 1))
            return np.stack([row_layer, col_layer], axis=0)

    def print_dataset_statistics(self):
        """Print statistics about the number of samples per anchor, DoW, and service."""
        if not self.build_index:
            print("Index building is disabled. Cannot print statistics.")
            return

        # Print anchor statistics
        print("\nAnchor Statistics:")
        for anchor, indices in self.anchor_index.items():
            print(f"Anchor {anchor}: {len(indices)} samples")

        # Print DoW statistics
        print("\nDay of Week Statistics:")
        for dow, indices in self.DoW_index.items():
            print(f"DoW {dow}: {len(indices)} samples")

        # Print service statistics
        print("\nService Statistics:")
        for service, indices in self.service_index.items():
            print(f"Service {service}: {len(indices)} samples")

        # Print total statistics
        total_samples = len(self.samples)
        print(f"\nTotal samples in dataset: {total_samples}")