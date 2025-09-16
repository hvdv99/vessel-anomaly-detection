import os
import numpy as np
import random
import pickle as pkl
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Tuple, Iterator

LAT, LON, SOG, COG, SWH, MWD, MWP, U10, V10, TS, MMSI = np.arange(11)

def tgt_to_idx(tgt):
    transformed = {k: tgt[k].argmax(axis=1) for k in tgt.keys()}
    return transformed

class TrajectoryDataset(Dataset):
    def __init__(self,
                 ds_type,
                 lat_bins, lon_bins, sog_bins, cog_bins,
                 file_directory,
                 filename=None,
                 test_config=None,
                 include_weather=False):
        self.ds_type = ds_type
        self.test_config = test_config
        self.weather_included = include_weather

        # Bin configurations
        self.lat_bins = lat_bins
        self.lon_bins = lon_bins
        self.sog_bins = sog_bins
        self.cog_bins = cog_bins

        self.file_directory = file_directory
        self.filename = filename

        # Initialize data structures
        self.windows_per_trj = []
        self.labels = None
        self.trajectory_weather_stats = {}  # Store weather stats per trajectory
        self.window_to_trajectory = []  # Map window index to trajectory ID

        # Load and process dataset
        self.dataset = self.open_and_transform_dataset()
        self.batch_boundaries = self.define_batch_boundaries()

    def open_and_transform_dataset(self):
        self.dataset = self.open_dataset()
        return self.transform_dataset()


    def open_dataset(self):
        labels_path = None

        if self.ds_type in {'train', 'valid'}:
            path = os.path.join(self.file_directory, self.filename)
        elif self.ds_type == 'test':
            if not self.test_config:
                raise ValueError('When loading test set, config must be provided')

            test_type_num = self.test_config.get('type')
            test_type = 'type_' + str(test_type_num)
            d = self.test_config.get('d')
            r = self.test_config.get('r')
            p = self.test_config.get('p')

            if d:
                test_setup = '_'.join([str(v) for v in ['d', d, 'r', r, 'p', p]])
            else:
                test_setup = '_'.join([str(v) for v in ['r', r, 'p', p]])

            path = os.path.join(self.file_directory, test_type, test_setup, 'ct_test.pkl')
            labels_path = os.path.join(self.file_directory, test_type, test_setup, 'test_labels.pkl')
        else:
            raise ValueError("ds_type should either be 'train', 'valid' or 'test'")

        # Load data
        with open(path, 'rb') as dataset_file:
            tmp_dataset = pkl.load(dataset_file)

        if labels_path:
            with open(labels_path, 'rb') as labels_file:
                self.labels = pkl.load(labels_file)

        return tmp_dataset

    def transform_dataset(self):
        # Process trajectories
        windows = []

        for i, trajectory_data in enumerate(self.dataset):
            trj = trajectory_data['traj']
            mmsi = trajectory_data['mmsi']

            # Create unique trajectory ID
            trj_id = f"{int(mmsi)}_{i}"

            if self.weather_included:
                # Compute weather statistics once per trajectory
                weather_stats = self._compute_weather_stats(trj)

                # Store weather stats
                self.trajectory_weather_stats[trj_id] = weather_stats

            # Convert trajectory datapoint to bin indices
            bin_indices = np.array([self.row_to_bin_idx(row) for row in trj])

            # Create sliding windows
            trj_windows = self.make_windows(bin_indices, points_per_window=10)

            # Store the number of windows per trajectory
            num_windows = trj_windows.shape[0]
            self.windows_per_trj.append(num_windows)

            # Process each window
            for window in trj_windows:
                window = window.squeeze()

                # Create feature dictionary for transformer input
                binned_features = {
                    'lat': window[:, LAT],
                    'lon': window[:, LON],
                    'sog': window[:, SOG],
                    'cog': window[:, COG]
                }

                windows.append(binned_features)
                self.window_to_trajectory.append(trj_id)

        return windows

    def _compute_weather_stats(self, trajectory):
        """Compute weather statistics for a single trajectory."""
        weather_vars = [SWH, MWD, MWP, U10, V10]
        stats = {}

        for var_idx in weather_vars:
            var_data = trajectory[:, var_idx]

            mean = np.mean(var_data)
            std = np.std(var_data)

            stats[var_idx] = {
                'mean': float(mean),
                'std': float(std)
            }

        return stats

    def row_to_bin_idx(self, row):
        """Convert a row to bin indices."""
        # Clamp values at 1 to prevent overflow
        row = np.clip(row, 0, 0.9999)

        return (
            int(self.lat_bins * row[LAT]),
            int(self.lon_bins * row[LON]),
            int(self.sog_bins * row[SOG]),
            int(self.cog_bins * row[COG])
        )

    def make_windows(self, trj, points_per_window):
        """Create sliding windows from trajectory."""
        window_shape = (points_per_window, trj.shape[1])
        return np.lib.stride_tricks.sliding_window_view(
            trj, window_shape=window_shape, axis=(0, 1)
        )

    def define_batch_boundaries(self):
        """Define batch boundaries based on the number of windows for a trajectory."""
        start_boundry = (0, self.windows_per_trj[0] - 2)
        boundries = [start_boundry]

        for i in range(1, len(self.windows_per_trj)):
            start = boundries[-1][1] + 2
            end = start + self.windows_per_trj[i] - 2
            boundries.append((start, end))
        return boundries

    def get_weather_stats_vector(self, trj_id):
        """Convert weather statistics to a flattened vector for GMM input."""
        if trj_id not in self.trajectory_weather_stats:
            raise ValueError(f"Weather stats not found for trajectory {trj_id}")

        stats = self.trajectory_weather_stats[trj_id]
        vector = []

        # Flatten all statistics into a single vector
        for var_idx in [SWH, MWD, MWP, U10, V10]:
            var_stats = stats[var_idx]
            vector.extend([
                var_stats['mean'],
                var_stats['std']
            ])

        return np.array(vector, dtype=np.float32)

    def __len__(self):
        return sum(self.windows_per_trj)

    def __getitem__(self, idx):
        if idx >= len(self) - 1:
            raise IndexError(f"Index {idx} out of range")

        # Get source and target windows
        src_window = self.dataset[idx]
        tgt_window = self.dataset[idx + 1]

        # Get trajectory IDs
        src_trj_id = self.window_to_trajectory[idx]
        tgt_trj_id = self.window_to_trajectory[idx + 1]

        # Ensure windows are from the same trajectory
        if src_trj_id != tgt_trj_id:
            raise ValueError('Source and target windows are from different trajectories')

        # Convert to tensors with proper copying
        src_tensors = {}
        tgt_tensors = {}

        for key in ['lat', 'lon', 'sog', 'cog']:
            # Ensure arrays are writable before converting to tensors
            src_data = np.array(src_window[key], copy=True)
            tgt_data = np.array(tgt_window[key], copy=True)

            src_tensors[key] = torch.from_numpy(src_data).long()
            tgt_tensors[key] = torch.from_numpy(tgt_data).long()

        if self.weather_included:
            # Get weather statistics for this trajectory (for GMM input)
            weather_stats = self.get_weather_stats_vector(src_trj_id)
        else:
            weather_stats = torch.empty(0)

        return {
            'trajectory_id': src_trj_id,
            'src_window': src_tensors,      # For transformer input
            'tgt_window': tgt_tensors,      # For transformer target
            'weather_stats': weather_stats  # For GMM input
        }


class ExactBatchSampler(Sampler):
    """
    A batch sampler that takes exact start and end indices for each batch.
    """

    def __init__(self,
                 batch_boundaries: List[Tuple[int, int]],
                 shuffle_batches: bool = False,
                 drop_last: bool = False):
        self.batch_boundaries = batch_boundaries
        self.shuffle_batches = shuffle_batches
        self.drop_last = drop_last

        # Validate input
        for start_idx, end_idx in batch_boundaries:
            if not isinstance(start_idx, int) or not isinstance(end_idx, int):
                raise TypeError("Batch boundary indices must be integers")
            if start_idx < 0 or end_idx < 0:
                raise ValueError("Batch boundary indices must be non-negative")
            if start_idx >= end_idx:
                raise ValueError("End index must be greater than start index")

    def __iter__(self) -> Iterator[List[int]]:

        # Create list of batch indices
        batch_indices = list(range(len(self.batch_boundaries)))

        # Optionally shuffle batch order
        if self.shuffle_batches:
            random.shuffle(batch_indices)

        # Handle drop_last
        if self.drop_last and len(self.batch_boundaries) > 1:
            batch_sizes = [end - start for start, end in self.batch_boundaries]
            last_size = batch_sizes[-1]
            max_size = max(batch_sizes[:-1]) if len(batch_sizes) > 1 else last_size

            if last_size < max_size:
                batch_indices = batch_indices[:-1]

        # Yield batches
        for idx in batch_indices:
            start_idx, end_idx = self.batch_boundaries[idx]
            yield list(range(start_idx, end_idx))

    def __len__(self) -> int:
        if self.drop_last and len(self.batch_boundaries) > 1:
            batch_sizes = [end - start for start, end in self.batch_boundaries]
            last_size = batch_sizes[-1]
            max_size = max(batch_sizes[:-1])

            if last_size < max_size:
                return len(self.batch_boundaries) - 1
        return len(self.batch_boundaries)