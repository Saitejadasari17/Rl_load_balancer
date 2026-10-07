"""
Azure Serverless Functions Dataset Integration
Loads and processes the Microsoft Azure Functions Invocation dataset (July 2019)
for realistic load balancing simulation.

Supports loading multiple CSV files from the extracted dataset folder:
  - invocations_per_function_md.anon.d01.csv through d14.csv
  - function_durations_percentiles.anon.d01.csv through d14.csv
  - app_memory_percentiles.anon.d01.csv through d12.csv

Dataset: https://github.com/Azure/AzurePublicDataset
Paper: "Serverless in the Wild" (ATC 2020)
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from glob import glob


class AzureDatasetLoader:
    """Load and process Azure Functions dataset traces from multiple CSV files."""

    def __init__(self, data_dir=None):
        """
        Initialize dataset loader.

        Args:
            data_dir: Path to folder containing extracted Azure CSV files.
                      Defaults to 'data/extracted'.
        """
        self.data_dir = data_dir or "data/extracted"
        self.invocation_data = None
        self.duration_data = None
        self.traffic_pattern = None

    def load_invocation_traces(self, day=1):
        """
        Load invocation data for a specific day.

        The invocation CSV format:
            HashOwner, HashApp, HashFunction, Trigger,
            1, 2, 3, ..., 1440  (per-minute invocation counts)

        Args:
            day: Day number (1-14)

        Returns:
            numpy array of per-minute total invocations
        """
        filename = f"invocations_per_function_md.anon.d{day:02d}.csv"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            print(f"Invocation file not found: {filepath}")
            return None

        print(f"Loading invocations for Day {day}: {filename}")
        print(f"  File size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")

        # Read CSV — first 4 columns are metadata, rest are minute-by-minute counts
        df = pd.read_csv(filepath)
        print(f"  Functions loaded: {len(df):,}")

        # Columns 4 onwards are the 1440 minute-by-minute invocation counts
        # Column names are '1', '2', ..., '1440' (string numbers)
        minute_cols = [col for col in df.columns if col.isdigit()]

        if not minute_cols:
            # Try numeric column detection as fallback
            minute_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        print(f"  Time columns: {len(minute_cols)} minutes")

        # Sum invocations across ALL functions for each minute
        # This gives total traffic per minute across the entire platform
        per_minute_traffic = df[minute_cols].sum(axis=0).values.astype(float)

        print(f"  Total invocations: {per_minute_traffic.sum():,.0f}")
        print(f"  Peak minute: {per_minute_traffic.max():,.0f} invocations")
        print(f"  Avg per minute: {per_minute_traffic.mean():,.0f} invocations")

        self.invocation_data = per_minute_traffic
        return per_minute_traffic

    def load_duration_data(self, day=1):
        """
        Load function duration percentiles for a specific day.

        CSV format: HashOwner, HashApp, HashFunction, Average, Count,
                    Minimum, Maximum, percentile_Average_0, ..., percentile_Average_100

        Returns:
            DataFrame with duration statistics
        """
        filename = f"function_durations_percentiles.anon.d{day:02d}.csv"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            print(f"Duration file not found: {filepath}")
            return None

        print(f"Loading durations for Day {day}: {filename}")
        df = pd.read_csv(filepath)
        print(f"  Functions with duration data: {len(df):,}")

        self.duration_data = df
        return df

    def get_traffic_pattern(self, day=1, n_servers=3):
        """
        Convert Azure invocation data into a normalized load pattern
        suitable for the RL environment.

        Args:
            day: Which day to load (1-14)
            n_servers: Number of servers (for context)

        Returns:
            numpy array of normalized load values [0.05, 0.3] per minute
        """
        if self.invocation_data is None:
            data = self.load_invocation_traces(day)
            if data is None:
                print("Falling back to synthetic traffic pattern")
                return self._generate_synthetic_pattern(1440, n_servers)
        else:
            data = self.invocation_data

        # Normalize invocation counts to [0.05, 0.3] load range
        data_min = data.min()
        data_max = data.max()

        if data_max - data_min < 1e-6:
            # All values are the same — use flat load
            normalized = np.full_like(data, 0.15)
        else:
            normalized = 0.05 + (data - data_min) / (data_max - data_min) * 0.25

        self.traffic_pattern = normalized
        return normalized

    def get_execution_time_distribution(self, day=1):
        """
        Get execution time distribution from duration data.
        Used to create heterogeneous request loads.

        Returns:
            numpy array of average execution times (ms), normalized
        """
        if self.duration_data is None:
            df = self.load_duration_data(day)
            if df is None:
                return None
        else:
            df = self.duration_data

        if 'Average' not in df.columns:
            return None

        exec_times = df['Average'].dropna().values.astype(float)
        # Remove outliers (above 99th percentile)
        p99 = np.percentile(exec_times, 99)
        exec_times = np.clip(exec_times, 0, p99)

        # Normalize to [0.05, 0.3]
        if exec_times.max() > 0:
            normalized = 0.05 + (exec_times / exec_times.max()) * 0.25
        else:
            normalized = np.full_like(exec_times, 0.15)

        return normalized

    def _generate_synthetic_pattern(self, duration_minutes, n_servers):
        """Generate synthetic traffic pattern when dataset unavailable."""
        pattern = []
        for t in range(duration_minutes):
            base_load = np.random.uniform(0.05, 0.15)

            # Simulate diurnal pattern (24-hour cycle over 1440 minutes)
            hour = (t / 60) % 24
            if 9 <= hour <= 17:  # Business hours
                burst_factor = 1.5
            else:
                burst_factor = 1.0

            # Random spike (1% chance per minute)
            if np.random.random() < 0.01:
                burst_factor *= 2.0

            load = np.clip(base_load * burst_factor, 0.05, 0.3)
            pattern.append(load)

        return np.array(pattern)

    def get_statistics(self, day=1):
        """Print comprehensive dataset statistics."""
        print("\n" + "=" * 60)
        print(f"  AZURE FUNCTIONS DATASET — DAY {day}")
        print("=" * 60)

        # Load data if needed
        traffic = self.get_traffic_pattern(day)
        durations = self.load_duration_data(day)

        if self.invocation_data is not None:
            data = self.invocation_data
            print(f"\n📊 Invocations (per-minute aggregated):")
            print(f"   Total invocations:  {data.sum():>15,.0f}")
            print(f"   Peak minute:        {data.max():>15,.0f}")
            print(f"   Min minute:         {data.min():>15,.0f}")
            print(f"   Mean per minute:    {data.mean():>15,.0f}")
            print(f"   Std deviation:      {data.std():>15,.0f}")
            print(f"   Duration:           {len(data):>15,} minutes ({len(data)/60:.1f} hours)")

        if durations is not None and 'Average' in durations.columns:
            avg_dur = durations['Average'].dropna()
            print(f"\n⏱️  Execution Durations (ms):")
            print(f"   Mean:               {avg_dur.mean():>15.1f} ms")
            print(f"   Median:             {avg_dur.median():>15.1f} ms")
            print(f"   P99:                {avg_dur.quantile(0.99):>15.1f} ms")
            print(f"   Max:                {avg_dur.max():>15.1f} ms")

        print()

    def list_available_files(self):
        """List all available dataset files in the data directory."""
        if not os.path.exists(self.data_dir):
            print(f"Data directory not found: {self.data_dir}")
            return {}

        files = {
            'invocations': sorted(glob(os.path.join(self.data_dir, "invocations_per_function_md.anon.d*.csv"))),
            'durations': sorted(glob(os.path.join(self.data_dir, "function_durations_percentiles.anon.d*.csv"))),
            'memory': sorted(glob(os.path.join(self.data_dir, "app_memory_percentiles.anon.d*.csv"))),
        }

        print(f"\n📁 Available files in {self.data_dir}:")
        for category, file_list in files.items():
            print(f"   {category}: {len(file_list)} files")
            for f in file_list[:3]:
                size_mb = os.path.getsize(f) / 1024 / 1024
                print(f"     - {os.path.basename(f)} ({size_mb:.1f} MB)")
            if len(file_list) > 3:
                print(f"     ... and {len(file_list) - 3} more")

        return files


class TraceReplayEnvironment:
    """Environment that replays real Azure traces instead of synthetic data."""

    def __init__(self, traces, n_servers=3):
        self.traces = traces
        self.n_servers = n_servers
        self.current_step = 0

    def get_request_load(self):
        """Get next request load from traces."""
        if self.current_step >= len(self.traces):
            self.current_step = 0
        load = self.traces[self.current_step]
        self.current_step += 1
        return load

    def reset(self):
        """Reset trace playback."""
        self.current_step = 0


def load_azure_traffic(data_dir="data/extracted", day=1):
    """
    Convenience function to load Azure traffic for a specific day.

    Returns:
        numpy array of normalized load values, or None if not available
    """
    loader = AzureDatasetLoader(data_dir)
    return loader.get_traffic_pattern(day=day)


if __name__ == "__main__":
    loader = AzureDatasetLoader()
    loader.list_available_files()
    loader.get_statistics(day=1)
    traffic = loader.get_traffic_pattern(day=1)
    if traffic is not None:
        print(f"Traffic pattern: {len(traffic)} steps")
        print(f"Sample loads: {traffic[:10]}")