"""Defines the DataLoader class for loading CARE SCADA datasets."""

import os
from pathlib import Path
import pandas as pd
from .sensor_config import SensorConfig

# DataLoader class #################################################################################
class DataLoader():
    """
    Data loader for CARE SCADA datasets.
    Loads data from a specified farm, unifies units, and standardizes column names.
    """

    def __init__(self, farm_name, data_path="dat/Care_To_Compare", config_path=None):
        """
        Initialize the DataLoader for a specific farm.
        Args:
            farm_name (str): Name of the farm ('A', 'B', or 'C').
            data_path (str): Path to folder containing CARE SCADA datasets.
            config_path (str): Path to sensor_config.yaml. If None, uses default location.
        """
        if farm_name not in ('A', 'B', 'C'):
            raise ValueError(f"Invalid farm_name '{farm_name}'. Must be 'A', 'B', or 'C'.")

        self.farm_name = farm_name

        # Get the absolute path to the project root
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(base_dir, "..")

        # Construct absolute path to datasets
        self.path = os.path.join(project_root, f"{data_path}/Wind Farm {self.farm_name}/datasets")

        # Load sensor configuration
        self.sensor_config = SensorConfig(config_path)

    def load(self) -> pd.DataFrame:
        """Full loading pipeline:
        - load raw data
        - unify units
        - standardize column names
        Returns:
            pd.DataFrame: Processed DataFrame with unified units and standardized column names.
        """
        df = self.load_raw()
        df = self.unify_units(df)
        df = self.standardize_column_names(df)
        return df


    def load_raw(self) -> pd.DataFrame:
        """
        Load raw CSV files into a single DataFrame with MultiIndex (dataset_id, time_stamp).
        Reads only columns present in the sensor config for the specific farm.
        Returns:
            pd.DataFrame: Combined DataFrame of all datasets with raw column names.
        """

        # Get all allowed raw column names from sensor config
        allowed_raw = self.sensor_config.get_all_allowed_columns(self.farm_name)

        if not allowed_raw:
            raise ValueError(f"No column mappings found for farm '{self.farm_name}' in sensor config.")
        print(f"Loading raw data for farm '{self.farm_name}' with {len(allowed_raw)} allowed columns.")

        csv_files = list(Path(self.path).glob('*.csv'))
        df_list = []
        
        # Check the first file to see which columns actually exist
        # Helpful for debugging missing columns in allowed_raw
        if csv_files:
            first_file = csv_files[0]
            sample_df = pd.read_csv(first_file, sep=';', nrows=0)
            actual_cols = set(sample_df.columns)
            missing_in_csv = allowed_raw - actual_cols
            if missing_in_csv:
                print(f"Debug: {len(missing_in_csv)} expected columns missing in CSV: {sorted(missing_in_csv)}")

        for file in csv_files:
            try:
                # only columns present and in allowed_raw are read
                turbine_df = pd.read_csv(file,
                                        sep=';',
                                        # use a callable so missing columns don't raise error
                                        usecols=lambda c: c in allowed_raw,
                                        parse_dates=['time_stamp'])
            except Exception as e:
                print(f"Warning: Could not read {file} with usecols because {e}. \
                      Falling back to full read and filtering.")
                # fallback: read full file and filter columns
                # (safe but uses more memory for the single file)
                turbine_df = pd.read_csv(file, sep=';')
                cols = [c for c in turbine_df.columns if c in allowed_raw]
                turbine_df = turbine_df[cols]
                if 'time_stamp' in turbine_df.columns:
                    turbine_df['time_stamp'] = pd.to_datetime(turbine_df['time_stamp'])

            turbine_df["dataset_id"] = file.stem
            df_list.append(turbine_df)

        if not df_list:
            raise RuntimeError(f"Warning: No CSV files found in path: {self.path}")
        # Combine all datasets into a single DataFrame and set MultiIndex
        try:
            df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
            df = df.set_index(['dataset_id', 'time_stamp']).sort_index()
        except Exception as e:
            raise RuntimeError(f"Error combining datasets: {e}")

        print(f"Loaded {len(df_list)} datasets. Columns loaded: {sorted(df.columns.tolist())}")
        return df

    def unify_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply farm-specific unit conversions using sensor config."""
        df = df.copy()

        # Apply conversions to raw columns (before renaming)
        rename_map = self.sensor_config.get_rename_map(self.farm_name)
        reverse_map = {v: k for k, v in rename_map.items()}
        conversions = self.sensor_config.get_unit_conversions(self.farm_name)

        for std_column, conversion_func in conversions.items():
            # Find the raw column name for this standardized column
            raw_column = reverse_map.get(std_column)
            if raw_column and raw_column in df.columns:
                df[raw_column] = df[raw_column].apply(conversion_func)

        return df

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to unified naming scheme using sensor config."""
        df = df.copy()
        rename_dict = self.sensor_config.get_rename_map(self.farm_name)

        # Only keep mappings for columns that actually exist
        rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}

        # Apply the renaming
        df.rename(columns=rename_dict, inplace=True)

        # Check for columns that weren't renamed
        original_cols = set(df.columns) - set(rename_dict.values())
        if original_cols:
            print(f"Warning: {len(original_cols)} columns were not renamed: {sorted(original_cols)}")

        print(f"Renamed {len(rename_dict)} columns.")
        return df
