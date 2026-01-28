"""Defines the DataLoader class for loading CARE SCADA datasets."""

import json
import os
from pathlib import Path
import pandas as pd

# DataLoader class #################################################################################
class DataLoader():
    """
    Data loader for CARE SCADA datasets.
    Loads data from a specified farm, unifies units, and standardizes column names.
    """

    FARM_UNIT_CONVERSIONS = {
        'A': {'sensor_50': lambda x: x / 1000},  # Wh -> kWh conversion for Farm A
        'B': {},  # No unit conversions needed for Farm B
        'C': {}   # No unit conversions needed for Farm C
    }

    def __init__(self, farm_name, data_path="dat/raw", rename_map_path="rename_map.json"):
        """
        Initialize the DataLoader for a specific farm.
        Args:
            farm_name (str): Name of the farm ('A', 'B', or 'C').
            data_path (str): Path to folder containing CARE SCADA datasets.
            rename_map_path (str): Relative path to JSON file with column mappings.
        """
        if farm_name not in ('A', 'B', 'C'):
            raise ValueError(f"Invalid farm_name '{farm_name}'. Must be 'A', 'B', or 'C'.")

        self.farm_name = farm_name

        # Get the absolute path to the project root
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(base_dir, "..")

        # Construct absolute path to datasets
        self.path = os.path.join(project_root, f"{data_path}/Wind Farm {self.farm_name}/datasets")

        # Load rename map from JSON (mapping raw to standard column names)
        schema_path = os.path.join(base_dir, rename_map_path)
        with open(schema_path, "r") as f:
            self.rename_map = json.load(f)

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
        Reads only columns present in the rename map for the specific farm.
        Returns:
            pd.DataFrame: Combined DataFrame of all datasets with raw column names.
        """

        # Only load raw column names that appear in the rename map for this farm
        allowed_raw = set()
        for _, farms in self.rename_map.items():
            if self.farm_name in farms:
                allowed_raw.update(farms[self.farm_name])

        if not allowed_raw:
            raise ValueError(f"No column mappings found for farm '{self.farm_name}' in rename_map.")
        print(f"Loading raw data for farm '{self.farm_name}'\
                with {len(allowed_raw)} allowed columns.")

        csv_files = list(Path(self.path).glob('*.csv'))
        df_list = []
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

        # Combine all datasets into a single DataFrame and set MultiIndex
        df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
        df = df.set_index(['dataset_id', 'time_stamp']).sort_index()

        print(f"Loaded {len(df_list)} datasets. Columns loaded: {sorted(df.columns.tolist())}")
        return df

    def unify_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply farm-specific unit conversions based on the farm_name."""
        df = df.copy()
        conversions = self.FARM_UNIT_CONVERSIONS.get(self.farm_name, {})
        for column, conversion_func in conversions.items():
            if column in df.columns:
                df[column] = df[column].apply(conversion_func)
        return df



    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to unified naming scheme using the rename_map."""
        df = df.copy()
        rename_dict = {}
        for std_name, farms in self.rename_map.items():
            if self.farm_name in farms:
                for raw_name in farms[self.farm_name]:
                    if raw_name in df.columns:
                        rename_dict[raw_name] = std_name

        # Apply the renaming
        df.rename(columns=rename_dict, inplace=True)
        print(f"Renamed {len(rename_dict)} columns.")
        return df
