"""Defines the base loader parent class."""

import json
import os
from pathlib import Path
import pandas as pd
from abc import abstractmethod

# Parent DataLoader class ##########################################################################
class DataLoader():
    """
    Base class for all farm specific data loaders.
    Defines common methods for loading, unit unification, and column renaming.
    Child classes must implement the `unify_units` method as unit conventions differ per farm.
    """


    def __init__(self, rename_map_path="rename_map.json"):

        # Get the absolute path to the project root
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(base_dir, "..")
    
        # Construct absolute path to datasets
        self.path = os.path.join(project_root, f"dat/raw/Wind Farm {self.farm_name}/datasets")

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
        df_train, df_test = self.train_test_split(df)
        return df_train, df_test


    def load_raw(self) -> pd.DataFrame:
        """
        Load raw CSV files into a single DataFrame with MultiIndex (dataset_id, time_stamp).
        Reads only columns present in the rename map for the specific farm.
        Returns:
            pd.DataFrame: Combined DataFrame of all datasets with raw column names.
        """

        # Only load raw column names that appear in the rename map for this farm
        allowed_raw = set()
        for std_name, farms in self.rename_map.items():
            if self.farm_name in farms:
                allowed_raw.update(farms[self.farm_name])

        if not allowed_raw:
            raise ValueError(f"No column mappings found for farm '{self.farm_name}' in rename_map.")
        else:
            print(f"Loading raw data for farm '{self.farm_name}' with {len(allowed_raw)} allowed columns.")

        csv_files = list(Path(self.path).glob('*.csv'))
        df_list = []
        for file in csv_files:

            try:
                # only columns present and in allowed_raw are read
                turbine_df = pd.read_csv(file,
                                        sep=';',
                                        usecols=lambda c: c in allowed_raw, # use a callable so missing columns don't raise error
                                        parse_dates=['time_stamp'])
            except Exception as e:
                print(f"Warning: Could not read {file} with usecols because {e}. Falling back to full read and filtering.")
                # fallback: read full file and filter columns (safe but uses more memory for the single file)
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
    
    @abstractmethod
    def unify_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Each Dataset (A, B, C) comes in potentially different units for some sensors"""
        pass


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


    def train_test_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Splits the DataFrame into training and testing based on the train_test column provided 
        by the authors of CARE dataset.
        """
        df = df.copy()
        if len(df) == 0:
            raise ValueError("Empty DataFrame passed to train_test_split.")
        if 'train_test' not in df.columns:
            raise ValueError("train_test column not found in DataFrame.")
        
        df_train = df[df['train_test'] == 'train']
        df_test = df[df['train_test'] == 'prediction']

        print(f"Train - Test ratio: {round(len(df_train)/len(df), 2)}.")
        return df_train, df_test


# Farm specific DataLoader implementations #########################################################
class FarmALoader(DataLoader):
    """
    Data loader for Wind Farm A.
    Implements unit unification specific to Farm A.
    """

    def __init__(self):
        self.farm_name = 'A'
        super().__init__() 

    def unify_units(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # active power comes in Wh -> convert to kWh
        df["sensor_50"] = df["sensor_50"].div(1000)
        return df


class FarmBLoader(DataLoader):
    """
    Data loader for Wind Farm B.
    Implements unit unification specific to Farm B.
    """

    def __init__(self):
        self.farm_name = 'B'
        super().__init__() 

    def unify_units(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        return df


class FarmCLoader(DataLoader):
    """
    Data loader for Wind Farm C.
    Implements unit unification specific to Farm C.
    """

    def __init__(self):
        self.farm_name = 'C'
        super().__init__() 

    def unify_units(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        return df
