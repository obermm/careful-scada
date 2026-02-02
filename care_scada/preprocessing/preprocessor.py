"""
Main preprocessor class for SCADA data preprocessing pipeline.
Coordinates data cleaning, feature engineering, and outlier handling.
"""

import pandas as pd
import numpy as np

from ..sensor_config import get_sensor_config
from .data_quality import apply_uniqueness, apply_completeness
from .missing_values import handle_missing_values
from .outlier_detection import flag_outliers
from .outlier_handling import handle_outliers


class Preprocessor:
    """
    SCADA preprocessing pipeline:
        - cleaning
        - feature engineering

    Args:
        farm_name: Wind farm name ("A", "B", "C") to apply farm specific preprocessing rules.
        dataset_id: If specified, only process this dataset_id (faster than processing all!)
        remove_status: List of status_type_id values to remove from the data.
                      If None, uses default from sensor_config.yaml.
        missing_values: Strategy for handling missing values 
                        ("keep", "drop", "linear_interpolation", "median").
                        If None, uses default from sensor_config.yaml.
        outlier_handling: Strategy for handling outliers ("flag_only", "set_to_nan", "drop").
                         If None, uses default from sensor_config.yaml.
        numeric_cols: List of numeric columns to consider. If None, auto-detects numeric columns.
        verbose: If True, print information and statistics during preprocessing.
    """

    def __init__(
        self,
        farm_name='A',
        dataset_id=None,
        remove_status = None,
        missing_values = None,
        outlier_handling = None,
        numeric_cols = None,
        verbose=True
    ):
        # Load settings from config
        config = get_sensor_config()
        settings = config.get_preprocessing_settings()
        
        self.farm_name = farm_name
        self.dataset_id = str(dataset_id) if dataset_id is not None else None
        self.numeric_cols = numeric_cols
        
        # Apply settings if not specified
        if remove_status is None:
            remove_status = settings['remove_status']
        if isinstance(remove_status, list):
            self.remove_status = remove_status
        elif isinstance(remove_status, int):
            self.remove_status = [remove_status]
        else:
            self.remove_status = None

        self.missing_values = missing_values if missing_values is not None else settings['missing_values_strategy']
        self.outlier_handling = outlier_handling if outlier_handling is not None else settings['outlier_handling']
        self.verbose = verbose

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def process(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full preprocessing pipeline:
        Args:
            df: Raw SCADA DataFrame with MultiIndex (dataset_id, time_stamp)
        Returns:
            df_train: Cleaned and feature-engineered training DataFrame
            df_test: Cleaned and feature-engineered testing DataFrame
        """
        df = df.copy()

        # Validate input structure
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("DataFrame must have MultiIndex (dataset_id, time_stamp)")
        if df.index.names != ['dataset_id', 'time_stamp']:
            raise ValueError(f"Index levels must be ['dataset_id', 'time_stamp'], \
                             got {df.index.names}")

        # for testing only keep one dataset_id
        if self.dataset_id is not None:
            if self.dataset_id not in df.index.get_level_values('dataset_id'):
                raise KeyError(f"dataset_id {self.dataset_id} not found in DataFrame index.")

            df = df.xs(self.dataset_id, level='dataset_id', drop_level=False)

        df = self.clean(df)
        df = self.add_features(df)
        df_train, df_test = self._train_test_split(df)

        return df_train, df_test


    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Data cleaning pipeline:
        - store numeric columns
        - filter on status
        - enforce uniqueness
        - enforce completeness
        - handle missing values (according to the specified strategy)
        - flag outliers (absolute values and gradients if specified)
        - handle outliers (according to the specified strategy)
        Args:
            df: Raw SCADA DataFrame with MultiIndex (dataset_id, time_stamp)
        Returns:
            df: Cleaned DataFrame
        """
        df = df.copy()

        # store numeric columns
        df = self._set_numeric_cols(df)

        # filter on status
        df = self._remove_status(df)

        # Apply Data Quality checks
        df = apply_uniqueness(df, verbose=self.verbose)
        df = apply_completeness(df, verbose=self.verbose)

        # handle missing values
        df = handle_missing_values(
            df,
            farm_name=self.farm_name,
            numeric_cols=self.numeric_cols,
            strategy=self.missing_values,
            verbose=self.verbose
        )
        df = flag_outliers(df, verbose=self.verbose)

        # remove outliers except for sensors that measure the same underlying latent variable
        # e.g. different wind measurements
        df = handle_outliers(df, strategy=self.outlier_handling, verbose=self.verbose)

        return df


    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering pipeline:
        - sort index
        Args:
            df: Cleaned SCADA DataFrame with MultiIndex (dataset_id, time_stamp)
        Returns:
            df: DataFrame with added features
        """
        df = df.copy()
        df = df.sort_index()

        #TODO add Features and lags here

        # TODO: Implement feature engineering
        # - Combine sensor readings using State Space Model ?
        # - Add lag features
        # - Min-max scaling

        return df


    # =====================================================================
    # INTERNAL HELPERS
    # =====================================================================

    def _set_numeric_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select numeric columns if not already set and store them on the object
        """
        if self.numeric_cols is None:
            self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # remove asset id colum if present
            self.numeric_cols = [col for col in self.numeric_cols if col not in ['asset_id', 'id']]
        return df

    def _remove_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows based on the 'status' column.
        Copied from dat/raw/README.md:
        "Status of the wind turbine at the given time stamp:
        There are 6 different status IDs where status 0 and 2 are considered normal operation:
        * 0: Normal Operation - The turbine is in normal power production mode
        * 1: Derated Operation - Derated power generation with a power restriction
        * 2: Idling - Asset is idling and waits to operate again
        * 3: Service - Asset is in service mode / service team is at the site
        * 4: Downtime - Asset is down due a fault or other reasons
        * 5: Other - Other operational states"

        Returns:
            DataFrame with specified statuses removed.
        """
        if not self.remove_status:
            return df

        # Before collect filtering statistics, len per dataset_id
        before = df.groupby(level='dataset_id').size()

        # Apply filtering
        if "status_type_id" in df.columns:
            df = df[~df["status_type_id"].isin(self.remove_status)]

        # after filtering, len per dataset_id
        after  = df.groupby(level='dataset_id').size()

        if self.verbose:
            print(f"\nStatus Filtering: removed statuses {self.remove_status}")
            # Statistics: Combine before and after to get filtered counts
            filter_stats = []
            for ds_id in before.index:
                before_count = before.get(ds_id, 0)
                after_count = after.get(ds_id, 0)
                filtered_count = before_count - after_count
                filtered_percent = filtered_count / before_count * 100 if before_count > 0 else 0
                filter_stats.append({
                    "dataset_id": ds_id,
                    "before_count": before_count,
                    "after_count": after_count,
                    "filtered_count": filtered_count,
                    "filtered_percent": round(filtered_percent, 2)
                })
            filter_stats_df = pd.DataFrame(filter_stats).set_index("dataset_id")

            # Printing: only show places where something was filtered
            if filter_stats_df.empty:
                print("Filtering Stats: no values were filtered.")
            else:
                filter_stats_df = filter_stats_df[filter_stats_df["filtered_count"] > 0].sort_values(
                    ["filtered_count"], ascending=False
                )
                print("Filtering Stats (only nonzero):")
                print(filter_stats_df)

        return df

    def _train_test_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        if self.verbose:
            print(f"Train - Test ratio: {round(len(df_train)/len(df), 2)}.")
        return df_train, df_test
