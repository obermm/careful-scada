"""
Data quality checks for SCADA data preprocessing.
Includes uniqueness and completeness validation.
"""

import pandas as pd
import numpy as np


def apply_uniqueness(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Check for duplicate timestamps in the data and drop both duplicates.
    Args:
        df: DataFrame with MultiIndex (dataset_id, time_stamp)
        verbose: If True, print information about duplicates
    Returns:
        DataFrame with duplicates removed.
    """
    def check_duplicates(group: pd.DataFrame) -> pd.DataFrame:

        # mark all duplicates
        duplicates = group.index.get_level_values('time_stamp').duplicated(keep=False)

        if duplicates.any():
            if verbose:
                dataset_id = group.index.get_level_values('dataset_id')[0]
                dup_timestamps = group.index.get_level_values('time_stamp')[duplicates].unique()
                print(f"Dataset ID {dataset_id}: Dropped all {len(dup_timestamps)} duplicate \
                      timestamps: {dup_timestamps.tolist()}")

            # Drop both duplicates
            group = group[~duplicates]

        elif verbose:
            dataset_id = group.index.get_level_values('dataset_id')[0]
            print(f"Dataset ID {dataset_id}: No duplicate timestamps found.")

        return group

    df = df.groupby(level='dataset_id', group_keys=False).apply(check_duplicates)
    return df


def apply_completeness(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Check if all timestamps are present in the data time_stamp index level.
    If timestamps are missing, they are added with NaN values for all columns.

    Args:
        df: DataFrame with MultiIndex (dataset_id, time_stamp)
        verbose: If True, print information about missing timestamps
    Returns:
        DataFrame with missing timestamps added.
    """
    def missing_timestamps(group: pd.DataFrame) -> pd.DataFrame:
        # Work on a copy and make sure time_stamp is a proper DatetimeIndex
        df_group = group.reset_index()
        df_group['time_stamp'] = pd.to_datetime(df_group['time_stamp'])

        start = df_group['time_stamp'].min()
        end = df_group['time_stamp'].max()
        full_index = pd.date_range(start=start, end=end, freq='10min', tz=start.tz)

        # Reindex on the time axis and then rebuild the MultiIndex (dataset_id, time_stamp)
        dataset_id = df_group['dataset_id'].iat[0]
        df_group = df_group.set_index('time_stamp')
        df_group = df_group.reindex(full_index)

        # restore dataset_id column (fill with the group id)
        df_group['dataset_id'] = dataset_id

        # set MultiIndex with the same level names as the original
        df_group = df_group.reset_index().rename(columns={'index': 'time_stamp'})
        df_group = df_group.set_index(['dataset_id', 'time_stamp'])

        missing = full_index.difference(group.index.get_level_values('time_stamp'))
        if len(missing) > 0 and verbose:
            print(f"Dataset ID {dataset_id} is missing {len(missing)} timestamps.")

        return df_group

    df = df.groupby(level='dataset_id', group_keys=False).apply(missing_timestamps)
    return df
