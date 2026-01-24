"""
Missing value handling for SCADA data preprocessing.
"""

import pandas as pd
import numpy as np
from ..thresholds import MAX_CONSECUTIVE_ZEROS, RARELY_ZERO_SENSORS


def handle_missing_values(
    df: pd.DataFrame,
    farm_name: str,
    numeric_cols: list,
    strategy: str = "keep",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Handle missing values in the dataframe according to the specified strategy.
    Quote from the article:
    "The data for wind farm B and C were provided by the operator, with 0 values replacing
    all missing values, so large amounts of consecutive 0 values must be treated with caution."

    Therefore we set all large amouts of consecutive 0 values to NaN before applying the missing
    value strategy.

    Args:
        df: DataFrame with MultiIndex (dataset_id, time_stamp)
        farm_name: Name of the farm ('A', 'B', or 'C')
        numeric_cols: List of numeric column names
        strategy: Strategy for handling missing values
            - "keep": Keep NaN values as is, no filling or dropping.
            - "drop": Drop rows with any NaN values in numeric columns.
            - "linear_interpolation": Fill NaN values using linear interpolation.
            - "median": Fill NaN values with the median of each numeric column per 'dataset_id'.
        verbose: If True, print information
    Returns:
        DataFrame without missing values, filled or dropped according to strategy.
    """
    if farm_name in ['B', 'C']:

        # define sensors that can never be zero:
        # -> these are NaNs if zero
        never_zero_sensors = []
        for col in never_zero_sensors:
            if col in df.columns:
                df.loc[df[col] == 0, col] = np.nan

        # Some sensors can be zero rarely, but long sequences of zeros are suspicious:
        for col in RARELY_ZERO_SENSORS:

            if col not in df.columns:
                continue

            def _mark_long_zeros(group):
                is_zero = group[col] == 0
                # run id: increases when value changes (True/False runs)
                run_id = (is_zero != is_zero.shift()).cumsum()
                # counts per run
                run_counts = is_zero.groupby(run_id).transform('sum')
                long_mask = is_zero & (run_counts >= MAX_CONSECUTIVE_ZEROS)
                group.loc[long_mask, col] = np.nan
                return group

            # apply per-dataset_id
            df = df.groupby('dataset_id', group_keys=False).apply(_mark_long_zeros)


        # define sensors which can be zero often:
        #  -> use cross sensore validation to detect faulty zeros
        often_zero_sensors = ["rotor_speed_rpm_avg", "active_power_kWh", "wind_speed1_ms_avg",
                              "wind_speed2_ms_avg", "wind_speed3_ms_avg"]

        for col in often_zero_sensors:
            if col not in df.columns:
                continue
            if col == "rotor_speed_rpm_avg":
                def invalid_zero_condition(row):
                    return row["rotor_speed_rpm_avg"] == 0 and row["active_power_kWh"] > 100
                df.loc[df.apply(invalid_zero_condition, axis=1), col] = np.nan

            elif col == "active_power_kWh":
                def invalid_zero_condition(row):
                    return row["active_power_kWh"] == 0 and row["rotor_speed_rpm_avg"] > 2
                df.loc[df.apply(invalid_zero_condition, axis=1), col] = np.nan

            elif col.startswith("wind_speed") and col.endswith("_ms_avg"):
                def invalid_zero_condition(row):
                    return row[col] == 0 and row["active_power_kWh"] > 100
                df.loc[df.apply(invalid_zero_condition, axis=1), col] = np.nan



    # Print NaN counts per column
    if verbose:
        nans_per_group = df.groupby('dataset_id')[numeric_cols].apply(lambda x: x.isna().sum())
        print("\nMissing Values (NaN) per dataset_id and column:")
        # drop cols and rows with all zeros
        nans_per_group = nans_per_group.loc[:, (nans_per_group != 0).any(axis=0)]
        nans_per_group = nans_per_group.loc[(nans_per_group != 0).any(axis=1), :]
        if nans_per_group.empty:
            print("No missing values detected.")
        else:
            print(nans_per_group)

    if strategy == "keep":
        df = df

    elif strategy == "drop":
        df = df.dropna(subset=numeric_cols)

    elif strategy == "linear_interpolation":

        def interpolate_group(group):
            group[numeric_cols] = group[numeric_cols].interpolate(method='linear', 
                                                                        limit_direction='both')
            return group

        df = df.groupby('dataset_id', group_keys=False).apply(interpolate_group)

    elif strategy == "median":
        df = fill_missing_with_median(df, numeric_cols)

    else:
        raise NotImplementedError(f"Missing value strategy '{strategy}' \
                                  is not implemented.")

    remaining_nans = df[numeric_cols].isna().sum().sum()
    if remaining_nans > 0:
        if verbose:
            print(f"Warning: {remaining_nans} NaN values remain after handling missing values")
        if strategy != "keep":
            raise ValueError(f"Missing value strategy '{strategy}' failed\
                              to remove all NaNs")
    return df


def fill_missing_with_median(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Fill missing values with the median of each numeric column per 'dataset_id'.
    
    Args:
        df: DataFrame with MultiIndex (dataset_id, time_stamp)
        numeric_cols: List of numeric column names
    Returns:
        DataFrame with missing values filled.
    """
    def median_group(group):
        for col in numeric_cols:
            group[col] = group[col].fillna(group[col].median())
        return group

    df = df.groupby("dataset_id", group_keys=True).apply(median_group)
    df = df.droplevel(0)

    return df
