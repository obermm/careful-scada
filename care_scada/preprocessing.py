"""
This script implements a preprocessing pipeline for SCADA data.
It includes data cleaning and outlier flagging.
"""

import pandas as pd
import numpy as np

from .thresholds import (
    ABSOLUTE_THRESHOLDS,
    GRADIENT_THRESHOLDS,
    DRIFT_THRESHOLDS,
    FLATLINE_THRESHOLDS,
    SHORT_INTERIOR_THRESHOLD,
    MAX_CONSECUTIVE_ZEROS,
    RARLEY_ZERO_SENSORS,
    GRADIENT_TOL,
    COLS_TO_IGNORE_ON_DROPPING_OUTLIER_ROWS,
)


class Preprocessor:
    """
    SCADA preprocessing pipeline:
        - cleaning
        - feature engineering

    Args:
        wind_farm: Wind farm identifier (A, B, C) to apply farm specific preprocessing rules.
        dataset_id: If specified, only process this dataset_id (faster than processing all!)
        remove_status: List of status_type_id values to remove from the data.
        missing_values: Strategy for handling missing values ("keep", "drop", "linear_interpolation", "median").
        outlier_handling: Strategy for handling outliers ("flag_only", "set_to_nan", "drop").
        numeric_cols: List of numeric columns to consider. If None, auto-detects numeric columns.
        verbose: If True, print information and statistics during preprocessing.
    """

    def __init__(
        self,
        wind_farm='A',
        dataset_id=None,
        remove_status = [],
        missing_values = "keep",
        outlier_handling = "flag_only",
        numeric_cols = None,
        verbose=True
    ):
        self.wind_farm = wind_farm
        self.dataset_id = str(dataset_id) if dataset_id is not None else None
        self.numeric_cols = numeric_cols
        if type(remove_status) == list:
            self.remove_status = remove_status
        elif type(remove_status) == int:
            self.remove_status = [remove_status]
        self.missing_values = missing_values
        self.outlier_handling = outlier_handling
        self.verbose = verbose

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline:
        Args:
            df: Raw SCADA DataFrame with MultiIndex (dataset_id, time_stamp)
        Returns:
            df: Cleaned and feature-engineered DataFrame
        """
        df = df.copy()

        # Validate input structure
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("DataFrame must have MultiIndex (dataset_id, time_stamp)")
        if df.index.names != ['dataset_id', 'time_stamp']:
            raise ValueError(f"Index levels must be ['dataset_id', 'time_stamp'], got {df.index.names}")
        
        # for testing only keep one dataset_id
        if self.dataset_id is not None:
            df = df.xs(self.dataset_id, level='dataset_id', drop_level=False)

        df = self.clean(df)
        df = self.add_features(df)

        return df


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

        # store nummeric columns
        df = self._set_numeric_cols(df)

        # filter on status
        df = self._remove_status(df)

        # Apply Data Quality checks
        df = self._apply_uniqueness(df)
        df = self._apply_completeness(df)

        # handle missing values
        df = self._handle_missing_values(df)
        df = self._flag_outliers(df) 

        # remove outliers except for sensors that measure the same underlying latent variable
        # e.g. different wind measurements
        df = self._handle_outliers(df)

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
    # INTERNAL LOW-LEVEL HELPERS
    # =====================================================================
    def _apply_uniqueness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for duplicate timestamps in the data and drop both duplicates.
        Special Case: the 28th of February is sometimes duplicated during non-leap years.
        In this case, we keep the 28th (first occurrence) and drop the 29th (second occurrence),
        assuming that during the anonymization process the year was changed to from a leap year
        to non-leap year and thus the 29th became a duplicate of the 28th.

        Returns:
            DataFrame with duplicates removed.
        """
        def check_duplicates(group):
             
            # mark all duplicates
            duplicates = group.index.get_level_values('time_stamp').duplicated(keep=False)
            all_duplicates_on_28th = False

            # Special Case: the 28th of February is sometimes duplicated due to leap years.
            # In this case, we keep the 28th (first occurrence) and drop the 29th (second occurrence).
            # (Instead of dropping both duplicates)
            if group.index.get_level_values('time_stamp').duplicated(keep='first').any():
                duplicates_times = group.index.get_level_values('time_stamp')[group.index.get_level_values('time_stamp').duplicated(keep='first')]

                # Check if the duplicated timestamps include the full day of Feb 28th
                feb_28 = pd.to_datetime(f"{duplicates_times.year[0]}-02-28" + " 00:00:00") + pd.to_timedelta(np.arange(0, 24*6)*10, unit='min')

                # check if all duplicates are on feb 28th
                if len(duplicates_times) == len(feb_28) and set(duplicates_times) == set(feb_28):
                    all_duplicates_on_28th = True

            if duplicates.any():

                if all_duplicates_on_28th:
                    # Drop only the second Feb 28 duplicates
                    duplicates = group.index.get_level_values('time_stamp').duplicated(keep='first')

                if all_duplicates_on_28th and self.verbose:
                    dataset_id = group.index.get_level_values('dataset_id')[0]
                    dup_timestamps = group.index.get_level_values('time_stamp')[duplicates].unique()
                    print(f"Dataset ID {dataset_id}: Kept first occurance of duplicate timestamps on Feb 28th during non-leap year.")

                elif self.verbose:
                    dataset_id = group.index.get_level_values('dataset_id')[0]
                    dup_timestamps = group.index.get_level_values('time_stamp')[duplicates].unique()
                    print(f"Dataset ID {dataset_id}: Dropped all {len(dup_timestamps)} duplicate timestamps: {dup_timestamps.tolist()}")

                # Drop both duplicates
                group = group[~duplicates]
            
            elif self.verbose:
                dataset_id = group.index.get_level_values('dataset_id')[0]
                print(f"Dataset ID {dataset_id}: No duplicate timestamps found.")
            
            return group

        df = df.groupby(level='dataset_id', group_keys=False).apply(check_duplicates)
        return df


    def _apply_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check if all timestamps are present in the data time_stamp index level.
        If timestamps are missing, they are added with NaN values for all columns.

        Returns:
            DataFrame with missing timestamps added.
        """
        def missing_timestamps(group):
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
            if len(missing) > 0 and self.verbose:
                print(f"Dataset ID {dataset_id} is missing {len(missing)} timestamps.")

            return df_group

        df = df.groupby(level='dataset_id', group_keys=False).apply(missing_timestamps)
        return df
    

    def _set_numeric_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select numeric columns if not already set and store them on the object
        """
        if self.numeric_cols is None:
            self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # remove asset id colum if present
            self.numeric_cols = [col for col in self.numeric_cols if col not in ['asset_id', 'id']]
        return df



    # ----- Status filtering -----------------------------------------------
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
        
        # Apply filterinng
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



    # ----- Missing values -------------------------------------------------
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataframe according to the specified strategy.
        Quote from the article:
            " The data for wind farm B and C were provided by the operator, with 0 values replacing all 
            missing values, so large amounts of consecutive 0 values must be treated with caution."

        Therefore we set all large amouts of consecutive 0 values to NaN before applying the missing
        value strategy. 

        Strategies:
        - "keep": Keep NaN values as is, no filling or dropping.
        - "drop": Drop rows with any NaN values in numeric columns.
        - "linear_interpolation": Fill NaN values using linear interpolation.
        - "median": Fill NaN values with the median of each numeric column per 'dataset_id'.
        Returns: 
            DataFrame without missing values, filled or dropped according to strategy.
        """
        if self.wind_farm in ['B', 'C']:

            # define sensors that can never be zero:
            # -> these are NaNs if zero
            never_zero_sensors = []
            for col in never_zero_sensors:
                if col in df.columns:
                    df.loc[df[col] == 0, col] = np.nan

            # Some sensors can be zero rarely, but long sequences of zeros are suspicious:
            for col in RARLEY_ZERO_SENSORS:

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
        if self.verbose:
            nan_counts_per_group = df.groupby('dataset_id')[self.numeric_cols].apply(lambda x: x.isna().sum())
            print("\nMissing Values (NaN) per dataset_id and column:")
            # drop cols and rows with all zeros
            nan_counts_per_group = nan_counts_per_group.loc[:, (nan_counts_per_group != 0).any(axis=0)]
            nan_counts_per_group = nan_counts_per_group.loc[(nan_counts_per_group != 0).any(axis=1), :]
            if nan_counts_per_group.empty:
                print("No missing values detected.")
            else:
                print(nan_counts_per_group)

        if self.missing_values == "keep":
            df = df

        elif self.missing_values == "drop":
            df = df.dropna(subset=self.numeric_cols)

        elif self.missing_values == "linear_interpolation":

            def interpolate_group(group):
                group[self.numeric_cols] = group[self.numeric_cols].interpolate(method='linear', limit_direction='both')
                return group

            df = df.groupby('dataset_id', group_keys=False).apply(interpolate_group)

        elif self.missing_values == "median":
            df = self._fill_missing_with_median(df)
        
        else:
            raise NotImplementedError(f"Missing value strategy '{self.missing_values}' is not implemented.")

        remaining_nans = df[self.numeric_cols].isna().sum().sum()
        if remaining_nans > 0:
            if self.verbose:
                print(f"Warning: {remaining_nans} NaN values remain after handling missing values")
            if self.missing_values != "keep":
                raise ValueError(f"Missing value strategy '{self.missing_values}' failed to remove all NaNs")
        return df


    def _fill_missing_with_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values with the median of each numeric column per 'dataset_id'.
        Returns:
            DataFrame with missing values filled.
        """
        def median_group(group):
            for col in self.numeric_cols:
                group[col] = group[col].fillna(group[col].median())
            return group

        df = df.groupby("dataset_id", group_keys=True).apply(median_group)
        df = df.droplevel(0)

        return df



    # ----- Outliers -------------------------------------------------------

    def _flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag outliers in the dataframe.
        Steps:
        - Apply threshold based outlier detection (eg. no negative wind speeds)
        - Apply gradient based outlier detection if specified (eg. no sudden jumps in values)
        - Find drifts and flatlines (eg. no long periods of constant values)
        Returns:
            DataFrame with outlier flags added.
            is_outlier_gradient: True if gradient based outlier detection flagged the value, 
                                combining both gradient exceedings and drift/flatline detection
            is_outlier_threshold: True if threshold based outlier detection flagged the value
            is_outlier: True if any outlier flag is set
        """
        # apply thresholds
        df = self._flag_outliers_thresholds(df)

        # apply gradient based method
        df = df.groupby("dataset_id", group_keys=False).apply(self._flag_outliers_gradient_thresholds)

        # Find drifts and flatlines
        df = df.groupby("dataset_id", group_keys=False).apply(self._flag_outliers_drifts_flatlines)

        # Combine the results
        df["is_outlier_gradient"] = df["is_outlier_gr_ex"] | df["is_outlier_drift_flat"]
        df["is_outlier"] = df["is_outlier_gradient"] | df["is_outlier_threshold"]
        df["is_outlier_cols"] = df.apply(
            lambda row: list(set(
                (row["is_outlier_gr_ex_cols"] if isinstance(row["is_outlier_gr_ex_cols"], list) else []) +
                (row["is_outlier_drift_flat_cols"] if isinstance(row["is_outlier_drift_flat_cols"], list) else []) +
                (row["is_outlier_threshold_cols"] if isinstance(row["is_outlier_threshold_cols"], list) else [])
            )), axis=1
        )

        # Drop the outlier flag columns
        cols_to_drop = ["threshold_too_low", "threshold_too_high", "rotor_moving"]
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)

        return df


    def _flag_outliers_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply predefined thresholds to flag extreme values.
        Thresholds are defined per column as (min, max) tuples.

        Returns:
            DataFrame with outlier flags added:
            - is_outlier_threshold (bool)
            - threshold_too_low (list of column names that were too low)
            - threshold_too_high (list of column names that were too high)
            - is_outlier_threshold_cols (list of all column names that exceeded thresholds)
        """
        df["is_outlier_threshold"] = False
        df["threshold_too_high"] = None
        df["threshold_too_low"] = None

        # initialize dicts for per-row lists once
        too_low_map = {idx: [] for idx in df.index}
        too_high_map = {idx: [] for idx in df.index}

        threshold_stats = []

        for col, (lower, upper) in ABSOLUTE_THRESHOLDS.items():
            if col not in df.columns:
                continue
    
            # boolean masks
            mask_low = df[col] < lower
            mask_high = df[col] > upper

            # mark exceeded flags
            df.loc[mask_low | mask_high, "is_outlier_threshold"] = True

            # collect column names per-row into dicts
            for idx in df.index[mask_low]:
                too_low_map[idx].append(col)
            for idx in df.index[mask_high]:
                too_high_map[idx].append(col)

            # stats
            too_low = int(mask_low.sum())
            too_high = int(mask_high.sum())
            threshold_stats.append({
                "column": col,
                "min_threshold": lower,
                "max_threshold": upper,
                "dropped_too_low": too_low,
                "dropped_too_high": too_high
            })

        # store index-aligned list in columns
        df["threshold_too_low"] = [too_low_map[idx] for idx in df.index]
        df["threshold_too_high"] = [too_high_map[idx] for idx in df.index]
        df["is_outlier_threshold_cols"] = df.apply(lambda row: list(set(row["threshold_too_low"] + row["threshold_too_high"])), axis=1)

        if self.verbose:
            print("\nThreshold Exceeding Stats:")
            df_stats = df[df["is_outlier_threshold"]]
            if df_stats.empty:
                print("No values exceeded thresholds.")
            else:
                # too_low
                too_low = df_stats["threshold_too_low"].explode().dropna()
                dataset_ids_low = too_low.index.get_level_values("dataset_id")
                too_low_df = pd.DataFrame({
                    "dataset_id": dataset_ids_low,
                    "column": too_low.values
                })
                too_low_df["min_threshold"] = too_low_df["column"].map(lambda col: ABSOLUTE_THRESHOLDS.get(col, (None, None))[0])
                too_low_df["max_threshold"] = too_low_df["column"].map(lambda col: ABSOLUTE_THRESHOLDS.get(col, (None, None))[1])
                too_low_counts = too_low_df.groupby(["dataset_id", "column", "min_threshold", "max_threshold"]).size().reset_index(name="too_low_count")

                # too_high
                too_high = df_stats["threshold_too_high"].explode().dropna()
                dataset_ids_high = too_high.index.get_level_values("dataset_id")
                too_high_df = pd.DataFrame({
                    "dataset_id": dataset_ids_high,
                    "column": too_high.values
                })
                too_high_df["min_threshold"] = too_high_df["column"].map(lambda col: ABSOLUTE_THRESHOLDS.get(col, (None, None))[0])
                too_high_df["max_threshold"] = too_high_df["column"].map(lambda col: ABSOLUTE_THRESHOLDS.get(col, (None, None))[1])
                too_high_counts = too_high_df.groupby(["dataset_id", "column", "min_threshold", "max_threshold"]).size().reset_index(name="too_high_count")

                result = too_low_counts.merge(
                    too_high_counts,
                    on=["dataset_id", "column", "min_threshold", "max_threshold"],
                    how="outer"
                ).fillna(0)

                result = result.astype({"too_low_count": int, "too_high_count": int})
                result["allowed_range"] = "[" + result["min_threshold"].astype(str) + ", " + result["max_threshold"].astype(str) + "]"
                display_result = result[["dataset_id", "column", "allowed_range", "too_low_count", "too_high_count"]].sort_values(["dataset_id", "column"])
                print(display_result.set_index("dataset_id").to_string())

        return df


    def _flag_outliers_gradient_thresholds(self, group):
        """
        Flag outliers based on sudden changes (gradients) in the data that exceed manually defined thresholds.
        This method computes the absolute gradient per column and flags values where the
        gradient exceeds a predefined threshold.
        Stores:
        - is_outlier_gr_ex (bool)
        - is_outlier_gr_ex_cols (list of column names that exceeded)
        - outlier_gradient_type (str): type of gradient exceeding detected
        """
        df = group.copy()

        # Initialize column and mapping for exceeded gradients and types
        df["is_outlier_gr_ex"] = False
        exceeded_cols_map = {idx: [] for idx in df.index}
        exceeded_type_map = {idx: [] for idx in df.index}

        # Compute gradients for each numeric column
        for col in self.numeric_cols:
            if col not in df.columns or col in ['dataset_id', 'turbine_id', 'timestamp', 'status_type_id']:
                continue
                
            # Compute gradient for the current column relative to the previous timestamp
            # Only consider gradients where the previous timestamp is not longer ago than 30minutes earlier.
            # If the previous timestamp is further in the past (e.g. due to missing rows),
            # mark as invalid and do not consider for gradient-based outlier detection.
            # NaNs are replaced via forward-fill for gradient computation.
            if 'time_stamp' in df.index.names:
                times = pd.to_datetime(df.index.get_level_values('time_stamp'))
            else:
                times = pd.to_datetime(df.index)
            times_series = pd.Series(times, index=df.index)
            prev_value_filled = df[col].ffill().shift(1)  # last valid observation
            last_valid_time = times_series.where(df[col].notna()).ffill().shift(1)
            gradient = df[col] - prev_value_filled

            # Zeitdifferenz zur letzten gültigen Beobachtung
            time_diff = times_series - last_valid_time

            # Get threshold for the current column
            threshold = GRADIENT_THRESHOLDS.get(col, None)
            if threshold is None:
                continue

            # invalid where time gap != 30min or either value is NaN
            invalid = (time_diff >= pd.Timedelta(minutes=30)) #| df[col].isna() | prev.isna()

            # Create mask for absolute gradient exceedings, but only where valid
            mask_exceeded = (gradient.abs() > threshold) & (~invalid)
            # one-dim, boolean numpy array for easier manipulation, aligned with df.index
            arr = mask_exceeded.reindex(df.index, fill_value=False).astype(bool).values.copy()
            n = len(arr)
            # Find positions where gradient exceedings occur
            pos = np.flatnonzero(arr)

            # Case 0: No exceedings, continue to next column
            if pos.size == 0:
                continue
            
            # Case 1: A step in the data -> single gradient exceeding
            elif pos.size == 1:
                #print(f"Single gradient exceeding detected at index {df.index[pos[0]]} for column {col}.")
                # Mark current point as outlier
                # Note that potentially the neighboring points should also be marked as outliers
                # in order to "flatten" the spike and allow a reasonable chenge in data
                arr[pos[0]] = True
                # note the type of exceeding
                exceeded_type_map[df.index[pos[0]]].append(f"case1_step_{col}")


            # if more than one exceeding, analyze the segments between exceedings
            else:
                for start, end in zip(pos[:-1], pos[1:]):
                    #print("Found positions of gradient exceedings at indices:", start, end)
                    #print(f"Starttime: {df.index[start]}, Endtime: {df.index[end]}")
                    
                    # Assert that start and end well defined
                    assert start <=end, "Error 1 in gradient exceeding positions."
                    assert end - start >= 1, "Error 2 in gradient exceeding positions."

                    # Tow gradient exceedings in a row:
                    if end == start + 1:

                        # Case 2: Isolated spike -> jump up + jump down
                        if (gradient.iloc[start] * gradient.iloc[end] < 0):
                            #print("Isolated spike detected.")
                            # The outlier is the point in between the two gradient exceedings
                            arr[end] = False
                            arr[start] = True
                            try:
                                arr[end+1] = False
                            except IndexError:
                                pass
                            exceeded_type_map[df.index[start]] = [f"case2_isolated_spike_{col}"]
                            continue
                        # Case 1: A step in the data (two consecutive exceedings in same direction)
                        else:
                            #print("Consecutive gradient exceedings in same direction detected, marking both as step.")
                            exceeded_type_map[df.index[start]] = [f"case1_step_{col}"]
                            continue
                    
                    # compute interior (positions strictly between the two exceedings)
                    interior_len = end - start - 1

                    # check values between spikes when there is at least one interior element
                    if interior_len > 0:
                        interior = gradient.iloc[start+1:end]
                        #print(interior)

                        # Case 3: short time between strong gradient changes
                        # -> very likely a spike
                        if interior_len < SHORT_INTERIOR_THRESHOLD:
                            #print("Short time between strong gradient changes detected.")
                            # The outliers are the point in between the two gradient exceedings
                            arr[start:end] = True
                            arr[end] = False # update the original exceeding gradient to not be an outlier
                            for i in range(start, end):
                                exceeded_type_map[df.index[i]].append(f"case3_short_interior_{col}")
                            continue

                        # Case 4: interior gradients are (almost) all zero — step/plateau or sensor fault
                        # Mark as fault also if all but one interior gradient are zero.
                        elif len(interior) > 0:
                            # count interior gradients that are not "close to zero" (ignore NaNs)
                            interior_vals = interior.astype(float).fillna(0.0)
                            non_zero_count = (~np.isclose(interior_vals, 0.0, atol=GRADIENT_TOL)).sum()
                            # allow at most one interior gradient that is not close to zero
                            if non_zero_count <= 1:
                                #print("Step function or sensor fault detected (almost all zeros).")
                                arr[start:end] = True
                                arr[end] = False # update the original exceeding gradient to not be an outlier
                                for i in range(start, end):
                                    exceeded_type_map[df.index[i]].append(f"case4_step_function_or_sensor_fault_{col}") 
                                continue
                            
                        # If we check for an interior and find that the start is already marked as outlier,
                        # we need not check the interior, as we already marked the start of the segment.
                        # The end might still need to be marked in the next iteration.
                        elif arr[start-1]:
                            #print("Skipping already marked start exceeding!")
                            
                            # Case 1: A step in the data
                            # Special case: if the end is the last exceeding gradient, we need to mark it as well
                            # as there is no next iteration to check it. It belongs to case 1 then.
                            if end + 1 >= pos[-1]:
                                #print(f"Marking last exceeding at index {df.index[end]} as step.")
                                arr[end] = True
                                exceeded_type_map[df.index[end]].append(f"case1_step_{col}")
                            continue

                        # conservative fallback: mark the gradient exceedings only (similar to case 1)
                        #print("Conservative fallback: marking only the gradient exceedings as outliers.")
                        arr[start] = True
                        arr[end] = True
                        exceeded_type_map[df.index[start]].append(f"case1_step_{col}")
                        exceeded_type_map[df.index[end]].append(f"case1_step_{col}")
                        
            # Post-process: ensure isolated-spike/short-interior/step-function handling
            # did not leave a spurious 'case1_step_{col}' at the following timestamp.
            # Only remove a following 'case1_step' if it is clearly an artifact: the next
            # index must contain nothing but the matching single 'case1_step_{col}' entry
            # (i.e. it is not also part of another detection). This preserves legitimate
            # consecutive spikes.
            try:
                index_to_pos = {idx: pos for pos, idx in enumerate(df.index)}
                for idx, types in exceeded_type_map.items():
                    if not types:
                        continue
                    for t in types:
                        if not isinstance(t, str):
                            continue
                        # consider the patterns that should suppress a following single-step artifact
                        if t.startswith("case2_isolated_spike_"):
                            col_name = t[len("case2_isolated_spike_"):]
                        elif t.startswith("case3_short_interior_"):
                            col_name = t[len("case3_short_interior_"):]
                        elif t.startswith("case4_step_function_or_sensor_fault_"):
                            col_name = t[len("case4_step_function_or_sensor_fault_"):]
                        else:
                            continue

                        pos = index_to_pos.get(idx, None)
                        if pos is None:
                            continue
                        next_pos = pos + 1
                        if next_pos < len(arr):
                            next_idx = df.index[next_pos]
                            next_types = exceeded_type_map.get(next_idx, [])

                            # Determine whether next index contains only the single case1 artifact
                            has_case1_for_col = any(isinstance(nt, str) and nt == f"case1_step_{col_name}" for nt in next_types)
                            # non-case1 entries present besides the matching case1 -> do not remove
                            non_matching = [nt for nt in next_types if not (isinstance(nt, str) and nt == f"case1_step_{col_name}")]

                            if has_case1_for_col and len(non_matching) == 0:
                                # clear arr at next position (do not mark as outlier)
                                arr[next_pos] = False
                                # remove matching case1 entries at the next index
                                exceeded_type_map[next_idx] = [nt for nt in next_types if nt != f"case1_step_{col_name}"]
                                # remove column from exceeded_cols_map at next index if present
                                if col_name in exceeded_cols_map.get(next_idx, []):
                                    exceeded_cols_map[next_idx] = [c for c in exceeded_cols_map[next_idx] if c != col_name]
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not post-process gradient flagging: {e}")
                pass

            filled_mask = pd.Series(arr, index=df.index)

            # Mark rows that exceeded their gradient threshold
            df.loc[filled_mask, "is_outlier_gr_ex"] = True

            # Append column name to the list for rows that exceeded
            for idx in df.index[filled_mask]:
                exceeded_cols_map[idx].append(col)

        df["is_outlier_gr_ex_cols"] = [exceeded_cols_map[idx] for idx in df.index]
        df["outlier_gradient_type"] = [exceeded_type_map[idx] for idx in df.index]

        if self.verbose:
            print("\nGradient Threshold Exceeding Stats:")
            df_stats = df[df["is_outlier_gr_ex"] == True][["is_outlier_gr_ex_cols"]]
            if df_stats.empty:
                print("No values exceeded gradient thresholds.")
            else:
                # Explode the list column to count per (dataset_id, column)
                gradient_exceeded = df_stats["is_outlier_gr_ex_cols"].explode().dropna()
                
                if not gradient_exceeded.empty:
                    dataset_ids_gradient = gradient_exceeded.index.get_level_values("dataset_id")
                    gradient_exceeded_df = pd.DataFrame({
                        "dataset_id": dataset_ids_gradient,
                        "column": gradient_exceeded.values
                    })
                    gradient_exceeded_counts = gradient_exceeded_df.groupby(["dataset_id", "column"]).size().reset_index(name="is_outlier_gr_ex_cols")
                    display_result = gradient_exceeded_counts.sort_values(["dataset_id", "column"])
                    print(display_result.to_string(index=False))
        
        return df


    def _flag_outliers_drifts_flatlines(self, group):
        """
        Flag outliers based on drifts and flatlines in the data.
        This method computes the absolute gradient per column and flags values where the
        gradient indicates a drift or flatline.
        Stores:
        - is_outlier_drift (bool)
        - is_outlier_flatline (bool)
        - is_outlier_drift_flat (bool) (combination of both above)
        - is_outlier_drift_flat_cols (list of column names that exceeded)
        """
        df = group.copy()
        
        # Get numeric columns to process
        numeric = [c for c in self.numeric_cols if c not in ['dataset_id', 'turbine_id', 'timestamp', 'status_type_id']]

        # Initialize result columns and mappings for storing columns/types that are flagged
        df["is_outlier_drift"] = False
        df["is_outlier_flatline"] = False
        flagged_cols_map = {idx: [] for idx in df.index}
        flagged_type_map = {idx: [] for idx in df.index}

        # Get existing gradient exceedings to ignore them in drift/flatline detection
        if "is_outlier_gr_ex" not in df.columns:
            df["is_outlier_gr_ex"] = False
        gr_ex_outlier = df["is_outlier_gr_ex"] 

        # Buffer the gradient exceedings (needed as one point has two gradients...)
        gr_ex_buffer = gr_ex_outlier.shift(1, fill_value=False) | gr_ex_outlier | gr_ex_outlier.shift(-1, fill_value=False)

        # Compute gradient (for each numeric column
        for col in numeric:
            if col not in df.columns:
                continue
                
            # Compute signed gradient (not absolute!) for drift detection
            gradient_signed = df[col].diff()

            # --- Flatlines ------------------------------------------------------------------------
            gradient_flatline = gradient_signed.abs()
            is_flatline_raw = (gradient_flatline == 0)

            # Set is_flatline_raw to NaN where gradient exceedings are already marked (ignore these points)
            # Fill that NaNs forward and backward to create runs for flatline detection
            # This ensures a run is not interuppted by a spike outlier
            is_flatline_for_id = (
                is_flatline_raw.mask(gr_ex_buffer, np.nan)
                                .astype(bool)
                               .ffill().bfill()

            )
            # Create run IDs for flatline runs
            run_id = (is_flatline_for_id != is_flatline_for_id.shift()).cumsum()

            # Count the lenght of each flatline run (only counting valid flatline points, ignoring buffered gradient exceedings)
            run_counts = (
                is_flatline_raw.mask(gr_ex_buffer, np.nan)
                               .astype("Int64")
                               .groupby(run_id).transform("sum")
            )

            # Create mask for flatlines exceeding the set limit
            flatline_mask = is_flatline_raw & (~gr_ex_outlier) & (run_counts >= FLATLINE_THRESHOLDS.get(col))

            # Store results
            df.loc[flatline_mask, "is_outlier_flatline"] = True
            flagged_cols_map.update({idx: flagged_cols_map[idx] + [col] for idx in df.index[flatline_mask]})
            flagged_type_map.update({idx: flagged_type_map[idx] + [f"flatline_{col}"] for idx in df.index[flatline_mask]})


            # --- Drifts ---------------------------------------------------------------------------
            sign_series = np.sign(gradient_signed)
            # Bridge drifts over already marked points
            sign_series = sign_series.mask(gr_ex_outlier | df["is_outlier_flatline"], np.nan)
            valid_signs = (sign_series != 0) & (~sign_series.isna())
            has_positive = (sign_series == 1) & valid_signs
            has_negative = (sign_series == -1) & valid_signs

            # Create runs for positive and negative gradients separately
            # Positive gradients
            pos_for_id = (
                has_positive.mask(gr_ex_buffer, np.nan)
                            .astype(bool)
                            .ffill().bfill()
            )
            run_id_pos = (pos_for_id != pos_for_id.shift()).cumsum()
            run_counts_pos = (
                has_positive.mask(gr_ex_buffer, np.nan)
                            .astype("Int64")
                            .groupby(run_id_pos).transform("sum")
            )
            drift_mask_pos = has_positive & (run_counts_pos >= DRIFT_THRESHOLDS.get(col))

            # Negative gradients
            neg_for_id = (
                has_negative.mask(gr_ex_buffer, np.nan)
                            .astype(bool)
                            .ffill().bfill()
            )
            run_id_neg = (neg_for_id != neg_for_id.shift()).cumsum()
            run_counts_neg = (
                has_negative.mask(gr_ex_buffer, np.nan)
                            .astype("Int64")
                            .groupby(run_id_neg).transform("sum")
            )
            drift_mask_neg = has_negative & (run_counts_neg >= DRIFT_THRESHOLDS.get(col))

            # Combine positive and negative drift masks and ignore already marked outliers
            drift_mask = (drift_mask_pos | drift_mask_neg) & (~gr_ex_outlier) & (~df["is_outlier_flatline"])

            # Store results
            df.loc[drift_mask, "is_outlier_drift"] = True
            flagged_cols_map.update({idx: flagged_cols_map[idx] + [col] for idx in df.index[drift_mask]})
            flagged_type_map.update({idx: flagged_type_map[idx] + [f"drift_{col}"] for idx in df.index[drift_mask]})


        # Store results in dataframe----------------------------------------------------------------
        # Only update entries where existing value is an empty list; preserve non-empty lists
        if "outlier_gradient_type" in df.columns:
            existing_types = df["outlier_gradient_type"].copy()
        else:
            existing_types = pd.Series([[] for _ in range(len(df))], index=df.index)
        new_types = pd.Series([flagged_type_map[idx] for idx in df.index], index=df.index)

        def _merge_types(existing, new):
            if isinstance(existing, list) and len(existing) > 0:
                return existing
            return new

        df["outlier_gradient_type"] = [
            _merge_types(existing_types.loc[idx], new_types.loc[idx])
            for idx in df.index
        ]

        df["is_outlier_drift_flat_cols"] = [flagged_cols_map[idx] for idx in df.index]
        df["is_outlier_drift_flat"] = df["is_outlier_drift"] | df["is_outlier_flatline"]
        return df


    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in the dataframe according to the specified strategy.
        Strategies:
        - 'flag_only': keeps all flagged outliers in the data
        - 'drop': drop all rows that are marked as outliers if the outlier is not in one of the 
                    specified columns to ignore: 
                    wind_speed1_ms_avg, wind_speed2_ms_avg, wind_speed3_ms_avg, main_bearing_temp1_c_avg,
                    main_bearing_temp2_c_avg, main_bearing_temp3_c_avg
        Returns:
            DataFrame with outliers handled according to the specified strategy.

        """
        # keep the outliers in the data (also the is_outlier column so that one can work with it)
        if self.outlier_handling == 'flag_only':
            if self.verbose:
                n_outliers = df["is_outlier"].sum()
                print(f"\nOutlier Handling: 'flag_only' selected, keeping all {n_outliers} flagged outliers in the data.")
            return df
        
        # set outliers to nan
        if self.outlier_handling == "set_to_nan":
            # Track nan settings per dataset_id
            set_to_nan_stats = []
            
            def set_outlier_to_nan_group(group):
                dataset_id = group.index.get_level_values("dataset_id")[0]
                n_before_nans = group[self.numeric_cols].isna().sum().sum()
                
                # For each row that is marked as outlier, set the specific columns to NaN
                for idx in group.index[group["is_outlier"]]:
                    # Get the list of columns that caused the outlier flag
                    outlier_cols = group.loc[idx, "is_outlier_cols"]
                    
                    if isinstance(outlier_cols, list) and len(outlier_cols) > 0:
                        # Set only the problematic columns to NaN
                        for col in outlier_cols:
                            if col in group.columns:
                                group.loc[idx, col] = np.nan
                
                n_after_nans = group[self.numeric_cols].isna().sum().sum()
                n_set_to_nan = n_after_nans - n_before_nans
                
                set_to_nan_stats.append({
                    "dataset_id": dataset_id,
                    "values_set_to_nan": n_set_to_nan,
                    "outlier_rows": group["is_outlier"].sum()
                })
                
                return group

            df = df.groupby("dataset_id", group_keys=False).apply(set_outlier_to_nan_group)
            
            if self.verbose:
                print("\nOutlier Handling: 'set_to_nan' - Values set to NaN per dataset_id:")
                stats_df = pd.DataFrame(set_to_nan_stats)
                stats_df = stats_df[stats_df["values_set_to_nan"] > 0]
                if stats_df.empty:
                    print("No values were set to NaN.")
                else:
                    print(stats_df.to_string(index=False))
            
            return df


        if self.outlier_handling == 'drop':
            def is_outlier_effective(row):
                if not row["is_outlier"]:
                    return False
                # check if any of the outlier columns is not in the ignore list
                for col in row["is_outlier_cols"]:
                    if col not in COLS_TO_IGNORE_ON_DROPPING_OUTLIER_ROWS:
                        return True
                return False

            # Track drops per dataset_id
            drop_stats = []
            
            def drop_outliers_group(group):
                dataset_id = group.index.get_level_values("dataset_id")[0]
                n_before_group = len(group)
                effective_outlier_mask = group.apply(is_outlier_effective, axis=1)
                n_dropped_group = effective_outlier_mask.sum()
                drop_stats.append({
                    "dataset_id": dataset_id,
                    "rows_before": n_before_group,
                    "rows_dropped": n_dropped_group,
                    "percent_dropped": (n_dropped_group / n_before_group) * 100 if n_before_group > 0 else 0
                })
                return group[~effective_outlier_mask]

            n_before = len(df)
            df = df.groupby("dataset_id", group_keys=False).apply(drop_outliers_group)
            
            if self.verbose:
                print("\nRows dropped per dataset_id:")
                drop_stats_df = pd.DataFrame(drop_stats)
                drop_stats_df = drop_stats_df[drop_stats_df["rows_dropped"] > 0]
                if drop_stats_df.empty:
                    print("No rows were dropped from any dataset.")
                else:
                    print(drop_stats_df.to_string(index=False))

        else:
            raise NotImplementedError(f"Outlier handling strategy '{self.outlier_handling}' is not implemented.")

        return df