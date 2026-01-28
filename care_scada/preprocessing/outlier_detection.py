"""
Outlier detection for SCADA data preprocessing.
Implements threshold-based, gradient-based, and drift/flatline detection.
"""

import pandas as pd
import numpy as np
from care_scada.sensor_config import get_sensor_config

# Load thresholds from sensor config
_config = get_sensor_config()
ABSOLUTE_THRESHOLDS = _config.get_thresholds('absolute')
GRADIENT_THRESHOLDS = _config.get_thresholds('gradient')
DRIFT_THRESHOLDS = _config.get_thresholds('drift')
FLATLINE_THRESHOLDS = _config.get_thresholds('flatline')
SHORT_INTERIOR_THRESHOLD = _config.get_setting('short_interior_threshold')
GRADIENT_TOL = _config.get_setting('gradient_tolerance')
RARELY_ZERO_SENSORS = _config.get_rarely_zero_sensors()

def flag_outliers(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Flag outliers in the dataframe.
    Steps:
    - Apply threshold based outlier detection (eg. no negative wind speeds)
    - Apply gradient based outlier detection if specified (eg. no sudden jumps in values)
    - Find drifts and flatlines (eg. no long periods of constant values)
    
    Args:
        df: DataFrame with MultiIndex (dataset_id, time_stamp)
        verbose: If True, print statistics
    Returns:
        DataFrame with outlier flags added:
        - is_outlier_gradient: True if gradient based outlier detection flagged the value
        - is_outlier_threshold: True if threshold based outlier detection flagged the value
        - is_outlier: True if any outlier flag is set
        - is_outlier_cols: List of columns that caused the outlier flag
    """
    # apply thresholds
    df = flag_outliers_thresholds(df, verbose)

    # apply gradient based method
    df = df.groupby("dataset_id", group_keys=False).apply(
        lambda group: flag_outliers_gradient_thresholds(group, verbose)
    )

    # Find drifts and flatlines
    df = df.groupby("dataset_id", group_keys=False).apply(
        lambda group: flag_outliers_drifts_flatlines(group, verbose)
    )

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


def flag_outliers_thresholds(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply predefined thresholds to flag extreme values.
    Thresholds are defined per column as (min, max) tuples.

    Args:
        df: DataFrame with MultiIndex (dataset_id, time_stamp)
        verbose: If True, print statistics
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
    df["is_outlier_threshold_cols"] = df.apply(lambda row: list(set(row["threshold_too_low"] + \
                                                            row["threshold_too_high"])), axis=1)

    if verbose:
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
            too_low_df["min_threshold"] = too_low_df["column"].map(lambda col: \
                                                    ABSOLUTE_THRESHOLDS.get(col, (None, None))[0])
            too_low_df["max_threshold"] = too_low_df["column"].map(lambda col: \
                                                    ABSOLUTE_THRESHOLDS.get(col, (None, None))[1])
            too_low_counts = too_low_df.groupby(["dataset_id", "column", "min_threshold", \
                                        "max_threshold"]).size().reset_index(name="too_low_count")

            # too_high
            too_high = df_stats["threshold_too_high"].explode().dropna()
            dataset_ids_high = too_high.index.get_level_values("dataset_id")
            too_high_df = pd.DataFrame({
                "dataset_id": dataset_ids_high,
                "column": too_high.values
            })
            too_high_df["min_threshold"] = too_high_df["column"].map(lambda col: \
                                                    ABSOLUTE_THRESHOLDS.get(col, (None, None))[0])
            too_high_df["max_threshold"] = too_high_df["column"].map(lambda col: \
                                                    ABSOLUTE_THRESHOLDS.get(col, (None, None))[1])
            too_high_counts = too_high_df.groupby(["dataset_id", "column", "min_threshold", \
                                        "max_threshold"]).size().reset_index(name="too_high_count")

            result = too_low_counts.merge(
                too_high_counts,
                on=["dataset_id", "column", "min_threshold", "max_threshold"],
                how="outer"
            ).fillna(0)

            result = result.astype({"too_low_count": int, "too_high_count": int})
            result["allowed_range"] = "[" + result["min_threshold"].astype(str) + ", " + result["max_threshold"].astype(str) + "]"
            display_result = result[["dataset_id", "column", "allowed_range", "too_low_count", \
                                      "too_high_count"]].sort_values(["dataset_id", "column"])
            print(display_result.set_index("dataset_id").to_string())

    return df


def flag_outliers_gradient_thresholds(
    group: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Flag outliers based on sudden changes (gradients) in the data that exceed manually defined 
    thresholds. This method computes the absolute gradient per column and flags values where the
    gradient exceeds a predefined threshold.
    
    Args:
        group: DataFrame group for a single dataset_id
        verbose: If True, print statistics
    Returns:
        DataFrame with outlier flags:
        - is_outlier_gr_ex (bool)
        - is_outlier_gr_ex_cols (list of column names that exceeded)
        - outlier_gradient_type (str): type of gradient exceeding detected
    """
    df = group.copy()

    # Initialize column and mapping for exceeded gradients and types
    df["is_outlier_gr_ex"] = False
    exceeded_cols_map = {idx: [] for idx in df.index}
    exceeded_type_map = {idx: [] for idx in df.index}

    # Compute gradients for each column with defined threshold thats in the dataframe
    for col, threshold in GRADIENT_THRESHOLDS.items():
        if col not in df.columns:
            continue

        # Compute gradient for the current column relative to the previous timestamp
        # Only consider gradients where the previous timestamp is not longer ago than 30min
        # earlier. If the previous timestamp is further in the past (e.g. due to missing rows),
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

        # Time difference to the last valid observation
        time_diff = times_series - last_valid_time

        # invalid where time gap is larger than 30 minutes
        invalid = (time_diff >= pd.Timedelta(minutes=30))

        # Create mask for absolute gradient exceedings, but only where valid
        mask_exceeded = (gradient.abs() > threshold) & (~invalid)
        # one-dim, boolean numpy array for easier manipulation, aligned with df.index
        arr = mask_exceeded.reindex(df.index, fill_value=False).astype(bool).values.copy()
        # Find positions where gradient exceedings occur
        pos = np.flatnonzero(arr)

        # Case 0: No exceedings, continue to next column
        if pos.size == 0:
            continue

        # Case 1: A step in the data -> single gradient exceeding
        if pos.size == 1:
            # Mark current point as outlier
            # Note that potentially the neighboring points should also be marked as outliers
            # in order to "flatten" the spike and allow a reasonable change in data
            arr[pos[0]] = True
            # note the type of exceeding
            exceeded_type_map[df.index[pos[0]]].append(f"case1_step_{col}")


        # if more than one exceeding, analyze the segments between exceedings
        else:
            for start, end in zip(pos[:-1], pos[1:]):
                # Assert that start and end well defined
                assert start <=end, "Error 1 in gradient exceeding positions."
                assert end - start >= 1, "Error 2 in gradient exceeding positions."

                # Tow gradient exceedings in a row:
                if end == start + 1:

                    # Case 2: Isolated spike -> jump up + jump down
                    if gradient.iloc[start] * gradient.iloc[end] < 0:
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
                        arr[end] = False # update the original exceeding gradient to not outlier
                        for i in range(start, end):
                            exceeded_type_map[df.index[i]].append(f"case3_short_interior_{col}")
                        continue

                    # Case 4: interior gradients are (almost) all zero — plateau or sensor fault
                    # Marking as fault also if all but one interior gradient are zero.
                    elif len(interior) > 0:
                        # count interior gradients that are not "close to zero" (ignore NaNs)
                        interior_vals = interior.astype(float).fillna(0.0)
                        non_zero_count = (~np.isclose(interior_vals, 0.0, atol=GRADIENT_TOL)).sum()

                        # allow at most one interior gradient that is not close to zero
                        if non_zero_count <= 1:
                            # flag all interior points as outliers
                            arr[start:end] = True
                            # update the original exceeding gradient to not outlier
                            arr[end] = False 
                            for i in range(start, end):
                                exceeded_type_map[df.index[i]].append(f"case4_step_function_or_sensor_fault_{col}")
                            continue

                        # If we check for an interior and find that the start is already marked as
                        # outlier, we need not check the interior, as we already marked the start
                        # of the segment. The end might still need to be marked in the next
                        # iteration.
                        elif arr[start-1]:
                            # Case 1: A step in the data
                            # Special case: if the end is the last exceeding gradient, we need to  
                            # mark it as well as there is no next iteration to check it. 
                            # It belongs to case 1 then.
                            if end + 1 >= pos[-1]:
                                arr[end] = True
                                exceeded_type_map[df.index[end]].append(f"case1_step_{col}")
                            continue

                        #conservative fallback: mark the gradient exceedings only (similar to case1)
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
                        #consider the patterns that should suppress a following single-step artifact
                        if t.startswith("case2_isolated_spike_"):
                            col_name = t[len("case2_isolated_spike_"):]
                        elif t.startswith("case3_short_interior_"):
                            col_name = t[len("case3_short_interior_"):]
                        elif t.startswith("case4_step_function_or_sensor_fault_"):
                            col_name = t[len("case4_step_function_or_sensor_fault_"):]
                        else:
                            continue

                        pos_idx = index_to_pos.get(idx, None)
                        if pos_idx is None:
                            continue
                        next_pos = pos_idx + 1
                        if next_pos < len(arr):
                            next_idx = df.index[next_pos]
                            next_types = exceeded_type_map.get(next_idx, [])

                            # Determine whether next index contains only the single case1 artifact
                            has_case1_for_col = any(isinstance(nt, str) and \
                                                nt == f"case1_step_{col_name}" for nt in next_types)
                            # non-case1 entries present besides the matching case1 -> do not remove
                            non_matching = [nt for nt in next_types if not \
                                        (isinstance(nt, str) and nt == f"case1_step_{col_name}")]

                            if has_case1_for_col and len(non_matching) == 0:
                                # clear arr at next position (do not mark as outlier)
                                arr[next_pos] = False
                                # remove matching case1 entries at the next index
                                exceeded_type_map[next_idx] = \
                                    [nt for nt in next_types if nt != f"case1_step_{col_name}"]
                                # remove column from exceeded_cols_map at next index if present
                                if col_name in exceeded_cols_map.get(next_idx, []):
                                    exceeded_cols_map[next_idx] = \
                                        [c for c in exceeded_cols_map[next_idx] if c != col_name]
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not post-process gradient flagging: {e}")
                pass

        # Common post-processing for both single and multiple exceedings
        filled_mask = pd.Series(arr, index=df.index)

        # Mark rows that exceeded their gradient threshold
        df.loc[filled_mask, "is_outlier_gr_ex"] = True

        # Append column name to the list for rows that exceeded
        for idx in df.index[filled_mask]:
            exceeded_cols_map[idx].append(col)

    df["is_outlier_gr_ex_cols"] = [exceeded_cols_map[idx] for idx in df.index]
    df["outlier_gradient_type"] = [exceeded_type_map[idx] for idx in df.index]

    if verbose:
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
                gradient_exceeded_counts = gradient_exceeded_df\
                    .groupby(["dataset_id", "column"])\
                    .size().reset_index(name="is_outlier_gr_ex_cols")
                display_result = gradient_exceeded_counts.sort_values(["dataset_id", "column"])
                print(display_result.to_string(index=False))

    return df


def flag_outliers_drifts_flatlines(
    group: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Flag outliers based on drifts and flatlines in the data.
    This method computes the absolute gradient per column and flags values where the
    gradient indicates a drift or flatline.
    
    Args:
        group: DataFrame group for a single dataset_id
        verbose: If True, print statistics
    Returns:
        DataFrame with outlier flags:
        - is_outlier_drift (bool)
        - is_outlier_flatline (bool)
        - is_outlier_drift_flat (bool) (combination of both above)
        - is_outlier_drift_flat_cols (list of column names that exceeded)
    """
    df = group.copy()

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

    # Compute gradient (for each column with defined thresholds in the dataframe)
    for col, threshold  in FLATLINE_THRESHOLDS.items():
        if col not in df.columns:
            continue

        # Compute signed gradient (not absolute!) for drift detection
        gradient_signed = df[col].diff()

        # --- Flatlines ------------------------------------------------------------------------
        gradient_flatline = gradient_signed.abs()
        is_flatline_raw = (gradient_flatline == 0)

        # Set is_flatline_raw to NaN where gradient exceedings are already marked
        # (ignore these points)
        # Fill that NaNs forward and backward to create runs for flatline detection
        # This ensures a run is not interuppted by a spike outlier
        is_flatline_for_id = (
            is_flatline_raw.mask(gr_ex_buffer, np.nan)
                            .astype(bool)
                           .ffill().bfill()

        )
        # Create run IDs for flatline runs
        run_id = (is_flatline_for_id != is_flatline_for_id.shift()).cumsum()

        # Count the lenght of each flatline run (only counting valid flatline points
        # ignoring buffered gradient exceedings)
        run_counts = (
            is_flatline_raw.mask(gr_ex_buffer, np.nan)
                           .astype("Int64")
                           .groupby(run_id).transform("sum")
        )

        # Create mask for flatlines exceeding the set limit
        flatline_mask = is_flatline_raw & (~gr_ex_outlier) & \
                        (run_counts >= threshold)

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
        drift_mask = (drift_mask_pos | drift_mask_neg) & \
                     (~gr_ex_outlier) & (~df["is_outlier_flatline"])

        # Store results
        df.loc[drift_mask, "is_outlier_drift"] = True
        flagged_cols_map.update({idx: flagged_cols_map[idx] + \
                                 [col] for idx in df.index[drift_mask]})
        flagged_type_map.update({idx: flagged_type_map[idx] + \
                                 [f"drift_{col}"] for idx in df.index[drift_mask]})


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
