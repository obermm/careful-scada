"""
Outlier handling for SCADA data preprocessing.
Strategies for dealing with detected outliers.
"""

import pandas as pd
import numpy as np

def handle_outliers(
    df: pd.DataFrame,
    strategy: str = "flag_only",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Handle outliers in the dataframe according to the specified strategy.
    
    Args:
        df: DataFrame with outlier flags (is_outlier, is_outlier_cols)
        strategy: Strategy for handling outliers
            - 'flag_only': keeps all flagged outliers in the data
            - 'drop': drop all rows that contain an outlier
            - 'set_to_nan': set outlier column values to NaN in the respective columns
        verbose: If True, print statistics
    Returns:
        DataFrame with outliers handled according to the specified strategy.
    """
    # keep the outliers in the data (also the is_outlier column so that one can work with it)
    if strategy == 'flag_only':
        if verbose:
            n_outliers = df["is_outlier"].sum()
            print(f"\nOutlier Handling: 'flag_only' selected, keeping all {n_outliers} \
                  flagged outliers in the data.")
        return df

    # set outliers to nan
    if strategy == "set_to_nan":
        # Track nan settings per dataset_id
        set_to_nan_stats = []

        def set_outlier_to_nan_group(group):
            numeric_cols = group.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['asset_id', 'id']]
            dataset_id = group.index.get_level_values("dataset_id")[0]
            n_before_nans = group[numeric_cols].isna().sum().sum()

            # For each row that is marked as outlier, set the specific columns to NaN
            for idx in group.index[group["is_outlier"]]:
                # Get the list of columns that caused the outlier flag
                outlier_cols = group.loc[idx, "is_outlier_cols"]

                if isinstance(outlier_cols, list) and len(outlier_cols) > 0:
                    # Set only the problematic columns to NaN
                    for col in outlier_cols:
                        if col in group.columns:
                            group.loc[idx, col] = np.nan

            n_after_nans = group[numeric_cols].isna().sum().sum()
            n_set_to_nan = n_after_nans - n_before_nans

            set_to_nan_stats.append({
                "dataset_id": dataset_id,
                "values_set_to_nan": n_set_to_nan,
                "outlier_rows": group["is_outlier"].sum()
            })

            return group

        df = df.groupby("dataset_id", group_keys=False).apply(set_outlier_to_nan_group)

        if verbose:
            print("\nOutlier Handling: 'set_to_nan' - Values set to NaN per dataset_id:")
            stats_df = pd.DataFrame(set_to_nan_stats)
            stats_df = stats_df[stats_df["values_set_to_nan"] > 0]
            if stats_df.empty:
                print("No values were set to NaN.")
            else:
                print(stats_df.to_string(index=False))

        return df


    if strategy == 'drop':

        # Track drops per dataset_id
        drop_stats = []

        def drop_outliers_group(group):
            dataset_id = group.index.get_level_values("dataset_id")[0]
            n_before_group = len(group)
            outlier_mask = group["is_outlier"]
            n_dropped_group = outlier_mask.sum()
            drop_stats.append({
                "dataset_id": dataset_id,
                "rows_before": n_before_group,
                "rows_dropped": n_dropped_group,
                "percent_dropped": (n_dropped_group / n_before_group) * 100 if n_before_group > 0 else 0
            })
            return group[outlier_mask]

        df = df.groupby("dataset_id", group_keys=False).apply(drop_outliers_group)

        if verbose:
            print("\nRows dropped per dataset_id:")
            drop_stats_df = pd.DataFrame(drop_stats)
            drop_stats_df = drop_stats_df[drop_stats_df["rows_dropped"] > 0]
            if drop_stats_df.empty:
                print("No rows were dropped from any dataset.")
            else:
                print(drop_stats_df.to_string(index=False))

    else:
        raise NotImplementedError(f"Outlier handling strategy '{strategy}' \
                                  is not implemented.")

    return df
