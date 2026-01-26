"""
Visualization of the outlier detection test cases.
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Add the parent directory to the path so we can import care_scada
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_pre_flag_outliers import GRADIENT_THRESHOLD_TEST_CASES, THRESHOLD_TEST_CASES, \
    INTEGRATED_OUTLIER_FLAGGING_TEST_CASES, DRIFT_AND_FLATLINE_TEST_CASES, make_group
from care_scada.preprocessing import Preprocessor
from care_scada.preprocessing.outlier_detection import flag_outliers_gradient_thresholds, \
        flag_outliers_thresholds, flag_outliers_drifts_flatlines, flag_outliers



def plot_all_test_cases_in_one(test_cases, tests_type, save_dir="doc/fig/preprocessing/gradient_threshold"):
    """Plot all test cases in a single figure with matplotlib."""
    os.makedirs(save_dir, exist_ok=True)

    n_cases = len(test_cases)
    fig, axes = plt.subplots(n_cases, 2, figsize=(16, 4 * n_cases))

    # Ensure axes is always 2D
    if n_cases == 1:
        axes = axes.reshape(1, -1)

    categories_to_colors = {
        'absolute_threshold': 'pink',
        'case1_step': 'green',
        'case2_isolated_spike': 'orange',
        'case3_short_interior': 'purple',
        'case4_step_function_or_sensor_fault': 'brown',
        'case5_fallback': 'grey',
        'drift': 'cyan',
        'flatline': 'darkblue',
        'none': 'lightgrey'
    }

    # Plot each test case
    for idx, (ds_id, values, col, expected_count, case_name) in enumerate(test_cases):
        group = make_group(values, dataset_id=str(ds_id), col=col)
        if tests_type == "absolute_threshold":
            out = flag_outliers_thresholds(group)
        elif tests_type == "gradient_threshold":
            out = flag_outliers_gradient_thresholds(group, numeric_cols=[col])
            out["is_outlier_gradient"] = out["is_outlier_gr_ex"]
        elif tests_type == "drift_and_flatline":
            out = flag_outliers_drifts_flatlines(group, numeric_cols=[col])
        elif tests_type == "integrated_outlier_flagging":
            out = flag_outliers(group, numeric_cols=[col])
        timestamps = out.index.get_level_values("time_stamp")
        x_positions = np.arange(len(timestamps))

        # Left plot: Values with outliers
        ax_values = axes[idx, 0]
        ax_values.plot(x_positions, out[col], 'b-o', linewidth=2, markersize=5, label='Values')

        # Highlight gradient outliers
        if "is_outlier_gradient" in out.columns:
            outliers = out[out["is_outlier_gradient"]]
            if not outliers.empty:
                outlier_idx = [i for i, ts in enumerate(timestamps) if \
                               ts in outliers.index.get_level_values("time_stamp")]
                ax_values.scatter(outlier_idx, out[col].iloc[outlier_idx], color='red', s=100,
                                marker='x', linewidths=2, label='Gradient Outlier', zorder=5)

        # Highlight threshold outliers
        if "is_outlier_threshold" in out.columns:
            threshold_outliers = out[out["is_outlier_threshold"]]
            if not threshold_outliers.empty:
                threshold_idx = [i for i, ts in enumerate(timestamps) if \
                                 ts in threshold_outliers.index.get_level_values("time_stamp")]
                ax_values.scatter(threshold_idx, out[col].iloc[threshold_idx], color='orange', 
                                  s=100, marker='D', label='Threshold Outlier', zorder=5)

        ax_values.set_ylabel(col, fontsize=10)
        ax_values.set_title(f"Case {ds_id}: {case_name} (Expected Count: {expected_count})", \
                            fontsize=11, fontweight='bold')
        ax_values.grid(True, alpha=0.3)
        ax_values.legend(loc='upper right', fontsize=8)

        # Right plot: Flag categories
        if "is_outlier_gr_ex" not in out.columns:
            out["is_outlier_gr_ex"] = False
        if "is_outlier_drift" not in out.columns:
            out["is_outlier_drift"] = False
        if "is_outlier_flatline" not in out.columns:
            out["is_outlier_flatline"] = False
        if "is_outlier_threshold" not in out.columns:
            out["is_outlier_threshold"] = False
        flags = out["is_outlier_gr_ex"].astype(int)
        flags_drift = out["is_outlier_drift"].astype(int)
        flags_flatline = out["is_outlier_flatline"].astype(int)
        flags_threshold = out["is_outlier_threshold"].astype(int)
        combined_flags = flags + flags_drift + flags_flatline + flags_threshold
        combined_flags = combined_flags.apply(lambda x: 1 if x > 0 else 0)

        if "outlier_gradient_type" not in out.columns:
            out["outlier_gradient_type"] = None
        categories_gradient= out["outlier_gradient_type"].apply(
            lambda x: ','.join([item for item in x if item.endswith(f"_{col}")]) if x else ''
        )
        categories = categories_gradient.str.replace(f"_{col}", "", regex=False)

        # Build categories string: prioritize absolute_threshold, then gradient types
        categories_list = []
        for idx_row, row in out.iterrows():
            cat_parts = []
            # Add threshold first if present
            if row["is_outlier_threshold"]:
                cat_parts.append('absolute_threshold')
            # Add gradient types if present
            if isinstance(categories.loc[idx_row], str) and categories.loc[idx_row]:
                cat_parts.append(categories.loc[idx_row])
            cat_str = ','.join(cat_parts) if cat_parts else 'none'
            categories_list.append(cat_str)
        categories = pd.Series(categories_list, index=out.index)

        def _pick_color(cat_str):
            if not isinstance(cat_str, str) or cat_str == '' or cat_str == 'none':
                return categories_to_colors['none']
            # Prioritize absolute_threshold over other categories
            if 'absolute_threshold' in cat_str:
                return categories_to_colors['absolute_threshold']
            for c in cat_str.split(','):
                if c in categories_to_colors:
                    return categories_to_colors[c]
            return categories_to_colors['none']

        color_list = [_pick_color(cat) for cat in categories]

        ax_flags = axes[idx, 1]
        ax_flags.bar(x_positions, combined_flags, color=color_list, alpha=0.8,
                           edgecolor='black', linewidth=0.5)
        ax_flags.set_ylabel('Outlier Flag', fontsize=10)
        ax_flags.set_title(f"Case {ds_id}: {case_name} (Categories)", \
                            fontsize=11, fontweight='bold')
        ax_flags.set_ylim([0, 1.1])
        ax_flags.grid(True, alpha=0.3, axis='y')

        # Add legend for categories that appear in this test case
        unique_categories = set()
        for cat_str in categories:
            if isinstance(cat_str, str) and cat_str != '' and cat_str != 'none':
                for c in cat_str.split(','):
                    if c in categories_to_colors:
                        unique_categories.add(c)

        if unique_categories:
            legend_elements = [mpatches.Patch(color=categories_to_colors[cat], label=cat) 
                             for cat in sorted(unique_categories)]
            ax_flags.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # Synchronize x-axis for both plots
        ax_values.set_xlim([-0.5, len(timestamps) - 0.5])
        ax_flags.set_xlim([-0.5, len(timestamps) - 0.5])

    plt.tight_layout()

    # Save as PNG
    filename_png = "all_test_cases_combined.png"
    filepath_png = os.path.join(save_dir, filename_png)
    plt.savefig(filepath_png, dpi=100, bbox_inches='tight')
    print(f"✓ PNG saved: {filepath_png}")
    plt.close()


if __name__ == "__main__":
    # Set which tests to plot:
    # options:
    # "absolute_threshold",
    # "gradient_threshold",
    # "drift_and_flatline",
    # "integrated_outlier_flagging"
    TESTS_TO_PLOT = "integrated_outlier_flagging"
    TEST_CASES_TO_PLOT = None  # Set to a list of integers to plot only those test cases

    if TESTS_TO_PLOT == "absolute_threshold":
        TEST_DATA = THRESHOLD_TEST_CASES
    elif TESTS_TO_PLOT == "gradient_threshold":
        TEST_DATA = GRADIENT_THRESHOLD_TEST_CASES
    elif TESTS_TO_PLOT == "drift_and_flatline":
        TEST_DATA = DRIFT_AND_FLATLINE_TEST_CASES
    elif TESTS_TO_PLOT == "integrated_outlier_flagging":
        TEST_DATA = INTEGRATED_OUTLIER_FLAGGING_TEST_CASES
    else:
        raise ValueError(f"Unknown tests_to_plot: {TESTS_TO_PLOT}")

    if TEST_CASES_TO_PLOT is not None:
        TEST_DATA = [tc for tc in TEST_DATA if tc[0] in TEST_CASES_TO_PLOT]

    print(f"Plotting {len(TEST_DATA)} test cases with Matplotlib...")
    pre = Preprocessor(
        farm_name="A",
        numeric_cols=["wind_speed1_ms_avg", "outside_temp_c_avg", "main_bearing_temp2_c_avg"],
        verbose=False
    )

    # Plot all cases in one combined plot
    plot_all_test_cases_in_one(TEST_DATA, TESTS_TO_PLOT, \
                               save_dir=f"doc/fig/preprocessing/{TESTS_TO_PLOT}")
    print(f"✓ Done! Check the 'doc/fig/preprocessing/{TESTS_TO_PLOT}' directory.")
