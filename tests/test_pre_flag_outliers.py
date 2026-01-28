"""
Tests for outlier flagging (_flag_outliers) in Preprocessor.
First single methods are testes in isolation, then the integrated method.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add the parent directory to the path so we can import care_scada
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from care_scada.preprocessing import Preprocessor
from care_scada.preprocessing.outlier_detection import (
    flag_outliers_thresholds,
    flag_outliers_gradient_thresholds,
    flag_outliers_drifts_flatlines,
    flag_outliers
)
from care_scada.sensor_config import get_sensor_config

_config = get_sensor_config()
DRIFT_THRESHOLDS = _config.get_thresholds('drift')
FLATLINE_THRESHOLDS = _config.get_thresholds('flatline')
GRADIENT_THRESHOLDS = _config.get_thresholds('gradient')

def make_group(values, dataset_id="1", col="wind_speed1_ms_avg", start="2022-01-01 00:00:00",
               freq="10min"):
    """Helper to create a test DataFrame with a MultiIndex as expected by Preprocessor methods."""
    times = pd.date_range(start=start, periods=len(values), freq=freq)
    idx = pd.MultiIndex.from_product([[dataset_id], times], names=["dataset_id", "time_stamp"])
    return pd.DataFrame({col: values}, index=idx)

def flagged_count(out_df, flag_col="is_outlier_gr_ex"):
    """Count number of flagged outliers in the DataFrame."""
    return int(out_df[flag_col].sum())

def flagged_cols_contains(out_df, colname, flag_col="is_outlier_gr_ex"):
    """Check if the flagged columns list contains the specified column name."""
    flagged = out_df[out_df[flag_col]]
    return any(isinstance(lst, list) and colname in lst for lst in flagged[f"{flag_col}_cols"])

# Get thresholds - fail early if keys are missing
mb_threshold = GRADIENT_THRESHOLDS.get("main_bearing_temp2_c_avg")
ws_threshold = GRADIENT_THRESHOLDS.get("wind_speed1_ms_avg")
ot_threshold = GRADIENT_THRESHOLDS.get("outside_temp_c_avg")


# Flagging for values that exceed absolute thresholds
THRESHOLD_TEST_CASES = [
    # Case 0: No outliers
    (0, [5]*10, "wind_speed1_ms_avg", 0, "no outliers"),
    (1, [10]*10, "outside_temp_c_avg", 0, "no outliers"),

    # Case 1: Single outlier
    (11, [5, 5, 51, 5, 5], "wind_speed1_ms_avg", 1, "single high outlier"),
    (12, [10, 10, -51, 10, 10], "outside_temp_c_avg", 1, "single low outlier"),

    # Case 2: Multiple outliers
    (21, [5, 100, 5, -50, 5, 200], "wind_speed1_ms_avg", 3, "multiple outliers"),
    (22, [10, -51, 10, 51, 10, -51], "outside_temp_c_avg", 3, "multiple outliers"),

    # Case 3: Outliers below threshold
    (31, [5, 49, 5, 5, 5], "wind_speed1_ms_avg", 0, "outliers below threshold"),
    (32, [10, -49, 10, 49, 10], "outside_temp_c_avg", 0, "outliers below threshold"),
]

@pytest.mark.parametrize("ds_id, values, col, expected_count, case", THRESHOLD_TEST_CASES)
def test_threshold_outlier_detection(ds_id, values, col, expected_count, case):
    group = make_group(values, dataset_id=str(ds_id), col=col)
    out = flag_outliers_thresholds(group, verbose=False)

    assert flagged_count(out, flag_col="is_outlier_threshold") == expected_count, f"{case}: expected {expected_count}, got {flagged_count(out, flag_col='is_outlier_threshold')}"
    if expected_count > 0:
        assert flagged_cols_contains(out, col, flag_col="is_outlier_threshold"), f"{case}: expected column name in threshold_exceeded_cols"


# Flagging for gradient threshold exceedings
GRADIENT_THRESHOLD_TEST_CASES = [
    # Case 0: No exceeding gradients
    (0, [5]*5, "wind_speed1_ms_avg", 0, "no exceedings"),
    (1, [10]*5, "outside_temp_c_avg", 0, "no exceedings"),

    # Case 1: single exceeding gradients (no interval to fill)
    # Case 1.1: interior point
    (11, [5]*5 + [5+ws_threshold+1] + [25]*5, "wind_speed1_ms_avg", 1, "single spike up"),
    (11, [30]*5 + [5-ot_threshold-1] + [5]*5, "outside_temp_c_avg", 1, "single spike down"),
    # Case 1.2: edge cases start/end point
    (12, [20] + [20-ws_threshold-1]*5, "wind_speed1_ms_avg", 1, "start spike down"),
    (12, [5]*5 + [5+ot_threshold+1], "outside_temp_c_avg", 1, "end spike up"),

    # Case 2:
    # true isolated spike -> jump up + jump down => 1 flagged data point in between
    (21, [5]*5 + [20] + [5]*5, "wind_speed1_ms_avg", 1, "isolated spike up"),
    (22, [5]*5 + [-10] + [5]*5, "outside_temp_c_avg", 1, "isolated spike down"),
    #  spike below threshold -> no flags
    (23, [5]*5 + [10] + [5]*5, "wind_speed1_ms_avg", 0, "isolated spike up below threshold"),
    (24, [5]*5 + [8] + [5]*5, "outside_temp_c_avg", 0, "isolated spike up below threshold"),
    (25, [5, 49, 5, 5, 5], "wind_speed1_ms_avg", 1, "single spike up"),
    (26, [10, -49, 49, -49, 40, 10], "outside_temp_c_avg", 4, "four spikes"),

    # Case 3:
    # less than 2 hours or less of interior points between strong gradient changes
    # short interior -> code fills entire interval between up and down (inclusive up to 12 datapoints)
    (31, [5]*5 + [20]*5 + [5]*5, "wind_speed1_ms_avg", 5, "short interior"),
    (32, [0]*5 + [20, 20] + [0]*5, "wind_speed1_ms_avg", 2, "short interior minimal case"),
    (33, [5]*5 + [-10]*5 + [5]*5, "outside_temp_c_avg", 5, "short interior"),
    # long interior with non zero gradients-> only flags spikes +/- one datapoint, not entire interval => 2 flagged rows
    (34, [5]*5 + [20]*4 + [21]*4 + [16]*4 + [17]*4 + [3]*5, "wind_speed1_ms_avg", 2, "long interior with non zero gradients"),
    (35, [5]*5 + [-10]*4 + [-11]*4  + [-7]*4 + [-6]*4 + [5]*5, "outside_temp_c_avg", 2, "long interior with non zero gradients"),

    # Case 4:
    # step up/down plateau with only zero gradient between strong gradient changes
    # step up plateau -> interval covers up..down inclusive => 13 flagged rows
    (41, [5]*4 + [20]*13 + [5]*4, "wind_speed1_ms_avg", 13, "step up plateau"),
    (42, [15]*5 + [0]*13 + [15]*5, "wind_speed1_ms_avg", 13, "step down plateau"),
    (43, [20]*5 + [0]*13 + [20]*5, "outside_temp_c_avg", 13, "step down plateau"),

    # Test combinations of multiple cases
    (51, [5]*5 + [20] + [5]*5 + [6]*5 + [7]*5 + [-10] + [5]*5, "wind_speed1_ms_avg", 2, "combination single spikes"),
    (52, [5]*5 + [20]*5 + [5]*5 + [6]*5 + [7]*5 + [-10]*5 + [5]*5, "outside_temp_c_avg", 10, "combination short interiors"),
    (53, [5]*5 + [20]*13 + [5]*5 + [6]*5 + [7]*5 + [20]*7 + [5]*5, "wind_speed1_ms_avg", 20, "combination step plateau + short interior"),
    (54, [5]*5 + [20]*13 + [5]*5 + [6]*5 + [7]*5 + [30] + [5]*5, "wind_speed1_ms_avg", 14, "combination step plateau + spike"),
    (55, [5]*5 + [20] + [25]*7 + [30]*5 + [5] + [30]*2, "wind_speed1_ms_avg", 2, "combination single spike up + spike"),
    (56, [5]*5 + [20] + [5]*7 + [8] + [6]*7 + [30]*5, "wind_speed1_ms_avg", 2, "combination single spike up + spike"),

    (57, [50]*5 + [50+mb_threshold+1] + [60+mb_threshold+1] + [55]*5 , "main_bearing_temp2_c_avg", 2, "combination step + spike"),
    (58, [50]*5 + [50+mb_threshold+1] + [50+mb_threshold+2]*3 + [50]*5 , "main_bearing_temp2_c_avg", 4, "combination step + short interior"),
    (59, [50]*5 + [50+mb_threshold+1] + [50+mb_threshold+2]*3 + [40]*1 + [50]*5 , "main_bearing_temp2_c_avg", 5, "combination step + short interior"),

    (60, [50]*5 + [30]*2 + [50]*1 + [50]*2 + [30]*2 + [50]*1 + [51]*5 , "main_bearing_temp2_c_avg", 7, "combination step + short interior"),
    (61, [50]*5 + [30]*2 + [50]*5 + [51]*5 + [52]*5 + [30]*2 + [50]*1 + [51]*5 , "main_bearing_temp2_c_avg", 4, "combination step + long interior"),

    # When spikes are close together, the algorithm groups them as "short interior" pattern
    (80, [5] + [20]*2 + [5] + [21]*2 + [5] + [22]*2 + [5] + [23]*2 + [5]*20, "wind_speed1_ms_avg", 11, "dense spikes - alternating pattern"),
    (81, [5]*3 + [25, 5, 26, 5, 27, 5, 28, 5, 29, 5] + [5]*20, "wind_speed1_ms_avg", 9, "dense spikes - scattered"),
    (82, [5]*3 + [25, 5, 5, 26, 5, 5, 5, 27, 5, 5,  28, 5, 5, 29, 5] + [5]*20, "wind_speed1_ms_avg", 14, "dense spikes - scattered"),
]

@pytest.mark.parametrize("ds_id, values, col, expected_count, case", GRADIENT_THRESHOLD_TEST_CASES)
def test_gradient_threshold(ds_id, values, col, expected_count, case):
    group = make_group(values, dataset_id=str(ds_id), col=col)
    out = flag_outliers_gradient_thresholds(group, verbose=False)

    assert flagged_count(out) == expected_count, f"{case}: expected {expected_count}, got {flagged_count(out)}"
    if expected_count > 0:
        assert flagged_cols_contains(out, col), f"{case}: expected column name in gradient_exceeded_cols"



# Flagging for drifts and flatlines
# Use centralized thresholds for tests (per-column values)
DRIFT_LIMIT = DRIFT_THRESHOLDS.get('main_bearing_temp2_c_avg')
FLATLINE_LIMIT = FLATLINE_THRESHOLDS.get('main_bearing_temp2_c_avg')

DRIFT_AND_FLATLINE_TEST_CASES = [
    (2, [50]*5 + np.arange(50, 55, 1).tolist() + [54]*5 , "main_bearing_temp2_c_avg", 0, "Short drift up - no flags"),
    (3, [50]*5 + np.arange(50, 70, 0.005).tolist() + [70]*5 , "main_bearing_temp2_c_avg", 4000, "Long drift up - flags"),
    (4, [50]*(FLATLINE_LIMIT+1), "main_bearing_temp2_c_avg", FLATLINE_LIMIT, "Flatline"),
    (5, [50]*(FLATLINE_LIMIT-1) + [51], "main_bearing_temp2_c_avg", 0, "Short flatline - no flags"),
    (6, [50]*5 + [51]*(FLATLINE_LIMIT+1)+ [50]*5, "main_bearing_temp2_c_avg", FLATLINE_LIMIT, "Flatline in middle"),
    (7, [50]*5 + [70]* (FLATLINE_LIMIT -1) + [50]*5, "main_bearing_temp2_c_avg", 0, "Short flatline in middle - no flags"),
    (8, [50]*(FLATLINE_LIMIT-5) + np.arange(50, 70, 0.005).tolist() + [70]*(FLATLINE_LIMIT-5), "main_bearing_temp2_c_avg", 4000, " short flatline + long drift up + short latline"),
    ("8a", [50]*(FLATLINE_LIMIT+5) + np.arange(50, 70, 0.005).tolist() + [70]*(FLATLINE_LIMIT+5), "main_bearing_temp2_c_avg", 4000+2*(FLATLINE_LIMIT+5)-1, " long flatline + long drift up + long latline"),

    # Flatline or drift interrupted by spikes -> no flagging in the pure flatline/drift detection
    (70, [50]*6 + [70] + [50]*6, "main_bearing_temp2_c_avg", 0, "Short Flatline + spike up"),
    (71, [50]*20 + [70] + [50]*20, "main_bearing_temp2_c_avg", 0, "Flatline + spike up"),
    (72, np.arange(50, 60, 0.1).tolist() + [40] + np.arange(60, 65, 0.1).tolist() ,"main_bearing_temp2_c_avg", 0, "One long drift up, interuppted by Spike down"),
    (73, np.arange(50, 60, 0.01).tolist() + [40] + np.arange(60, 70, 0.01).tolist() ,"main_bearing_temp2_c_avg", 1999, "Two long drift ups + Spike down"),
]

@pytest.mark.parametrize("ds_id, values, col, expected_count, case", DRIFT_AND_FLATLINE_TEST_CASES)
def test_drift_and_flatline_detection(ds_id, values, col, expected_count, case):
    group = make_group(values, dataset_id=str(ds_id), col=col)
    out = flag_outliers_drifts_flatlines(group, verbose=False)
    assert flagged_count(out, flag_col="is_outlier_drift_flat") == expected_count, f"{case}: expected {expected_count}, got {flagged_count(out, flag_col='is_outlier_drift_flat')}"
    if expected_count > 0:
        assert flagged_cols_contains(out, col, flag_col="is_outlier_drift_flat"), f"{case}: expected column name in gradient_exceeded_cols"


  # Integrated test of _flag_outliers with gradient thresholds, drift and flatline detection
INTEGRATED_OUTLIER_FLAGGING_TEST_CASES = [
    # Threshold tests from above:
    # Case 0: No outliers
    (0, [5]*10, "wind_speed1_ms_avg", 0, "no outliers"),
    (1, [10]*10, "outside_temp_c_avg", 0, "no outliers"),

    # Case 1: Single outlier
    (11, [5, 5, 5+ws_threshold+1, 5, 5], "wind_speed1_ms_avg", 1, "single high outlier"),
    (12, [10, 10, 10 - ot_threshold - 1, 10, 10], "outside_temp_c_avg", 1, "single low outlier"),

    # Case 2: Multiple outliers
    (21, [10, 10+ws_threshold+1, 10, 10 - ws_threshold - 1, 10, 10+ws_threshold+1, 10], "wind_speed1_ms_avg", 3, "multiple outliers"),
    (22, [10, 10 - ot_threshold - 1, 10, 10 + ot_threshold + 1, 10, 10 - ot_threshold - 1, 10], "outside_temp_c_avg", 3, "multiple outliers"),

    # Case 3: Outliers below threshold -> no flags due to gradient thresholding
    (31, [5, 49, 5, 5, 5], "wind_speed1_ms_avg", 1, "single spike up"),
    (32, [10, -49, 10, 49, 10], "outside_temp_c_avg", 2, "two spikes"),

    # Gradient threshold tests from above:
    # Case 0: No exceeding gradients
    (0, [5]*5, "wind_speed1_ms_avg", 0, "no exceedings"),
    (1, [10]*5, "outside_temp_c_avg", 0, "no exceedings"),

    # Case 1: single exceeding gradients (no interval to fill)
    # Case 1.1: interior point
    (11, [5]*5 + [20] + [25]*5, "wind_speed1_ms_avg", 1, "single spike up"),
    (11, [30]*5 + [10] + [5]*5, "outside_temp_c_avg", 1, "single spike down"),
    # Case 1.2: edge cases start/end point
    (12, [20] + [5]*5, "wind_speed1_ms_avg", 1, "start spike down"),
    (12, [5]*5 + [20], "outside_temp_c_avg", 1, "end spike up"),

    # Case 2:
    # true isolated spike -> jump up + jump down => 1 flagged data point in between
    (21, [5]*5 + [20] + [5]*5, "wind_speed1_ms_avg", 1, "isolated spike up"),
    (22, [5]*5 + [-10] + [5]*5, "outside_temp_c_avg", 1, "isolated spike down"),
    #  spike below threshold -> no flags
    (23, [5]*5 + [10] + [5]*5, "wind_speed1_ms_avg", 0, "isolated spike up below threshold"),
    (24, [5]*5 + [8] + [5]*5, "outside_temp_c_avg", 0, "isolated spike up below threshold"),
    (25, [5, 49, 5, 5, 5], "wind_speed1_ms_avg", 1, "single spike up"),

    # Case 3:
    # less than 2 hours or less of interior points between strong gradient changes
    # short interior -> code fills entire interval between up and down (inclusive up to 12 datapoints)
    (31, [5]*5 + [20]*5 + [5]*5, "wind_speed1_ms_avg", 5, "short interior"),
    (32, [0]*5 + [20, 20] + [0]*5, "wind_speed1_ms_avg", 2, "short interior minimal case"),
    (33, [5]*5 + [-10]*5 + [5]*5, "outside_temp_c_avg", 5, "short interior"),
    # long interior with non zero gradients-> only flags spikes +/- one datapoint, not entire interval => 2 flagged rows
    (34, [5]*5 + [20]*4 + [21]*4 + [16]*4 + [17]*4 + [3]*5, "wind_speed1_ms_avg", 2, "long interior with non zero gradients"),
    (35, [5]*5 + [-10]*4 + [-11]*4  + [-7]*4 + [-6]*4 + [5]*5, "outside_temp_c_avg", 2, "long interior with non zero gradients"),

    # Case 4:
    # step up/down plateau with only zero gradient between strong gradient changes
    # step up plateau -> interval covers up..down inclusive => 13 flagged rows
    (41, [5]*4 + [20]*13 + [5]*4, "wind_speed1_ms_avg", 13, "step up plateau"),
    (42, [15]*5 + [0]*13 + [15]*5, "wind_speed1_ms_avg", 13, "step down plateau"),
    (43, [20]*5 + [0]*13 + [20]*5, "outside_temp_c_avg", 13, "step down plateau"),

    # Test combinations of multiple cases
    (51, [5]*5 + [20] + [5]*5 + [6]*5 + [7]*5 + [-10] + [5]*5, "wind_speed1_ms_avg", 2, "combination single spikes"),
    (52, [5]*5 + [20]*5 + [5]*5 + [6]*5 + [7]*5 + [-10]*5 + [5]*5, "outside_temp_c_avg", 10, "combination short interiors"),
    (53, [5]*5 + [20]*13 + [5]*5 + [6]*5 + [7]*5 + [20]*7 + [5]*5, "wind_speed1_ms_avg", 20, "combination step plateau + short interior"),
    (54, [5]*5 + [20]*13 + [5]*5 + [6]*5 + [7]*5 + [30] + [5]*5, "wind_speed1_ms_avg", 14, "combination step plateau + spike"),
    (55, [5]*5 + [20] + [25]*7 + [30]*5 + [5] + [30]*2, "wind_speed1_ms_avg", 2, "combination single spike up + spike"),
    (56, [5]*5 + [20] + [5]*5 + [6]*7 + [5] + [30]*5, "wind_speed1_ms_avg", 2, "combination single spike up + spike"),

    (57, [50]*5 + [50+mb_threshold+1] + [50+mb_threshold*2]*1 + [55]*5 , "main_bearing_temp2_c_avg", 2, "combination step + spike"),
    (58, [50]*5 + [50+mb_threshold+1] + [50+mb_threshold+2]*3 + [50]*5 , "main_bearing_temp2_c_avg", 4, "combination step + short interior"),
    (59, [50]*5 + [50+mb_threshold+1] + [50+mb_threshold+2]*3 + [40]*1 + [50]*5 , "main_bearing_temp2_c_avg", 5, "combination step + short interior"),

    (60, [50]*5 + [30]*2 + [50]*1 + [50]*2 + [30]*2 + [50]*1 + [51]*5 , "main_bearing_temp2_c_avg", 7, "combination step + short interior"),
    (61, [50]*5 + [30]*2 + [50]*5 + [51]*5 + [52]*5 + [30]*2 + [50]*1 + [51]*5 , "main_bearing_temp2_c_avg", 4, "combination step + long interior"),


    # Drift and flatline tests from above:
    (2, [50]*5 + np.arange(50, 55, 1).tolist() + [54]*5 , "main_bearing_temp2_c_avg", 0, "Short drift up - no flags"),
    (3, [50]*5 + np.arange(50, 70, 0.005).tolist() + [70]*5 , "main_bearing_temp2_c_avg", 4000, "Long drift up - flags"),
    (4, [50]*(FLATLINE_LIMIT+1), "main_bearing_temp2_c_avg", FLATLINE_LIMIT, "Flatline"),
    (5, [50]*(FLATLINE_LIMIT) + [51], "main_bearing_temp2_c_avg", 0, "Short flatline - no flags"),
    (6, [50]*5 + [70]*(FLATLINE_LIMIT)+ [50]*5, "main_bearing_temp2_c_avg", FLATLINE_LIMIT, " Spiked Flatline between normal values"),
    ("6a", [50]*3 + [51]*2 + [70]*(FLATLINE_LIMIT)+ [50]*3 + [51]*2, "main_bearing_temp2_c_avg", FLATLINE_LIMIT, "Sensor fault in middle of normal values"),
    (7, [50]*5 + [70]* (FLATLINE_LIMIT) + [50]*5, "main_bearing_temp2_c_avg", FLATLINE_LIMIT, "Sensor fault in middle of normal values"),
    ("7a", [50]*3 + [51] + [50] + [70]* (FLATLINE_LIMIT) + [50] + [51] + [52]*5, "main_bearing_temp2_c_avg", FLATLINE_LIMIT, "Sensor fault in the middle of normal values"),
    (8, [50]*(FLATLINE_LIMIT-5) + np.arange(50, 70, 0.005).tolist() + [70]*(FLATLINE_LIMIT-5), "main_bearing_temp2_c_avg", 4000, " short flatline + long drift up + short latline"),
    (8, [50]*(FLATLINE_LIMIT+5) + np.arange(50, 70, 0.005).tolist() + [70]*(FLATLINE_LIMIT+5), "main_bearing_temp2_c_avg", 4000+2*(FLATLINE_LIMIT+5)-1, " long flatline + long drift up + long latline"),

    # Flatline or drift interrupted by spikes -> flagging in the integrated method !
    (70, [50]*6 + [70] + [50]*6, "main_bearing_temp2_c_avg", 1, "Short Flatline + spike up"),
    (71, [50]*200 + [70] + [50]*200, "main_bearing_temp2_c_avg", 399, "Flatline + spike up"),
    (72, np.arange(50, 60, 0.01).tolist() + [40] + np.arange(60, 65, 0.10).tolist() ,"main_bearing_temp2_c_avg", 1050, "One long drift up, interupted by Spike down"),
    (73, np.arange(50, 60, 0.025).tolist() + [40] + np.arange(60, 70, 0.025).tolist() ,"main_bearing_temp2_c_avg", 800, "Drift up + Spike down"),
]

@pytest.mark.parametrize("ds_id, values, col, expected_count, case", INTEGRATED_OUTLIER_FLAGGING_TEST_CASES)
def test_integrated_outlier_flagging(ds_id, values, col, expected_count, case):
    group = make_group(values, dataset_id=str(ds_id), col=col)
    out = flag_outliers(group, verbose=True)
    assert flagged_count(out, flag_col="is_outlier") == expected_count, f"{case}: expected {expected_count}, got {flagged_count(out, flag_col='is_outlier')}"
    if expected_count > 0:
        assert flagged_cols_contains(out, col, flag_col="is_outlier"), f"{case}: expected column name in is_outlier_cols"