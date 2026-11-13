"""
Tests for the handling of missing values.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from care_scada.preprocessing.missing_values import handle_missing_values, MAX_CONSECUTIVE_ZEROS

# Add the parent directory to the path so we can import care_scada
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

####################################################################################################
# Tests missing value detection for Windfarms B & C, where missing values have been set to zero by 
# the data providers.
####################################################################################################

df_normal = pd.DataFrame({
    "dataset_id": ["1", "1", "1", "1", "1"],
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:20",
                                  "2022-01-01 00:30", "2022-01-01 00:40"]),
    "outside_temp_c_avg": [10.0, 10.0, 11.0, 12.0, 12.0], #rarly zero
    "rotor_speed_rpm_avg": [20.0, 19.0, 21.0, 20.0, 22.0], #often zero
    "active_power_kWh": [100.0, 102.0, 101.0, 99.0, 98.0] #often zero
}).set_index(["dataset_id", "time_stamp"])

df_with_short_zero_run = pd.DataFrame({
    "dataset_id": ["1"] * 5,
    "time_stamp": pd.date_range(start="2022-01-01 00:00", periods=5, freq="10min"),
    "outside_temp_c_avg": [10.0, 0.0, 0.0, 12.0, 12.0], # rarely zero
    "rotor_speed_rpm_avg": [20.0, 0.0, 0.0, 20.0, 22.0], # often zero
    "active_power_kWh": [100.0, 0.0, 0.0, 99.0, 98.0] # often zero
}).set_index(["dataset_id", "time_stamp"])

# zero run length is tied to the configured threshold (max_consecutive_zeros)
N_ZEROS = MAX_CONSECUTIVE_ZEROS
df_with_long_zero_run = pd.DataFrame({
    "dataset_id": ["1"] * (N_ZEROS + 2),
    "time_stamp": pd.date_range(start="2022-01-01 00:00", periods=N_ZEROS + 2, freq="10min"),
    "outside_temp_c_avg": [10.0] + [0.0]*N_ZEROS + [12.0], # rarely zero
    "rotor_speed_rpm_avg": [20.0] + [0.0]*N_ZEROS + [22.0], # often zero
    "active_power_kWh": [100.0] + [0.0]*N_ZEROS + [98.0] # often zero
}).set_index(["dataset_id", "time_stamp"])

df_power_with_no_rotor_speed = pd.DataFrame({
    "dataset_id": ["1"] * 5,
    "time_stamp": pd.date_range(start="2022-01-01 00:00", periods=5, freq="10min"),
    "outside_temp_c_avg": [10.0, 11.0, 12.0, 12.0, 13.0], # rarely zero
    "rotor_speed_rpm_avg": [0.0, 0.0, 0.0, 0.0, 0.0], # often zero
    "active_power_kWh": [1000.0, 1002.0, 1010.0, 990.0, 908.0] # often zero
}).set_index(["dataset_id", "time_stamp"])

df_rotor_speed_with_no_power = pd.DataFrame({
    "dataset_id": ["1"] * 5,
    "time_stamp": pd.date_range(start="2022-01-01 00:00", periods=5, freq="10min"),
    "outside_temp_c_avg": [10.0, 11.0, 12.0, 12.0, 13.0], # rarely zero
    "rotor_speed_rpm_avg": [20.0, 19.0, 21.0, 20.0, 22.0], # often zero
    "active_power_kWh": [0.0, 0.0, 0.0, 0.0, 0.0] # often zero
}).set_index(["dataset_id", "time_stamp"])

df_power_with_no_wind_speed = pd.DataFrame({
    "dataset_id": ["1"] * 5,
    "time_stamp": pd.date_range(start="2022-01-01 00:00", periods=5, freq="10min"),
    "outside_temp_c_avg": [10.0, 11.0, 12.0, 12.0, 13.0], # rarely zero
    "wind_speed1_ms_avg": [0.0, 0.0, 0.0, 0.0, 0.0], # often zero
    "rotor_speed_rpm_avg": [20.0, 19.0, 21.0, 20.0, 22.0], # often zero
    "active_power_kWh": [1000.0, 1002.0, 1010.0, 990.0, 908.0] # often zero
}).set_index(["dataset_id", "time_stamp"])

DETECT_MISSING_VALUES_TEST_CASES = [
    # normal data without zeros
    (df_normal, df_normal),

    # rarely zero sensor with short zero run -> no removal
    (df_with_short_zero_run, df_with_short_zero_run),

    # rarely zero sensor with long zero run -> removal
    # rotor_speed and active_power are often zero sensors and cross validated, so zeros are kept
    (df_with_long_zero_run,
     df_with_long_zero_run.assign(outside_temp_c_avg=[10.0] + [np.nan]*N_ZEROS + [12.0])),

    # cross sensor validation: active_power > 100kw while rotor_speed is zero 
    # -> rotor_speed zeros are nans
    (df_power_with_no_rotor_speed,
     df_power_with_no_rotor_speed.assign(rotor_speed_rpm_avg=[np.nan]*5)),

     # cross sensor validation: active_power very low while rotor_speed is zero 
     # -> rotor_speed zeros are kept
    (df_power_with_no_rotor_speed.assign(active_power_kWh=[0.4, 0.5, 0.3, 0.3, 0.3]),
     df_power_with_no_rotor_speed.assign(active_power_kWh=[0.4, 0.5, 0.3, 0.3, 0.3])),

    # cross sensor validation: rotor_speed > 2 rpm while active_power is zero 
    # -> active_power zeros are nans
    (df_rotor_speed_with_no_power,
     df_rotor_speed_with_no_power.assign(active_power_kWh=[np.nan]*5)),

    # cross sensor validation: rotor_speed very low while active_power is zero 
    # -> active_power zeros are kept
    (df_rotor_speed_with_no_power.assign(rotor_speed_rpm_avg=[1.0, 1.5, 1.2, 1.0, 1.3]),
     df_rotor_speed_with_no_power.assign(rotor_speed_rpm_avg=[1.0, 1.5, 1.2, 1.0, 1.3])),

    # cross sensor validation: active_power > 100kw while wind_speed is zero 
    # -> wind_speed zeros are nans
    (df_power_with_no_wind_speed,
     df_power_with_no_wind_speed.assign(wind_speed1_ms_avg=[np.nan]*5)),

]

@pytest.mark.parametrize("input_df, expected_df", DETECT_MISSING_VALUES_TEST_CASES)
def test_detect_missing_values(input_df, expected_df):
    """
    Test the detection of missing values for Windfarms B, where missing values have been set to zero.       
    :param input_df: Input DataFrame with potential missing values represented as zeros.
    :type input_df: pd.DataFrame
    :param expected_df: Expected DataFrame after detecting zeros as missing values.
    :type expected_df: pd.DataFrame
    """
    numeric_cols = ["outside_temp_c_avg", "rotor_speed_rpm_avg", "active_power_kWh"]
    output_df = handle_missing_values(
        input_df,
        farm_name="B",
        numeric_cols=numeric_cols,
        strategy="keep",
        verbose=False
    )

    pd.testing.assert_frame_equal(output_df, expected_df)


####################################################################################################
# Test the different handling strategies
####################################################################################################

HANDLE_MISSING_VALUES_TEST_CASES = [
    # dropping
    (df_with_long_zero_run, "drop",
     df_with_long_zero_run.assign(outside_temp_c_avg=[10.0] + [np.nan]*N_ZEROS + [12.0]).dropna()),
    # linear interpolation
    (df_with_long_zero_run, "linear_interpolation",
     df_with_long_zero_run.assign(outside_temp_c_avg=[10.0] + \
                                  [10.0 + i*(12.0-10.0)/(N_ZEROS + 1) for i in range(1, N_ZEROS + 1)] + [12.0])),
    # median
    (df_with_long_zero_run, "median",
     df_with_long_zero_run.assign(outside_temp_c_avg=[10.0] + [11.0]*N_ZEROS + [12.0])),
]
@pytest.mark.parametrize("input_df, strategy, expected_df", HANDLE_MISSING_VALUES_TEST_CASES)
def test_handle_missing_values(input_df, strategy, expected_df):
    """
    Test the different missing value handling strategies.
    
    :param input_df: Input DataFrame with potential missing values (nans).
    :type input_df: pd.DataFrame
    :param strategy: The missing value handling strategy.
    :type strategy: str
    :param expected_df: Expected DataFrame after applying the missing value handling strategy.
    :type expected_df: pd.DataFrame
    """
    numeric_cols = ["outside_temp_c_avg", "rotor_speed_rpm_avg", "active_power_kWh"]
    output_df = handle_missing_values(
        input_df,
        farm_name="B",
        numeric_cols=numeric_cols,
        strategy=strategy,
        verbose=False
    )

    pd.testing.assert_frame_equal(output_df, expected_df)
