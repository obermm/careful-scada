"""
Test the status removal, numeric col setter, uniqueness & completeness checks  and their
interplay as applied in the preprocessing pipeline.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add the parent directory to the path so we can import care_scada
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from care_scada.preprocessing import Preprocessor
from care_scada.preprocessing.data_quality import apply_uniqueness, apply_completeness


####################################################################################################
# Test remove status ###############################################################################
####################################################################################################

def _remove_status_helper(df, remove_status):
    """Helper function to test _remove_status from Preprocessor."""
    p = Preprocessor(remove_status=remove_status, verbose=False)
    return p._remove_status(df)

df_normal = pd.DataFrame({
    "dataset_id": ["1", "1", "1", "1", "1"],
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:20",
                                  "2022-01-01 00:30", "2022-01-01 00:40"]),
    "wind_speed1_ms_avg": [5, 10, 15, 20, 25],
    "status_type_id": [0, 0, 0, 0,0],

}).set_index(["dataset_id", "time_stamp"])

df_2 = pd.DataFrame({
    "dataset_id": ["1", "1", "1", "1", "1"],
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:20",
                                  "2022-01-01 00:30", "2022-01-01 00:40"]),
    "wind_speed1_ms_avg": [5, 10, 15, 20, 25],
    "status_type_id": [0, 2, 0, 3, 0],
}).set_index(["dataset_id", "time_stamp"])

df_2_removed = pd.DataFrame({
    "dataset_id": ["1",  "1", "1", "1"],
    "time_stamp": pd.to_datetime(["2022-01-01 00:00",  "2022-01-01 00:20",
                                  "2022-01-01 00:30", "2022-01-01 00:40"]),
    "wind_speed1_ms_avg": [5, 15, 20, 25],
    "status_type_id": [0, 0, 3, 0],
}).set_index(["dataset_id", "time_stamp"])

df_23_removed = pd.DataFrame({
    "dataset_id": ["1",  "1", "1"],
    "time_stamp": pd.to_datetime(["2022-01-01 00:00",  "2022-01-01 00:20",
                                "2022-01-01 00:40"]),
    "wind_speed1_ms_avg": [5, 15, 25],
    "status_type_id": [0, 0, 0],
}).set_index(["dataset_id", "time_stamp"])


REMOVE_STATUS_TEST_CASES = [
    (df_normal, 6, df_normal),
    (df_2, 2, df_2_removed),
    (df_2, 5, df_2),
    (df_2, [2, 3], df_23_removed),
]

@pytest.mark.parametrize("input_df, remove_status, expected_df", REMOVE_STATUS_TEST_CASES)
def test_remove_status(input_df, remove_status, expected_df):
    """ 
    Test the removal of specified status codes from the DataFrame.
    :param input_df: Input DataFrame with status codes.
    :type input_df: pd.DataFrame
    :param remove_status: Status codes to remove.
    :type remove_status: int or list of int
    :param expected_df: Expected DataFrame after status removal.
    :type expected_df: pd.DataFrame
    """
    out_df = _remove_status_helper(input_df, remove_status)
    pd.testing.assert_frame_equal(out_df, expected_df)


####################################################################################################
# Test set numeric colum ########################################################################
####################################################################################################
df_complete = pd.DataFrame({
    "dataset_id": ["1"] * 6,
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:20",
                                  "2022-01-01 00:30", "2022-01-01 00:40", "2022-01-01 00:50"]),
    "wind_speed1_ms_avg": [5, 10, 15, 20, 25, 30]
}).set_index(["dataset_id", "time_stamp"])

df_numeric_cols = pd.DataFrame({
    "dataset_id": ["1"] * 6,
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:20",
                                  "2022-01-01 00:30", "2022-01-01 00:40", "2022-01-01 00:50"]),
    "wind_speed1_ms_avg": [5, 10, 15, 20, 25, 30],
    "wind_speed2_ms_avg": [5, 10, 15, 20, 25, 30],
    "non_numeric_column": ["5", 10, 15, 20, 25, 30],
}).set_index(["dataset_id", "time_stamp"])

NUMERIC_TEST_CASES = [
    (df_complete, ["wind_speed1_ms_avg"]),
    (df_numeric_cols, ["wind_speed1_ms_avg", "wind_speed2_ms_avg"]),

]
@pytest.mark.parametrize("input_df, expected_numeric_cols", NUMERIC_TEST_CASES)
def test_numeric_cols(input_df, expected_numeric_cols):
    """
    Test the setting of numeric columns in the Preprocessor.
    :param input_df: Input DataFrame to determine numeric columns from.
    :type input_df: pd.DataFrame
    :param expected_numeric_cols: Expected list of numeric columns.
    :type expected_numeric_cols: list
    """
    preprocessor = Preprocessor()
    preprocessor._set_numeric_cols(input_df)
    numeric_cols = preprocessor.numeric_cols
    assert numeric_cols == expected_numeric_cols



####################################################################################################
# Test Uniqueness Check ############################################################################
####################################################################################################

df_unique = pd.DataFrame({
    "dataset_id": ["1", "1", "1", "1", "1"],
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:20",
                                  "2022-01-01 00:30", "2022-01-01 00:40"]),
    "wind_speed1_ms_avg": [5, 10, 15, 20, 25]
}).set_index(["dataset_id", "time_stamp"])

df_duplicate = pd.DataFrame({
    "dataset_id": ["1", "1", "1", "1", "1"],
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:10",
                                  "2022-01-01 00:20", "2022-01-01 00:30"]),
    "wind_speed1_ms_avg": [5, 10, 10, 20, 25]
}).set_index(["dataset_id", "time_stamp"])

df_duplicate_removed = pd.DataFrame({
    "dataset_id": ["1", "1", "1"],
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:20", "2022-01-01 00:30"]),
    "wind_speed1_ms_avg": [5, 20, 25]
}).set_index(["dataset_id", "time_stamp"])

df_triplicate = pd.DataFrame({
    "dataset_id": ["1", "1", "1", "1", "1", "1"],
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:10",
                                  "2022-01-01 00:10", "2022-01-01 00:20", "2022-01-01 00:30"]),
    "wind_speed1_ms_avg": [5, 10, 10, 10, 20, 25]
}).set_index(["dataset_id", "time_stamp"])

feb_28 = pd.to_datetime("2019-02-28" + " 00:00:00") + pd.to_timedelta(np.arange(0, 24*6)*10,\
                         unit='min')

df_duplicated_feb28 = pd.DataFrame({
    "dataset_id": ["1"] * (len(feb_28) * 2),
    "time_stamp": list(feb_28) + list(feb_28),
    "wind_speed1_ms_avg": [10] * (len(feb_28) * 2)
}).set_index(["dataset_id", "time_stamp"])

df_duplicated_feb28_removed = pd.DataFrame({
    "dataset_id": ["1"] * len(feb_28),
    "time_stamp": list(feb_28),
    "wind_speed1_ms_avg": [10] * len(feb_28)
}).set_index(["dataset_id", "time_stamp"])

feb = pd.to_datetime("2024-02-01" + " 00:00:00") + pd.to_timedelta(np.arange(0, 29*24*6)*10, \
                    unit='min')
df_no_duplicates = pd.DataFrame({
    "dataset_id": ["1"] * len(feb),
    "time_stamp": list(feb),
    "wind_speed1_ms_avg": [15] * len(feb)
}).set_index(["dataset_id", "time_stamp"])

UNIQUENESS_TEST_CASES = [
    # Case 1: Unique entries
    (df_unique, df_unique),
    (df_no_duplicates, df_no_duplicates),
    # Case 2: Duplicate entries
    (df_duplicate, df_duplicate_removed),
    # Case 3: Duplicates on Feb 28th
    (df_duplicated_feb28, df_duplicated_feb28_removed),
    # Case 4: triplicates
    (df_triplicate, df_duplicate_removed)

]

@pytest.mark.parametrize("input_df, expected_df", UNIQUENESS_TEST_CASES)
def test_uniqueness_check(input_df, expected_df):
    """
    Test the uniqueness check and removal of duplicate timestamps.
    
    :param input_df: Input DataFrame to check for uniqueness with potential duplicate timestamps.
    :type input_df: pd.DataFrame
    :param expected_df: Expected DataFrame after removing duplicates.
    :type expected_df: pd.DataFrame
    """
    out_df = apply_uniqueness(input_df, verbose=False)
    pd.testing.assert_frame_equal(out_df, expected_df)



####################################################################################################
# Test Completeness Check ##########################################################################
####################################################################################################

df_incomplete = pd.DataFrame({
    "dataset_id": ["1"] * 4,
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10",
                                  "2022-01-01 00:30", "2022-01-01 00:50"]),
    "wind_speed1_ms_avg": [5, 10, 20, 30]
}).set_index(["dataset_id", "time_stamp"])

df_incomplete_filled = pd.DataFrame({
    "dataset_id": ["1"] * 6,
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:20",
                                  "2022-01-01 00:30", "2022-01-01 00:40", "2022-01-01 00:50"]),
    "wind_speed1_ms_avg": [5, 10, np.nan, 20, np.nan, 30]
}).set_index(["dataset_id", "time_stamp"])

df_duplicate_removed_filled = pd.DataFrame({
    "dataset_id": ["1", "1", "1", "1"],
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:20",
                                  "2022-01-01 00:30"]),
    "wind_speed1_ms_avg": [5, np.nan, 20, 25]
}).set_index(["dataset_id", "time_stamp"])

df_no_duplicates = pd.DataFrame({
    "dataset_id": ["1"] * len(feb),
    "time_stamp": list(feb),
    "wind_speed1_ms_avg": [15] * len(feb)
}).set_index(["dataset_id", "time_stamp"])



COMPLETENESS_TEST_CASES = [
    # Case 1: Complete data
    (df_complete, df_complete),
     (df_no_duplicates, df_no_duplicates),
    # Case 2: Incomplete data
    (df_incomplete, df_incomplete_filled),
    (df_duplicate_removed, df_duplicate_removed_filled),

]

@pytest.mark.parametrize("input_df, expected_df", COMPLETENESS_TEST_CASES)
def test_completeness_check(input_df, expected_df):
    """
    Test the completeness check and filling of missing timestamps.
    
    :param input_df: Input DataFrame to check for completeness with potential missing timestamps.
    :type input_df: pd.DataFrame
    :param expected_df: Expected DataFrame after filling missing timestamps.
    :type expected_df: pd.DataFrame
    """
    out_df = apply_completeness(input_df, verbose=False)
    pd.testing.assert_frame_equal(out_df, expected_df)

####################################################################################################
# Test Filtering and Completeness Checks ###########################################################
####################################################################################################
df_2_removed_filled = pd.DataFrame({
    "dataset_id": ["1",  "1", "1", "1", "1"],
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10",  "2022-01-01 00:20",
                                  "2022-01-01 00:30", "2022-01-01 00:40"]),
    "wind_speed1_ms_avg": [5, np.nan, 15, 20, 25],
    "status_type_id": [0, np.nan, 0, 3, 0],
}).set_index(["dataset_id", "time_stamp"])

df_23_removed_filled = pd.DataFrame({
    "dataset_id": ["1",  "1", "1", "1", "1"],
    "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10",  "2022-01-01 00:20",
                                  "2022-01-01 00:30", "2022-01-01 00:40"]),
    "wind_speed1_ms_avg": [5, np.nan, 15, np.nan, 25],
    "status_type_id": [0, np.nan, 0, np.nan, 0],
}).set_index(["dataset_id", "time_stamp"])

STATUS_COMPL_TEST_CASES = [
    (df_normal, 6, df_normal),
    (df_2, 2, df_2_removed_filled),
    (df_2, 5, df_2),
    (df_2, [2, 3], df_23_removed_filled),
]

@pytest.mark.parametrize("input_df, remove_status, expected_df", STATUS_COMPL_TEST_CASES)
def test_remove_complete(input_df, remove_status, expected_df):
    """
    Test the interplay of status removal and completeness check.
    :param input_df: Input DataFrame with status codes.
    :type input_df: pd.DataFrame
    :param remove_status: Status codes to remove.
    :type remove_status: int or list of int
    :param expected_df: Expected DataFrame after status removal and completeness check.
    :type expected_df: pd.DataFrame
    """
    filtered_df = _remove_status_helper(input_df, remove_status)
    out_df = apply_completeness(filtered_df, verbose=False)
    pd.testing.assert_frame_equal(out_df, expected_df)

####################################################################################################
# Test Uniqueness and Completeness Checks ##########################################################
####################################################################################################
UNIQ_COMPL_TEST_CASES = [
    # Case 1: Complete data
    (df_complete, df_complete),
    (df_no_duplicates, df_no_duplicates),

    # Case 1: Duplicate data
    (df_duplicate, df_duplicate_removed_filled),
    (df_duplicated_feb28, df_duplicated_feb28_removed),

]
@pytest.mark.parametrize("input_df, expected_df", UNIQ_COMPL_TEST_CASES)
def test_both_checks(input_df, expected_df):
    """
    Test the combination of uniqueness and completeness checks.
    
    :param input_df: Input DataFrame with potential duplicates and missing timestamps.
    :type input_df: pd.DataFrame
    :param expected_df: Expected DataFrame after applying uniqueness and completeness checks.
    :type expected_df: pd.DataFrame
    """
    out_df_step1 = apply_uniqueness(input_df, verbose=False)
    out_df_step2 = apply_completeness(out_df_step1, verbose=False)
    pd.testing.assert_frame_equal(out_df_step2, expected_df)
