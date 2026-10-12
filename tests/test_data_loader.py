"""
Tests for DataLoader with SensorConfig integration.
"""

import sys
import os
import pytest
import pandas as pd

# Add the parent directory to the path so we can import care_scada
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from care_scada.data_loader import DataLoader
from care_scada.sensor_config import SensorConfig


def test_dataloader_init_valid_farm():
    """Test DataLoader initialization with valid farm names."""
    for farm in ['A', 'B', 'C']:
        loader = DataLoader(farm)
        assert loader.farm_name == farm
        assert isinstance(loader.sensor_config, SensorConfig)


def test_dataloader_init_invalid_farm():
    """Test DataLoader initialization with invalid farm name."""
    with pytest.raises(ValueError, match="Invalid farm_name"):
        DataLoader('D')

def test_dataloader_gets_sensor_config():
    """Test that DataLoader has access to SensorConfig."""
    loader = DataLoader('A')

    # Check that we can get rename maps
    rename_map = loader.sensor_config.get_rename_map('A')
    assert len(rename_map) > 0

    # Check that we can get unit conversions
    conversions = loader.sensor_config.get_unit_conversions('A')
    assert isinstance(conversions, dict)


def test_dataloader_uses_sensor_config_in_unify_units():
    """Test that unify_units uses sensor config conversions."""
    loader = DataLoader('A')

    # Create a mock dataframe with raw power column and MultiIndex
    df = pd.DataFrame({
        "dataset_id": ["1", "1", "1", "1", "1"],
        "time_stamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:20",
                                    "2022-01-01 00:30", "2022-01-01 00:40"]),
        "sensor_50": [1000, 2000, 1500, 2000, 2500],

    }).set_index(["dataset_id", "time_stamp"])

    # Apply unify_units
    unified = loader.unify_units(df)

    # Check that the power column has been converted from Wh to kWh
    expected_power = [1.0, 2.0, 1.5, 2.0, 2.5]
    assert all(unified['sensor_50'] == expected_power)


def test_dataloader_uses_sensor_config_in_standardize_names():
    """Test that standardize_column_names uses sensor config."""
    loader = DataLoader('A')

    # Create a mock dataframe with raw column names
    df = pd.DataFrame({
        'sensor_0_avg': [10, 20, 30],
        'sensor_52_avg': [100, 110, 120],
        'wind_speed_3_avg': [5, 10, 15]
    })
    df.index.name = 'time_stamp'

    # Standardize names
    renamed = loader.standardize_column_names(df)

    # Check that columns have been renamed correctly
    df_new_cols = ['outside_temp_c_avg', 'rotor_speed_rpm_avg', 'wind_speed1_ms_avg']
    assert renamed.columns.tolist() == df_new_cols


def test_dataloader_path_construction():
    """Test that DataLoader constructs correct paths."""
    for farm in ['A', 'B', 'C']:
        loader = DataLoader(farm)

        # Check path contains Wind Farm and dataset folder
        assert f'Wind Farm {farm}' in loader.path
        assert 'datasets' in loader.path
        assert loader.path.endswith('datasets')
