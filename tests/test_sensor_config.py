"""
Tests for SensorConfig class and sensor configuration management.
"""

import sys
import os
import pytest

# Add the parent directory to the path so we can import care_scada
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from care_scada.sensor_config import SensorConfig, get_sensor_config

@pytest.fixture
def config():
    """Create a SensorConfig instance."""
    return SensorConfig()

def test_get_rename_map_farm_a(config):
    """Test rename map generation for Farm A."""
    rename_map = config.get_rename_map('A')

    # Check that expected mappings exist
    assert 'sensor_0_avg' in rename_map
    assert rename_map['sensor_0_avg'] == 'outside_temp_c_avg'
    assert rename_map['sensor_52_avg'] == 'rotor_speed_rpm_avg'
    assert rename_map['sensor_50'] == 'active_power_kWh'


def test_get_rename_map_farm_b(config):
    """Test rename map generation for Farm B."""
    rename_map = config.get_rename_map('B')

    assert 'sensor_8_avg' in rename_map
    assert rename_map['sensor_8_avg'] == 'outside_temp_c_avg'
    assert 'sensor_51_avg' in rename_map
    assert rename_map['sensor_51_avg'] == 'main_bearing_temp1_c_avg'


def test_get_rename_map_farm_c(config):
    """Test rename map generation for Farm C."""
    rename_map = config.get_rename_map('C')

    assert 'sensor_177_avg' in rename_map
    assert rename_map['sensor_177_avg'] == 'outside_temp_c_avg'
    assert 'sensor_8_avg' in rename_map
    assert rename_map['sensor_8_avg'] == 'gen_rpm_avg'


def test_get_unit_conversions_farm_a(config):
    """Test unit conversion lookup for Farm A."""
    conversions = config.get_unit_conversions('A')

    # Farm A has Wh to kWh conversion for active_power_kWh
    assert 'active_power_kWh' in conversions
    # Test conversion function
    assert conversions['active_power_kWh'](1000) == 1.0


def test_get_unit_conversions_farm_c(config):
    """Test unit conversion lookup for Farm C."""
    conversions = config.get_unit_conversions('C')

    # Farm C has rad/s to RPM conversion for gen_rpm_avg
    assert 'gen_rpm_avg' in conversions


def test_get_thresholds_absolute(config):
    """Test absolute threshold retrieval."""
    thresholds = config.get_thresholds('absolute')

    assert 'outside_temp_c_avg' in thresholds
    assert thresholds['outside_temp_c_avg'] == [-50, 50]
    assert thresholds['active_power_kWh'] == [-100, 5000]


def test_get_thresholds_gradient(config):
    """Test gradient threshold retrieval."""
    thresholds = config.get_thresholds('gradient')

    assert 'wind_speed1_ms_avg' in thresholds
    assert thresholds['wind_speed1_ms_avg'] == 12
    assert thresholds['main_bearing_temp1_c_avg'] == 6

def test_get_thresholds_flatline(config):
    """Test flatline threshold retrieval."""
    thresholds = config.get_thresholds('flatline')

    assert 'wind_speed1_ms_avg' in thresholds
    assert thresholds['wind_speed1_ms_avg'] == 144
    assert thresholds['outside_temp_c_avg'] == 144


def test_get_thresholds_drift(config):
    """Test drift threshold retrieval."""
    thresholds = config.get_thresholds('drift')

    assert 'outside_temp_c_avg' in thresholds
    assert thresholds['outside_temp_c_avg'] == 72
    assert thresholds['main_bearing_temp1_c_avg'] == 576  # 4 days


def test_get_rarely_zero_sensors(config):
    """Test rarely zero sensors retrieval."""
    rarely_zero = config.get_rarely_zero_sensors()

    assert 'outside_temp_c_avg' in rarely_zero
    assert 'main_bearing_temp1_c_avg' in rarely_zero
    assert 'gearbox_oil_temp1_c_avg' in rarely_zero
    assert 'wind_speed1_ms_avg' not in rarely_zero


def test_get_setting(config):
    """Test global setting retrieval."""
    short_interior = config.get_setting('short_interior_threshold')
    assert short_interior is not None

    max_zeros = config.get_setting('max_consecutive_zeros')
    assert max_zeros is not None

    gradient_tol = config.get_setting('gradient_tolerance')
    assert gradient_tol is not None


def test_get_setting_with_default(config):
    """Test setting retrieval with default value."""
    nonexistent = config.get_setting('nonexistent_setting', 'default_value')
    assert nonexistent == 'default_value'


def test_get_all_allowed_columns_farm_a(config):
    """Test getting all allowed columns for Farm A."""
    allowed = config.get_all_allowed_columns('A')

    assert 'sensor_0_avg' in allowed
    assert 'sensor_50' in allowed
    assert 'time_stamp' in allowed
    assert 'asset_id' in allowed

def test_singleton_pattern():
    """Test that get_sensor_config returns the same instance."""
    config1 = get_sensor_config()
    config2 = get_sensor_config()

    assert config1 is config2


def test_metadata_mappings(config):
    """Test that metadata columns are included in rename map."""
    rename_map = config.get_rename_map('A')

    # Metadata should be included
    assert 'time_stamp' in rename_map
    assert 'asset_id' in rename_map
    assert 'id' in rename_map
    assert 'train_test' in rename_map
    assert 'status_type_id' in rename_map


def test_all_thresholds_have_sensors(config):
    """Test that all threshold entries correspond to sensors."""
    absolute = config.get_thresholds('absolute')
    gradient = config.get_thresholds('gradient')
    flatline = config.get_thresholds('flatline')
    drift = config.get_thresholds('drift')

    # All should be non-empty
    assert len(absolute) > 0
    assert len(gradient) > 0
    assert len(flatline) > 0
    assert len(drift) > 0

    # Check overlap
    sensors_with_absolute = set(absolute.keys())
    sensors_with_gradient = set(gradient.keys())
    sensors_with_flatline = set(flatline.keys())
    sensors_with_drift = set(drift.keys())

    # All thresholds should have the same sensor set
    all_threshold_sensors = (
        sensors_with_absolute & sensors_with_gradient & 
        sensors_with_flatline & sensors_with_drift
    )

    assert len(all_threshold_sensors) > 0
