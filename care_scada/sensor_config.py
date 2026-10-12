"""Sensor configuration management for th eCARE SCADA datasets.
Defintions of sensor mappings, unit conversions, and validation threshold are found in 
sensor_config.yaml and managed through this module."""

import math
import os
import yaml


def rads_to_rpm(rads):
    """Converts radians per second to revolutions per minute (RPM)."""
    return rads * 60 / (2 * math.pi)


def wh_to_kwh(wh):
    """Converts Wh to kWh."""
    return wh / 1000


class SensorConfig:
    """
    Centralized sensor configuration for CARE SCADA datasets.
    Manages farm-specific mappings, unit conversions, and validation thresholds defined in
    the sensor_config.yaml.
    """

    # Unit conversion functions registry
    CONVERSION_FUNCTIONS = {
        'Wh_to_kWh': wh_to_kwh,
        'rad_to_rpm': rads_to_rpm,
    }


    def __init__(self, config_path=None):
        """
        Initialize the SensorConfig.
        
        Args:
            config_path (str): Path to sensor_config.yaml. If None, uses default location.
        """
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "sensor_config.yaml")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sensors = self.config.get('sensors', {})
        self.metadata = self.config.get('metadata', {})
        self.settings = self.config.get('settings', {})


    def get_rename_map(self, farm_name):
        """
        Extract rename map for specific farm (both sensors and metadata).
        E.g 'sensor_0_avg' -> 'outside_temp_c_avg'
        
        Args:
            farm_name (str): Farm identifier ('A', 'B', or 'C')
            
        Returns:
            dict: Mapping from raw column names to standardized names
        """
        rename_dict = {}

        # Add sensor mappings
        for std_name, sensor in self.sensors.items():
            if 'farm_mappings' in sensor and farm_name in sensor['farm_mappings']:
                for raw_name in sensor['farm_mappings'][farm_name]:
                    rename_dict[raw_name] = std_name

        # Add metadata mappings
        for std_name, meta in self.metadata.items():
            if 'farm_mappings' in meta and farm_name in meta['farm_mappings']:
                for raw_name in meta['farm_mappings'][farm_name]:
                    rename_dict[raw_name] = std_name

        return rename_dict


    def get_unit_conversions(self, farm_name):
        """
        Get unit conversion functions for specific farm.
        
        Args:
            farm_name (str): Farm identifier ('A', 'B', or 'C')
            
        Returns:
            dict: Mapping from standardized column names to conversion functions
        """
        conversions = {}
        for std_name, sensor in self.sensors.items():
            if 'unit_conversions' in sensor and farm_name in sensor['unit_conversions']:
                conversion_name = sensor['unit_conversions'][farm_name]
                if conversion_name in self.CONVERSION_FUNCTIONS:
                    conversions[std_name] = self.CONVERSION_FUNCTIONS[conversion_name]
        return conversions


    def get_thresholds(self, threshold_type):
        """
        Extract specific threshold type for all sensors.
        
        Args:
            threshold_type (str): Type of threshold ('absolute', 'gradient', 'flatline', 'drift')
            
        Returns:
            dict: Mapping from sensor name to threshold value
        """
        thresholds = {}
        for name, sensor in self.sensors.items():
            if 'thresholds' in sensor and threshold_type in sensor['thresholds']:
                thresholds[name] = sensor['thresholds'][threshold_type]
        return thresholds


    def get_rarely_zero_sensors(self):
        """
        Get list of sensors that rarely have zero values.
        
        Returns:
            list: List of sensor names that are rarely zero
        """
        return [
            name for name, sensor in self.sensors.items()
            if sensor.get('rarely_zero', False)
        ]


    def get_setting(self, setting_name, default=None):
        """
        Get a global setting value.
        
        Args:
            setting_name (str): Name of the setting
            default: Default value if setting not found
            
        Returns:
            The setting value or default
        """
        return self.settings.get(setting_name, default)


    def get_preprocessing_settings(self):
        """
        Get preprocessing configuration.
        
        Returns:
            dict: Dictionary with preprocessing settings:
                - missing_values_strategy: strategy for handling missing values
                - outlier_handling: strategy for handling outliers
                - remove_status: list of status IDs to remove
        """
        preprocessing = self.settings.get('preprocessing', {})
        return {
            'missing_values_strategy': preprocessing.get('missing_values_strategy', 'keep'),
            'outlier_handling': preprocessing.get('outlier_handling', 'flag_only'),
            'remove_status': preprocessing.get('remove_status', None)
        }


    def get_all_allowed_columns(self, farm_name):
        """
        Get all raw column names that should be loaded for a farm.
        
        Args:
            farm_name (str): Farm identifier ('A', 'B', or 'C')
            
        Returns:
            set: Set of raw column names
        """
        allowed = set()

        # Add sensor columns
        for sensor in self.sensors.values():
            if 'farm_mappings' in sensor and farm_name in sensor['farm_mappings']:
                allowed.update(sensor['farm_mappings'][farm_name])

        # Add metadata columns
        for meta in self.metadata.values():
            if 'farm_mappings' in meta and farm_name in meta['farm_mappings']:
                allowed.update(meta['farm_mappings'][farm_name])

        return allowed


# Singleton instance for easy access
_CONFIG_INSTANCE = None

def get_sensor_config():
    """Get or create the global SensorConfig instance."""
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = SensorConfig()
    return _CONFIG_INSTANCE
