"""
Centralized threshold definitions for the preprocessing steps.
Tests and the preprocessor import the thresholds from central definition. 

Notes:
21.01.26: The defined thresholds are based on best guesses from me and visual data inspection. 
TODO: Discuss with domain expert from Alterric.
"""
# These thresholds are used in the preprocessing module to identify outliers in SCADA data.
# The short interior threshold is used to identify short interiors between two strong gradient changes.
# E.G if two stron gradient changes occur, with less thatn SHORT_INTERIOR_THRESHOLD samples in between,
# the interior samples are also marked as outliers.
SHORT_INTERIOR_THRESHOLD = 6  # 1 hour

# Unfortunatly, NaN values in the SCADA data were replaced with zeros in windfarm B + C.
# (see _handle_missing_values in preprocessing.py)
# To identify these missing value replacements, we define a maximum allowed number of consecutive zeros.
# That number is applied for rarly zero sensors.
MAX_CONSECUTIVE_ZEROS = 60  # 6 hours
RARLEY_ZERO_SENSORS = ["outside_temp_c_avg", "main_bearing_temp1_c_avg",
                    "main_bearing_temp2_c_avg", "main_bearing_temp3_c_avg"]


# Tolerance for considering a gradient as flatline:
GRADIENT_TOL = 0.01

# Columns in which outlier rows are not dropped, only flagged if outliers handling is set to 'drop'
COLS_TO_IGNORE_ON_DROPPING_OUTLIER_ROWS = [
    "wind_speed1_ms_avg", "wind_speed2_ms_avg", "wind_speed3_ms_avg",
    "main_bearing_temp1_c_avg", "main_bearing_temp2_c_avg", "main_bearing_temp3_c_avg"
]


# Absolute value thresholds 
# in units of the sensor measurements
ABSOLUTE_THRESHOLDS = {
    'outside_temp_c_avg': (-50, 50),
    'rotor_speed_rpm_avg': (-0.05, 30),  # allow small negative values as I observed them in real data
    'active_power_kWh': (-100, 5000),
    'wind_speed1_ms_avg': (0, 50), 
    'wind_speed2_ms_avg': (0, 50),
    'wind_speed3_ms_avg': (0, 50),
    'main_bearing_temp1_c_avg': (0, 80),
    'main_bearing_temp2_c_avg': (0, 80),
    'main_bearing_temp3_c_avg': (0, 80),
}

# Absolute gradient thresholds (change per 10min SCADA interval)
# in units of the sensor measurements
GRADIENT_THRESHOLDS = {
    'outside_temp_c_avg': 8,
    'rotor_speed_rpm_avg': 20,
    'active_power_kWh': 2000,  # TODO: Check
    'wind_speed1_ms_avg': 12,
    'wind_speed2_ms_avg': 12,
    'wind_speed3_ms_avg': 12,
    'main_bearing_temp1_c_avg': 6,
    'main_bearing_temp2_c_avg': 6,
    'main_bearing_temp3_c_avg': 6,
}

# Flatline thresholds (max allowed number of consecutive sample with the very same value)
# in number of samples (10min SCADA intervals)
FLATLINE_THRESHOLDS = {
    'outside_temp_c_avg': 144,        # max 24 hours of flatline
    'rotor_speed_rpm_avg': 144*7,     # max one week of flatline
    'active_power_kWh': 144*7,        # max one week of flatline
    'wind_speed1_ms_avg': 36,
    'wind_speed2_ms_avg': 36,
    'wind_speed3_ms_avg': 36,
    'main_bearing_temp1_c_avg': 144,
    'main_bearing_temp2_c_avg': 144,
    'main_bearing_temp3_c_avg': 144,
}

# Drift thresholds (max allowed number of consecutive samples with same sign gradient)
# in number of samples (10min SCADA intervals)
DRIFT_THRESHOLDS = {
    'outside_temp_c_avg': 72,  # max 12 hours of drift
    'rotor_speed_rpm_avg': 72,  # max 12 hours of drift
    'active_power_kWh': 72,     # max 12 hours of drift
    'wind_speed1_ms_avg': 72,
    'wind_speed2_ms_avg': 72,
    'wind_speed3_ms_avg': 72,
    'main_bearing_temp1_c_avg': 144*4, # max 4 days of drift
    'main_bearing_temp2_c_avg': 144*4,
    'main_bearing_temp3_c_avg': 144*4,
}


__all__ = [
    'ABSOLUTE_THRESHOLDS',
    'GRADIENT_THRESHOLDS',
    'DRIFT_THRESHOLDS',
    'FLATLINE_THRESHOLDS',
]
