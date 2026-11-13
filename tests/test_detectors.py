"""Tests for the fault detectors (CUSUM alarms, residual and Kalman heat detectors)."""

import numpy as np
import pandas as pd
import pytest

from care_scada.detector_factory import (DetectorFactory, KalmanHeatDetector, ResidualDetector,
                                         cusum)

TARGET = "temp"


def _daily(values, start="2022-01-01"):
    return pd.DataFrame({TARGET: values}, index=pd.date_range(start, periods=len(values), freq="D"))


def test_cusum():
    inc = np.array([1.0, 1.0, np.nan, -5.0, 2.0, 3.0])
    np.testing.assert_allclose(cusum(inc, k=0.5), [0.5, 1.0, 1.0, 0.0, 1.5, 4.0])
    # reset after exceeding h; NaN keeps the sum
    np.testing.assert_allclose(cusum(inc, k=0.5, h=1.2), [0.5, 1.0, 1.0, 0.0, 1.5, 2.5])


def test_calibrate_and_detect():
    rng = np.random.default_rng(0)
    healthy = [_daily(rng.normal(0, 0.05, 200)) for _ in range(3)]
    values = rng.normal(0, 0.05, 200)
    values[10:] += 5.0                          # step inside the burn-in: ignored
    values[150:] += 0.1 * np.arange(50)         # fault: 0.1 K/day ramp from day 150
    fault = _daily(values)
    det = DetectorFactory.create("residual", monitored=[TARGET], burn_in_days=30)
    with pytest.raises(RuntimeError):
        det.detect(fault)
    thresholds = det.calibrate(healthy)
    assert set(thresholds) == {"rise", "level"} and all(h >= 0 for h in thresholds.values())
    assert all(det.detect(ind).empty for ind in healthy)
    alarms = det.detect(fault)
    assert list(alarms.columns) == ["day", "monitored", "kind"]
    assert fault.index[150] <= alarms["day"].iloc[0] <= fault.index[165]
    assert set(alarms["kind"]) <= {"rise", "level"} and set(alarms["monitored"]) == {TARGET}


def test_residual_detector_indicators():
    idx = pd.date_range("2022-01-01", periods=5 * 144, freq="10min")
    df = pd.DataFrame({TARGET: np.arange(len(idx), dtype=float), "x": 1.0}, index=idx)
    det = DetectorFactory.create("residual", monitored=[TARGET], model_name="constant_forecaster")
    ind = det.fit(df.iloc[:144]).indicators(df)
    # daily mean of measured - predicted (constant = last training value 143)
    expected = df[TARGET].groupby(idx.floor("D")).mean() - 143.0
    pd.testing.assert_series_equal(ind[TARGET], expected, check_names=False)


@pytest.mark.parametrize("name, cls", [("kalman", KalmanHeatDetector), ("residual", ResidualDetector)])
def test_factory(name, cls):
    assert isinstance(DetectorFactory.create(name, monitored=[TARGET]), cls)
    with pytest.raises(ValueError):
        DetectorFactory.create("unknown", monitored=[TARGET])


def test_kalman_detector_monitored_from_network():
    from care_scada.modeling.thermal_network import ThermalNetworkSpec
    spec = ThermalNetworkSpec(target=TARGET, neighbours=["amb"]).with_measured_states(["nb"])
    det = DetectorFactory.create("kalman", network=spec)
    assert det.monitored == [TARGET, "nb"] and det.model_kwargs["network"] is spec
    # without a network the other monitored temperatures become measured states of the config network
    assert DetectorFactory.create("kalman", monitored=[TARGET, "nb"]).model_kwargs["measured_states"] == ["nb"]
    with pytest.raises(ValueError):
        DetectorFactory.create("kalman", monitored=["nb", TARGET], network=spec)


# --------------------------------------------------------------------------- Kalman detector (JAX)
pytest.importorskip("jax")
pytest.importorskip("optax")

from test_models import _make_network_df  # noqa: E402


def test_kalman_detector_flags_extra_heat():
    days, fault_day = 30, 20
    turbines = {s: _make_network_df(days * 144, seed=s) for s in range(3)}
    turbines["fault"] = _make_network_df(days * 144, seed=3, extra_heat=0.4, extra_from=fault_day * 144)
    det = DetectorFactory.create("kalman", monitored=[TARGET, "nb"], calibration_days=3, burn_in_days=5,
                                 level_lag=3, level_window=5,
                                 model_kwargs=dict(neighbours=["amb"], sources=["rpm"]))
    det.fit(turbines[0])
    ind = {name: det.indicators(df) for name, df in turbines.items()}
    assert list(ind["fault"].columns) == [TARGET, "nb"] and len(ind["fault"]) == days
    det.calibrate([ind[s] for s in range(3)])
    alarms = det.detect(ind["fault"])
    first = alarms["day"].iloc[0]
    assert ind["fault"].index[fault_day] <= first <= ind["fault"].index[fault_day + 2]
    assert alarms["monitored"].iloc[0] == TARGET


def test_plot_kalman_detector_network():
    from care_scada.plotter import plot_thermal_network
    from care_scada.modeling.thermal_network import ThermalNetworkSpec
    spec = ThermalNetworkSpec(target=TARGET, neighbours=["amb"], sources={"rpm": "linear"}).with_measured_states(["nb"])
    det = DetectorFactory.create("kalman", network=spec, model_kwargs=dict(steps=20))
    det.fit(_make_network_df(1000))
    fig = plot_thermal_network(det)
    text = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "true temperature of temp (monitored)" in text and "true temperature of nb (monitored)" in text
    assert "random walk" in text
    # both bearings are predicted latent states with measured sensors; c are latent heat inputs
    nodes = {t.name: set(t.text) for t in fig.data if t.mode == "markers+text"}
    assert nodes["predicted, latent"] == {"T<sub>0</sub>", "T<sub>1</sub>"}
    assert nodes["predicted, measured"] == {"y<sub>0</sub>", "y<sub>1</sub>"}
    assert nodes["helper, latent"] == {"c<sub>0</sub>", "c<sub>1</sub>"}
