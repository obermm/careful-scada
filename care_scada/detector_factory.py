"""Fault detectors and Detector Factory.

A detector turns a turbine's SCADA series into daily heat indicators per monitored sensor (K)
and raises alarms with two one-sided CUSUMs per sensor: "rise" (daily increments) and "level"
(deviation from a lagged rolling median). Thresholds are the largest CUSUM values seen on
healthy turbines.

Typical use::

    det = DetectorFactory.create("kalman", monitored=[bearing2, bearing1]).fit(healthy_df)
    ind = {name: det.indicators(df) for name, df in turbines.items()}
    det.calibrate([ind[name] for name in healthy_turbines])
    alarms = det.detect(ind[name])
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from care_scada.model_factory import ModelFactory


def cusum(inc: np.ndarray, k: float, h: float = np.inf) -> np.ndarray:
    """One-sided CUSUM S_d = max(0, S_{d-1} + inc_d - k), reset after S_d > h; NaN = no change."""
    s, acc = np.zeros(len(inc)), 0.0
    for i, v in enumerate(inc):
        if not np.isnan(v):
            acc = max(0.0, acc + v - k)
        s[i] = acc
        if acc > h:
            acc = 0.0
    return s


class Detector(ABC):
    """
    Base class: subclasses provide ``fit`` and the per-row indicator ``_indicator``.

    :param monitored: monitored temperature columns (one indicator and alarm check each)
    :param rise_k, level_k: CUSUM allowances (K/day and K)
    :param level_lag, level_window: level baseline = median of days t-lag-window .. t-lag
    :param burn_in_days: no alarms in the first days of a series
    """

    KINDS = ("rise", "level")

    def __init__(self, monitored: Sequence[str], rise_k: float = 0.02, level_k: float = 0.5,
                 level_lag: int = 30, level_window: int = 60, burn_in_days: int = 90):
        self.monitored = list(monitored)
        self.k = {"rise": rise_k, "level": level_k}
        self.level_lag, self.level_window, self.burn_in_days = level_lag, level_window, burn_in_days
        self.thresholds: Optional[Dict[str, float]] = None

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "Detector":
        """Fit the underlying model(s) on healthy data."""

    @abstractmethod
    def _indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicator per row of df and monitored column (K)."""

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Daily indicators of one turbine series (one column per monitored temperature)."""
        ind = self._indicator(df)
        return ind.groupby(ind.index.floor("D")).mean()

    def _increments(self, x: pd.Series):
        """Rise / level CUSUM increments after the burn-in, and their days."""
        x = x.asfreq("D")
        x = x[x.index >= x.index[0] + pd.Timedelta(days=self.burn_in_days)]
        base = x.rolling(self.level_window, min_periods=max(1, self.level_window // 3)).median()
        base = base.shift(self.level_lag)
        return {"rise": x.diff().to_numpy(), "level": (x - base).to_numpy()}, x.index

    def calibrate(self, healthy: Sequence[pd.DataFrame]) -> Dict[str, float]:
        """Thresholds = maximum CUSUM over the daily indicators of healthy turbines."""
        peaks = {kind: 0.0 for kind in self.KINDS}
        for ind in healthy:
            for t in self.monitored:
                inc, _ = self._increments(ind[t])
                for kind in self.KINDS:
                    peaks[kind] = max(peaks[kind], float(cusum(inc[kind], self.k[kind]).max(initial=0.0)))
        self.thresholds = peaks
        return peaks

    def detect(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """Alarms (day, monitored, kind) of one turbine's daily indicators, sorted by day."""
        if self.thresholds is None:
            raise RuntimeError("Detector must be calibrated before calling detect.")
        rows = []
        for t in self.monitored:
            inc, days = self._increments(indicators[t])
            for kind in self.KINDS:
                h = self.thresholds[kind]
                for i in np.flatnonzero(cusum(inc[kind], self.k[kind], h) > h):
                    rows.append({"day": days[i], "monitored": t, "kind": kind})
        return pd.DataFrame(rows, columns=["day", "monitored", "kind"]).sort_values("day", ignore_index=True)


class KalmanHeatDetector(Detector):
    """
    Tracked steady-state excess temperature of every monitored temperature in a Kalman heat
    network, with their constant heats as random walks (Verbund-style). The shared physics can be
    recalibrated on each turbine's first ``calibration_days`` (L2 prior ``prior_weight``).

    :param monitored: monitored temperatures; the first is the network target, the others become
        measured states (with a ``network``: default = its target and measured states)
    :param network: optional ThermalNetworkSpec (otherwise the config network of monitored[0])
    """

    def __init__(self, monitored: Optional[Sequence[str]] = None, network=None, calibration_days: int = 90,
                 prior_weight: float = 1.0, ss_std_per_day: float = 0.05,
                 model_kwargs: Optional[dict] = None, **detector_kwargs):
        if monitored is None:
            if network is None:
                raise ValueError("Pass the monitored temperatures or a network.")
            monitored = [network.target] + network.measured_states
        super().__init__(monitored, **detector_kwargs)
        self.calibration_days, self.prior_weight, self.ss_std_per_day = calibration_days, prior_weight, ss_std_per_day
        self.model_kwargs = {"training": "kalman", "steps": 800, **(model_kwargs or {})}
        if network is not None:
            if self.monitored[0] != network.target or set(self.monitored) - {network.target, *network.measured_states}:
                raise ValueError("Monitored must be the network target followed by measured states.")
            self.model_kwargs["network"] = network
        else:
            self.model_kwargs["measured_states"] = self.monitored[1:]
        self.model = None

    def fit(self, train_df: pd.DataFrame) -> "KalmanHeatDetector":
        self.model = ModelFactory.create("kalman", **self.model_kwargs).fit(train_df, self.monitored[0])
        return self

    def _indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        model = self.model
        if self.calibration_days:
            cal = df[df.index < df.index[0] + pd.Timedelta(days=self.calibration_days)]
            model = model.refit(cal, prior_weight=self.prior_weight)
        return model.track_heat(df, ss_std_per_day=self.ss_std_per_day)[self.monitored]


class ResidualDetector(Detector):
    """Residual (measured - predicted) of one normal-behaviour model per monitored temperature."""

    def __init__(self, monitored: Sequence[str], model_name: str = "pcrnn",
                 model_kwargs: Optional[dict] = None, **detector_kwargs):
        super().__init__(monitored, **detector_kwargs)
        self.model_name, self.model_kwargs = model_name, model_kwargs or {}
        self.models = {}

    def fit(self, train_df: pd.DataFrame) -> "ResidualDetector":
        for t in self.monitored:
            self.models[t] = ModelFactory.create(self.model_name, **self.model_kwargs)
            self.models[t].fit(train_df, t)
        return self

    def _indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({t: df[t] - m.predict(df, t)[t] for t, m in self.models.items()})


class DetectorFactory:
    """Factory for creating fault detectors."""

    @staticmethod
    def create(detector_name: str, **kwargs) -> Detector:
        """
        :param detector_name: "kalman" (KalmanHeatDetector) or "residual" (ResidualDetector)
        :param kwargs: keyword arguments of the detector constructor
        """
        if detector_name == "kalman":
            return KalmanHeatDetector(**kwargs)
        if detector_name == "residual":
            return ResidualDetector(**kwargs)
        raise ValueError(f"Unknown detector_name '{detector_name}'")
