"""Fault detectors and Detector Factory.

A detector turns a turbine's SCADA series into daily heat indicators per monitored sensor (K)
and raises alarms with one of two routines:

* ``alarm="cusum"`` (default): two one-sided CUSUMs per sensor, "rise" (daily increments) and
  "level" (deviation from a lagged rolling median),
* ``alarm="level"``: trailing ``mean_days`` mean of the deviation from a FIXED per-turbine
  baseline (median of ``baseline_days``), alarm only if above the threshold on ``persist_days``
  consecutive days. Unlike the rolling baseline it does not absorb slow faults.

With ``reference="cross"`` every monitored indicator is first referenced to the others,
x_i - alpha_i * mean(other x), alpha_i in [0, 1] regressed on the baseline days. This removes the
drift common to neighbouring bearings (seasonal model misfit) but keeps the fault of one bearing.

Thresholds are ``margin`` times the largest statistic seen on healthy turbines.

Typical use::

    det = DetectorFactory.create("kalman", monitored=[bearing2, bearing1]).fit(healthy_df)
    ind = {name: det.indicators(df) for name, df in turbines.items()}
    det.calibrate([ind[name] for name in healthy_turbines])
    alarms = det.detect(ind[name])

The recommended main-bearing detector ("D4", see examples/Kalman_results/results/03_fault_detection/fault_detection.txt) is
``DetectorFactory.create("freerun", network=spec, cooling=[...])``: free-run residual of both
bearings, cross-referenced, fixed-baseline level routine, margin 1.15.
"""

import copy
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple

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
    :param alarm: "cusum" (rise + level CUSUM) or "level" (fixed-baseline level, see module doc)
    :param reference: "none" or "cross" (each indicator minus alpha * mean of the others)
    :param margin: thresholds = margin * maximum statistic over the healthy turbines
    :param rise_k, level_k: CUSUM allowances (K/day and K)
    :param level_lag, level_window: CUSUM level baseline = median of days t-lag-window .. t-lag
    :param mean_days, persist_days: "level" routine: trailing mean and persistence (days)
    :param baseline_days: (first, last) day of the fixed baseline and of the alpha regression,
        counted from the start of the series (should lie in the burn-in)
    :param burn_in_days: no alarms in the first days of a series
    """

    def __init__(self, monitored: Sequence[str], alarm: str = "cusum", reference: str = "none",
                 margin: float = 1.0, rise_k: float = 0.02, level_k: float = 0.5,
                 level_lag: int = 30, level_window: int = 60, mean_days: int = 14, persist_days: int = 7,
                 baseline_days: Tuple[int, int] = (30, 90), burn_in_days: int = 90):
        if alarm not in ("cusum", "level"):
            raise ValueError("alarm must be 'cusum' or 'level'")
        if reference not in ("none", "cross"):
            raise ValueError("reference must be 'none' or 'cross'")
        if reference == "cross" and len(monitored) < 2:
            raise ValueError("reference='cross' needs at least two monitored temperatures")
        self._alarm_kwargs = dict(alarm=alarm, reference=reference, margin=margin, rise_k=rise_k,
                                  level_k=level_k, level_lag=level_lag, level_window=level_window,
                                  mean_days=mean_days, persist_days=persist_days,
                                  baseline_days=baseline_days, burn_in_days=burn_in_days)
        self.monitored = list(monitored)
        self.alarm, self.reference, self.margin = alarm, reference, margin
        self.k = {"rise": rise_k, "level": level_k}
        self.level_lag, self.level_window, self.burn_in_days = level_lag, level_window, burn_in_days
        self.mean_days, self.persist_days, self.baseline_days = mean_days, persist_days, tuple(baseline_days)
        self.thresholds: Optional[Dict[str, float]] = None

    def with_alarm(self, **alarm_kwargs) -> "Detector":
        """Copy sharing the fitted model, with other alarm settings (alarm, reference, margin, ...);
        the copy has to be calibrated again. Indicators do not depend on these settings."""
        new = copy.copy(self)
        Detector.__init__(new, self.monitored, **{**self._alarm_kwargs, **alarm_kwargs})
        return new

    @property
    def kinds(self) -> Tuple[str, ...]:
        return ("rise", "level") if self.alarm == "cusum" else ("level",)

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

    def _baseline_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        t0 = index[0]
        return ((index >= t0 + pd.Timedelta(days=self.baseline_days[0]))
                & (index < t0 + pd.Timedelta(days=self.baseline_days[1])))

    def signals(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """Daily signals the alarm routine runs on (the indicators, or cross-referenced)."""
        ind = indicators[self.monitored].asfreq("D")
        if self.reference == "none":
            return ind
        w = self._baseline_mask(ind.index)
        out = {}
        for t in self.monitored:
            y, x = ind[t], ind.drop(columns=t).mean(axis=1)
            ok = w & y.notna().to_numpy() & x.notna().to_numpy()
            yc, xc = y[ok] - y[ok].mean(), x[ok] - x[ok].mean()
            alpha = float(np.clip((yc * xc).sum() / max(float((xc ** 2).sum()), 1e-9), 0.0, 1.0))
            out[t] = y - alpha * x
        return pd.DataFrame(out)

    def _statistics(self, x: pd.Series, h: Optional[Dict[str, float]] = None):
        """Statistic per kind on the days after the burn-in (CUSUMs reset after exceeding h)."""
        start = x.index[0] + pd.Timedelta(days=self.burn_in_days)
        if self.alarm == "level":
            base = np.nanmedian(x[self._baseline_mask(x.index)].to_numpy()) \
                if self._baseline_mask(x.index).any() else np.nan
            s = (x - base).rolling(self.mean_days, min_periods=max(1, self.mean_days // 2)).mean()
            if self.persist_days > 1:
                s = s.rolling(self.persist_days, min_periods=self.persist_days).min()
            keep = x.index >= start
            return {"level": s.to_numpy()[keep]}, x.index[keep]
        x = x[x.index >= start]
        base = x.rolling(self.level_window, min_periods=max(1, self.level_window // 3)).median()
        base = base.shift(self.level_lag)
        inc = {"rise": x.diff().to_numpy(), "level": (x - base).to_numpy()}
        h = h or {}
        return {kind: cusum(inc[kind], self.k[kind], h.get(kind, np.inf)) for kind in self.kinds}, x.index

    def calibrate(self, healthy: Sequence[pd.DataFrame]) -> Dict[str, float]:
        """Thresholds = margin * maximum statistic over the daily indicators of healthy turbines."""
        peaks = {kind: 0.0 for kind in self.kinds}
        for ind in healthy:
            sig = self.signals(ind)
            for t in self.monitored:
                stats, _ = self._statistics(sig[t])
                for kind in self.kinds:
                    peaks[kind] = max(peaks[kind], float(np.nanmax(stats[kind], initial=0.0)))
        self.thresholds = {kind: self.margin * v for kind, v in peaks.items()}
        return self.thresholds

    def detect(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """Alarms (day, monitored, kind) of one turbine's daily indicators, sorted by day."""
        if self.thresholds is None:
            raise RuntimeError("Detector must be calibrated before calling detect.")
        rows = []
        sig = self.signals(indicators)
        for t in self.monitored:
            stats, days = self._statistics(sig[t], self.thresholds)
            for kind in self.kinds:
                h = self.thresholds[kind]
                for i in np.flatnonzero(np.nan_to_num(stats[kind], nan=-np.inf) > h):
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
    """
    Residual (measured - predicted) of one normal-behaviour model per monitored temperature.

    :param monitored: monitored temperatures (default with a ``network``: its target and measured states)
    :param network: optional ThermalNetworkSpec for network models (pcrnn, kalman); the model of each
        monitored temperature gets the network retargeted to it (``ThermalNetworkSpec.retarget``)
    """

    def __init__(self, monitored: Optional[Sequence[str]] = None, model_name: str = "pcrnn",
                 model_kwargs: Optional[dict] = None, network=None, **detector_kwargs):
        if monitored is None:
            if network is None:
                raise ValueError("Pass the monitored temperatures or a network.")
            monitored = [network.target] + network.measured_states
        super().__init__(monitored, **detector_kwargs)
        self.model_name, self.model_kwargs, self.network = model_name, model_kwargs or {}, network
        self.models = {}

    def fit(self, train_df: pd.DataFrame) -> "ResidualDetector":
        for t in self.monitored:
            kwargs = dict(self.model_kwargs)
            if self.network is not None:
                kwargs["network"] = self.network.retarget(t)
            self.models[t] = ModelFactory.create(self.model_name, **kwargs)
            self.models[t].fit(train_df, t)
        return self

    def _indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({t: df[t] - m.predict(df, t)[t] for t, m in self.models.items()})


class FreeRunResidualDetector(Detector):
    """
    Free-run residual (measured - simulated, no sensor assimilated) of the network target and its
    measured states ("D4"). The shared physics is fitted on pooled healthy turbines and recalibrated
    on each turbine's first ``calibration_days`` (L2 prior ``prior_weight``). Defaults: cross-
    referenced indicators, fixed-baseline level routine, margin 1.15.

    On CARE Wind Farm B (main bearings, "simple" network below) it flagged the slow bearing-2 fault
    of dataset 77 ~3 months early and 53 at the event, without alarms on the 6 healthy turbines::

        spec = ThermalNetworkSpec(
            target=bearing2, measured_states=[bearing1],
            edges=[(bearing1, gearbox_bearing1), (bearing1, outside), (bearing2, gearbox_bearing1),
                   (bearing2, outside)],
            sources={wind_speed: [(bearing1, "linear"), (bearing2, "linear")],
                     rotor_speed: [(bearing1, "square"), (bearing2, "square")]})
        det = DetectorFactory.create("freerun", network=spec,
                                     cooling=[(wind_speed, bearing1), (wind_speed, bearing2)])

    :param network: ThermalNetworkSpec; monitored = its target followed by its measured states
    :param cooling: source links with a negative gain (see KalmanThermalForecaster)
    :param const_heat, temp_heat: constant / outside-temperature heat terms per measured state
    :param model_kwargs: further KalmanThermalForecaster arguments (steps, learning_rate, ...);
        the model is trained with ``training="free"``
    """

    def __init__(self, network, cooling: Sequence = (), const_heat: bool = False, temp_heat: bool = False,
                 calibration_days: int = 90, prior_weight: float = 1.0, model_kwargs: Optional[dict] = None,
                 alarm: str = "level", reference: str = "cross", margin: float = 1.15, **detector_kwargs):
        monitored = [network.target] + network.measured_states
        if reference == "cross" and len(monitored) < 2:
            reference = "none"
        super().__init__(monitored, alarm=alarm, reference=reference, margin=margin, **detector_kwargs)
        self.calibration_days, self.prior_weight = calibration_days, prior_weight
        self.model_kwargs = {"training": "free", "chunk_len": 1008, "steps": 800, "learning_rate": 0.03,
                             **(model_kwargs or {}), "network": network, "cooling": cooling,
                             "const_heat": const_heat, "temp_heat": temp_heat}
        self.model = None

    def fit(self, train_df: pd.DataFrame) -> "FreeRunResidualDetector":
        self.model = ModelFactory.create("kalman", **self.model_kwargs).fit(train_df, self.monitored[0])
        return self

    def _indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        model = self.model
        if self.calibration_days:
            cal = df[df.index < df.index[0] + pd.Timedelta(days=self.calibration_days)]
            model = model.refit(cal, prior_weight=self.prior_weight)
        return df[self.monitored] - model.simulate(df)[self.monitored]


class DetectorFactory:
    """Factory for creating fault detectors."""

    @staticmethod
    def create(detector_name: str, **kwargs) -> Detector:
        """
        :param detector_name: "kalman" (KalmanHeatDetector), "residual" (ResidualDetector) or
            "freerun" (FreeRunResidualDetector, the recommended main-bearing detector D4)
        :param kwargs: keyword arguments of the detector constructor
        """
        if detector_name == "kalman":
            return KalmanHeatDetector(**kwargs)
        if detector_name == "residual":
            return ResidualDetector(**kwargs)
        if detector_name == "freerun":
            return FreeRunResidualDetector(**kwargs)
        raise ValueError(f"Unknown detector_name '{detector_name}'")
