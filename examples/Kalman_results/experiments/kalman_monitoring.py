"""Kalman thermal network experiments on CARE Wind Farm B (fit 83/86, test 77) and fault monitor.

The model (care_scada.modeling.kalman_forecaster.KalmanThermalForecaster, ``ModelFactory.create("kalman")``)
and the detectors (care_scada.detector_factory) live in the library. This script keeps what is
specific to the study:

* KalmanVariantForecaster: the library model plus experimental extensions of the variant study
  (hidden nodes, sensor feedthrough, turbine offsets, MLP residual heat), none of which improved
  on the library model;
* the variant comparison (val / test modes);
* the monitor evaluation on farm B: per-turbine calibration choices, per-turbine residual models
  ("*_local"), the bearing 2 - bearing 1 asymmetry signal, leave-one-turbine-out false alarms
  and the plots.

Usage:
    python examples/Kalman_results/experiments/kalman_monitoring.py val  [--variants a,b,...]
        model selection: fit on the first 80 % of the training parts of datasets 83 and 86,
        RMSE on the remaining 20 % of each.
    python examples/Kalman_results/experiments/kalman_monitoring.py test [--variants a,b,...]
        fit on all of datasets 83 and 86, test on dataset 77 (rotor-bearing damage,
        anomaly event 2023-06-01..2023-07-31): normal RMSE on 2023-05-18..05-31, mean event
        residual and SNR = (mean event residual - mean normal residual) / std of the normal residual.
    python examples/Kalman_results/experiments/kalman_monitoring.py monitor --out DIR [--methods kf:1,kf,pcrnn,lstm]
        fault monitor: physics fitted on healthy data (83 + 86 before 2022-10), optionally
        recalibrated per turbine on its first 90 days ("kf:<prior weight>"); thresholds from
        healthy turbines, detection on the main-bearing faults 77, 53, 27.
    python examples/Kalman_results/experiments/kalman_monitoring.py monitor-report --out DIR --methods ...
        re-evaluates the saved indicators of DIR without refitting.
"""

import argparse
import json
import os
import pickle
import time
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import jax.numpy as jnp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from care_scada.detector_factory import Detector, DetectorFactory
from care_scada.model_factory import ModelFactory, feature_columns
from care_scada.modeling.kalman_forecaster import KalmanThermalForecaster

OUTSIDE = "outside_temp_c_avg"


class KalmanVariantForecaster(KalmanThermalForecaster):
    """
    Library model with the extensions of the variant study.

    :param n_hidden: unobserved nodes coupled to the target and to the outside temperature
    :param feedthrough: the target sensor also sees the centred sink temperatures directly
    :param asset_offsets: sum-to-zero constant heat of the target per turbine (unknown: mean)
    :param residual_hidden: width of an MLP residual heat q(x_t) on the target (0 = off)
    :param residual_penalty: weight of the L2 penalty on the standardised residual heat
    :param residual_features: MLP inputs (default: feature_columns() of the target)
    """

    def __init__(self, n_hidden: int = 0, feedthrough: bool = False, asset_offsets: bool = False,
                 residual_hidden: int = 0, residual_penalty: float = 1.0, residual_features=None,
                 seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.n_hidden, self.feedthrough, self.asset_offsets = n_hidden, feedthrough, asset_offsets
        self.residual_hidden, self.residual_penalty, self.seed = residual_hidden, residual_penalty, seed
        self.residual_features = None if residual_features is None else list(residual_features)

    def _resolve_columns(self, df):
        super()._resolve_columns(df)
        self._assets = (sorted(df["asset_id"].dropna().unique().tolist())
                        if self.asset_offsets and "asset_id" in df else [0.0])
        if not self.residual_hidden:
            self.features_ = []
        elif self.residual_features is None:
            self.features_ = feature_columns(df, self.target_col)
        else:
            self.features_ = [c for c in self.residual_features if c in df]
        x = df[self.features_].to_numpy(float)
        self._x_mean = np.nan_to_num(np.nanmean(x, axis=0)) if self.features_ else np.zeros(0)
        sd = np.nan_to_num(np.nanstd(x, axis=0)) if self.features_ else np.zeros(0)
        self._x_std = np.where(sd > 1e-8, sd, 1.0)
        self._y_std = float(np.nanstd(df[self.target_col]))
        self._n_offsets = len(self._assets) if self.asset_offsets else 0
        self._n_base = len(self.sinks_) + len(self.sources_) + 2  # library inputs

    def _inputs(self, df):
        """u = [library inputs, turbine one-hot, standardised MLP features]."""
        parts = [super()._inputs(df)]
        if self.asset_offsets:
            ids = df["asset_id"].to_numpy(float) if "asset_id" in df else np.full(len(df), np.nan)
            onehot = (ids[:, None] == np.asarray(self._assets)[None, :]).astype(float)
            onehot[onehot.sum(1) == 0] = 1.0 / len(self._assets)
            parts.append(onehot)
        if self.features_:
            parts.append((df[self.features_].to_numpy(float) - self._x_mean) / self._x_std)
        return np.column_stack(parts)

    def _init_params(self):
        p = super()._init_params()
        n, nh = len(self.states_), self.n_hidden
        if nh:
            p["log_q"] = jnp.full(n + nh, np.log(0.05))
            p["log_k_hidden"] = jnp.full((nh, 3), np.log(0.05))
        if self.feedthrough:
            p["d"] = jnp.zeros(len(self.sinks_))
        if self.asset_offsets:
            p["asset"] = jnp.zeros(len(self._assets))
        if self.residual_hidden:
            rng = np.random.default_rng(self.seed)
            F, H = len(self.features_), self.residual_hidden
            p["mlp"] = {"W1": jnp.asarray(rng.normal(0, 1 / np.sqrt(F), (F, H))), "b1": jnp.zeros(H),
                        "W2": jnp.asarray(rng.normal(0, 1 / np.sqrt(H), (H, H))), "b2": jnp.zeros(H),
                        "w3": jnp.zeros(H), "b3": jnp.zeros(())}
        return p

    def _residual(self, p, x):
        """Residual heat q(x) on the target in K/h (zero without the MLP)."""
        if not self.residual_hidden:
            return jnp.zeros(x.shape[:-1])
        w = p["mlp"]
        h = jnp.tanh(jnp.tanh(x @ w["W1"] + w["b1"]) @ w["W2"] + w["b2"])
        return (h @ w["w3"] + w["b3"]) * self._y_std

    def _system(self, p):
        """Library system + turbine offsets, zero columns for the MLP features, q, hidden nodes."""
        A, B = super()._system(p)
        n, nh = len(self.states_), self.n_hidden
        B = jnp.concatenate([B, jnp.zeros((n, self._n_offsets + len(self.features_) + 1))], axis=1)
        if self.asset_offsets:
            B = B.at[0, self._n_base:self._n_base + self._n_offsets].set(p["asset"] - p["asset"].mean())
        B = B.at[0, -1].set(1.0)  # residual heat q
        if nh:
            A, B = jnp.pad(A, ((0, nh), (0, nh))), jnp.pad(B, ((0, nh), (0, 0)))
            kh = jnp.exp(p["log_k_hidden"])  # (target<-h, h<-target, h<-outside)
            out = self.sinks_.index(OUTSIDE) if OUTSIDE in self.sinks_ else None
            for j in range(nh):
                h = n + j
                A = A.at[0, 0].add(-kh[j, 0]).at[0, h].add(kh[j, 0])
                A = A.at[h, h].add(-kh[j, 1] - kh[j, 2]).at[h, 0].add(kh[j, 1])
                if out is not None:
                    B = B.at[h, out].add(kh[j, 2])
                else:
                    A = A.at[h, h].add(kh[j, 2])
        return A, B

    def _filter(self, p, u, y, m):
        q = self._residual(p, u[:, self._n_base + self._n_offsets:])
        n_s = len(self.sinks_)
        ft = (u[:, :n_s] - self._u_fill[:n_s]) @ p["d"] if self.feedthrough else 0.0
        # feedthrough = known input term of the target measurement: shift the observation
        nll, est = super()._filter(p, jnp.column_stack([u, q]), y.at[:, 0].add(-ft), m)
        return nll, est + ft

    def _penalty(self, p, U):
        if not self.residual_hidden:
            return 0.0
        q = self._residual(p, U[:, :, self._n_base + self._n_offsets:])
        return self.residual_penalty * jnp.mean((q / self._y_std) ** 2)


# ====================================================================== experiment
T = "main_bearing_temp2_c_avg"
GB1 = "gearbox_bearing_temp1_c_avg"
MB1 = "main_bearing_temp1_c_avg"
OIL = "gearbox_oil_temp1_c_avg"
OPERATING = ("rotor_speed_rpm_avg", "active_power_kWh", "gen_rpm_avg", "pitch_angle_deg_avg",
             "wind_speed1_ms_avg", "wind_speed2_ms_avg", "wind_speed3_ms_avg")

VARIANTS = {
    # library baseline
    "pcrnn": ("pcrnn", {}),
    "lstm": ("lstm", {}),
    # Kalman ML: target observed while fitting
    "kf": ("kalman", dict(training="kalman")),
    "kf_mb1_input": ("kalman", dict(training="kalman", neighbours=(OUTSIDE, GB1, MB1), measured_states=())),
    # observer: trained on the free-run error of the unobserved target
    "obs": ("kalman", dict(training="observer")),
    "obs_mb1_input": ("kalman", dict(training="observer", neighbours=(OUTSIDE, GB1, MB1), measured_states=())),
    "obs_noheat": ("kalman", dict(training="observer", const_heat=False, temp_heat=False)),
    "obs_hidden": ("kalman", dict(training="observer", n_hidden=1)),
    "obs_ft": ("kalman", dict(training="observer", feedthrough=True)),
    "obs_asset": ("kalman", dict(training="observer", asset_offsets=True)),
    "obs_oil": ("kalman", dict(training="observer", neighbours=(OUTSIDE, GB1, OIL))),
    "obs_nosrc": ("kalman", dict(training="observer", sources=())),
    # with an MLP residual heat source (as in the library PCRNN)
    "obs_mlp": ("kalman", dict(training="observer", residual_hidden=16, steps=1500, learning_rate=0.01)),
    "obs_mb1_input_mlp": ("kalman", dict(training="observer", neighbours=(OUTSIDE, GB1, MB1), measured_states=(),
                                         residual_hidden=16, steps=1500, learning_rate=0.01)),
    "kf_mlp": ("kalman", dict(training="kalman", residual_hidden=16, steps=1500, learning_rate=0.01)),
    "obs_mlp_pen01": ("kalman", dict(training="observer", residual_hidden=16, residual_penalty=0.1,
                                     steps=1500, learning_rate=0.01)),
    "obs_mlp_asset": ("kalman", dict(training="observer", residual_hidden=16, asset_offsets=True,
                                     steps=1500, learning_rate=0.01)),
    "obs_mlp_hidden": ("kalman", dict(training="observer", residual_hidden=16, n_hidden=1,
                                      steps=1500, learning_rate=0.01)),
    # residual heat only from operating signals (no other temperatures)
    "obs_mlpop": ("kalman", dict(training="observer", residual_hidden=16, residual_features=OPERATING,
                                 steps=1500, learning_rate=0.01)),
    "obs_mb1_input_mlpop": ("kalman", dict(training="observer", neighbours=(OUTSIDE, GB1, MB1),
                                           measured_states=(), residual_hidden=16,
                                           residual_features=OPERATING, steps=1500, learning_rate=0.01)),
    "obs_mb1_input_asset": ("kalman", dict(training="observer", neighbours=(OUTSIDE, GB1, MB1),
                                           measured_states=(), asset_offsets=True)),
    "obs_mb1_input_mlpop_asset": ("kalman", dict(training="observer", neighbours=(OUTSIDE, GB1, MB1),
                                                 measured_states=(), residual_hidden=16, asset_offsets=True,
                                                 residual_features=OPERATING, steps=1500, learning_rate=0.01)),
}


def load_datasets(data_path: str, ids=(77, 83, 86)) -> Dict[int, tuple]:
    from care_scada.data_loader import DataLoader
    from care_scada.preprocessing import Preprocessor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = DataLoader(farm_name="B", data_path=data_path).load()
        out = {}
        for d in ids:
            tr, te = Preprocessor(farm_name="B", dataset_id=d, missing_values="drop",
                                  outlier_handling="drop").process(df)
            k = tr.index.get_level_values(0)[0]
            out[d] = (tr.xs(k, level=0), te.xs(k, level=0))
    return out


# network of the study unless a variant overrides it
KALMAN_DEFAULTS = dict(neighbours=(OUTSIDE, GB1), measured_states=(MB1,),
                       sources=("rotor_speed_rpm_avg", "active_power_kWh"))


def make_model(name):
    """Variant name, optionally with a seed suffix, e.g. 'pcrnn@s1'."""
    base, _, seed = name.partition("@s")
    kind, kw = VARIANTS[base]
    if seed:
        kw = dict(kw, seed=int(seed))
    if kind == "kalman":
        return KalmanVariantForecaster(**{**KALMAN_DEFAULTS, **kw})
    return ModelFactory.create(kind, **kw)


def rmse(a, b):
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def _fit(name, fit_df):
    model = make_model(name)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(fit_df, T)
    return model, round(time.time() - t0, 1)


def _with_params(res, model):
    if isinstance(model, KalmanThermalForecaster):
        res["params"] = {k: round(v, 4) for k, v in model.summary().items()}
    return res


def run_val(data, name, out_dir=None):
    """Fit on the first 80 % of the training parts of 83 and 86, evaluate on the last 20 %."""
    fit_parts, val_parts = [], {}
    for d in (83, 86):
        tr = data[d][0]
        cut = int(len(tr) * 0.8)
        fit_parts.append(tr.iloc[:cut])
        val_parts[d] = tr.iloc[cut - 432:]  # 3 days of history before the val part
    model, fit_s = _fit(name, pd.concat(fit_parts))
    res = {"variant": name, "mode": "val", "fit_s": fit_s}
    for d, df in val_parts.items():
        r = (df[T] - model.predict(df, T)[T]).to_numpy()[432:]
        res[f"val_rmse_{d}"] = round(rmse(r, 0), 3)
        res[f"val_bias_{d}"] = round(float(np.nanmean(r)), 3)
    return _with_params(res, model), model


def run_test(data, name, out_dir=None):
    """
    Fit on the CARE training parts of 83 and 86. Normal behaviour, out of sample: test parts of
    83 (asset 2) and 86 (asset 12, = the turbine of 77). Dataset 77: 2023-05-18..05-31 (not
    covered by 86) and the damage event 2023-06-01..07-31.
    """
    model, fit_s = _fit(name, pd.concat([data[83][0], data[86][0]]))
    res = {"variant": name, "mode": "test", "fit_s": fit_s}
    for d in (83, 86):
        hist = pd.concat([data[d][0].iloc[-432:], data[d][1]])
        r = (hist[T] - model.predict(hist, T)[T]).to_numpy()[432:]
        res[f"test_rmse_{d}"] = round(rmse(r, 0), 3)
        res[f"test_bias_{d}"] = round(float(np.nanmean(r)), 3)
    df77 = pd.concat(data[77])
    pred = model.predict(df77, T)[T]
    resid = df77[T] - pred
    idx = df77.index
    normal = (idx > data[86][1].index[-1]) & (idx < "2023-06-01")
    event = (idx >= "2023-06-01") & (idx < "2023-08-01")
    r_n, r_e = resid[normal], resid[event]
    res.update({"normal_rmse_77": round(rmse(r_n, 0), 3), "normal_bias_77": round(float(r_n.mean()), 3),
                "event_mean_resid_77": round(float(r_e.mean()), 3),
                "snr_77": round(float((r_e.mean() - r_n.mean()) / r_n.std()), 2),
                "monthly_resid_77": {str(k): round(float(v), 2) for k, v in
                                     resid.groupby(idx.to_period("M")).mean().items()}})
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({"measured": df77[T], "predicted": pred}).to_csv(
            os.path.join(out_dir, f"pred77_{name}.csv"))
    return _with_params(res, model), model


# ====================================================================== monitoring
BEARINGS = [T, MB1]
HEALTHY_END_86 = "2022-10-01"     # asset 12: bearing 2 - bearing 1 gap starts rising in 2022 Q4
NEGATIVE_DS = (83, 52, 21, 2, 74, 82)  # one normal dataset per turbine without a bearing fault
POSITIVE_DS = (77, 53, 27)             # main-bearing damage events (assets 12, 6, 7)
CAL_DAYS = 90                     # per-turbine calibration window: the first days of each series
BURN_IN_DAYS = 90                 # no alarms during the calibration window


def healthy_fit_data(data) -> pd.DataFrame:
    """All of dataset 83 (asset 2) and dataset 86 (asset 12) before its bearing-2 drift."""
    d86 = pd.concat(data[86])
    return pd.concat([data[83][0], data[83][1], d86[d86.index < HEALTHY_END_86]])


def full_series(data, d) -> pd.DataFrame:
    return pd.concat(data[d])


def calibration_part(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.index < df.index[0] + pd.Timedelta(days=CAL_DAYS)]


def indicators(data, method: str) -> Dict[int, pd.DataFrame]:
    """
    Daily indicator per bearing (K) for all monitored datasets.

    kf            kalman detector, shared physics (healthy 83/86)
    kf:<w>        + per-turbine refit on the first CAL_DAYS, L2 prior weight w towards the shared fit
    pcrnn, lstm   residual detector with shared models (one per bearing)
    pcrnn_local, lstm_local  residual detector fitted per turbine on its first CAL_DAYS
    """
    base, _, arg = method.partition(":")
    ids = NEGATIVE_DS + POSITIVE_DS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if base == "kf":
            det = DetectorFactory.create("kalman", monitored=BEARINGS, calibration_days=CAL_DAYS if arg else 0,
                                         prior_weight=float(arg or 1.0)).fit(healthy_fit_data(data))
            return {d: det.indicators(full_series(data, d)) for d in ids}
        kind = base.replace("_local", "")
        if base.endswith("_local"):
            out = {}
            for d in ids:
                df = full_series(data, d)
                det = DetectorFactory.create("residual", monitored=BEARINGS, model_name=kind)
                out[d] = det.fit(calibration_part(df)).indicators(df)
            return out
        det = DetectorFactory.create("residual", monitored=BEARINGS, model_name=kind).fit(healthy_fit_data(data))
        return {d: det.indicators(full_series(data, d)) for d in ids}


class SignalAlarms(Detector):
    """CUSUM alarms on precomputed daily signals (no model)."""

    def fit(self, train_df):
        return self

    def _indicator(self, df):
        return df


SIGNAL_SETS = {
    # per bearing: extra heat in either bearing
    "bearings": lambda f: f[BEARINGS],
    # asymmetry: bearing 2 minus bearing 1, both directions (a fault heats ONE bearing, while
    # misfit of the shared physics, e.g. unmodelled operating states, is common to both)
    "asymmetry": lambda f: pd.DataFrame({"diff+": f[T] - f[MB1], "diff-": f[MB1] - f[T]}),
}


def evaluate_monitor(ind: Dict[int, pd.DataFrame], event_info: pd.DataFrame,
                     signal_set: str = "bearings") -> dict:
    """
    Thresholds = maximum CUSUM over the healthy turbines (signals of a set pooled). False alarms
    on a healthy turbine use leave-one-turbine-out thresholds; detections use all of them.
    """
    sig = {d: SIGNAL_SETS[signal_set](f) for d, f in ind.items()}
    det = SignalAlarms(monitored=next(iter(sig.values())).columns, burn_in_days=BURN_IN_DAYS)
    h = det.calibrate([sig[d] for d in NEGATIVE_DS])
    res = {"signals": signal_set, "thresholds": {k: round(v, 3) for k, v in h.items()},
           "false_alarms": {}, "detections": {}}

    def listed(alarms):
        return [(str(a.day.date()), a.monitored, a.kind) for a in alarms.itertuples()]

    for d in POSITIVE_DS:
        start = pd.Timestamp(event_info.loc[d, "event_start"])
        alarms = det.detect(sig[d])
        first = alarms["day"].iloc[0] if len(alarms) else None
        res["detections"][d] = {"event_start": str(start.date()),
                                "first_alarm": str(first.date()) if first is not None else None,
                                "lead_days": int((start - first).days) if first is not None else None,
                                "alarms": listed(alarms)}
    for d in NEGATIVE_DS:
        det.calibrate([sig[o] for o in NEGATIVE_DS if o != d])
        res["false_alarms"][d] = listed(det.detect(sig[d]))
    res["n_false_alarm_turbines"] = sum(bool(v) for v in res["false_alarms"].values())
    return res


def report_monitor(out_dir: str, methods, data_path: str):
    """Re-evaluate saved indicators (indicators_<method>.csv) for both signal sets."""
    event_info = load_event_info(data_path)
    rows, ind_by_method = [], {}
    for mth in methods:
        f = pd.read_csv(os.path.join(out_dir, f"indicators_{mth}.csv"), index_col=[0, 1], parse_dates=[1])
        ind = {int(d): g.droplevel(0) for d, g in f.groupby(level=0)}
        ind_by_method[mth] = ind
        for sset in SIGNAL_SETS:
            r = evaluate_monitor(ind, event_info, sset)
            rows.append({"method": mth, "signals": sset, "false_alarm_turbines": r["n_false_alarm_turbines"],
                         **{f"lead_{d}": r["detections"][d]["lead_days"] for d in POSITIVE_DS},
                         "h_rise": r["thresholds"]["rise"], "h_level": r["thresholds"]["level"]})
            print("DETAIL", mth, sset, json.dumps(r), flush=True)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(out_dir, "monitor_summary.csv"), index=False)
    plot_monitor(ind_by_method, event_info, os.path.join(out_dir, "monitor_positives.png"))
    return table


def plot_monitor(ind_by_method: Dict[str, Dict[int, pd.DataFrame]], event_info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(ind_by_method)
    fig, axes = plt.subplots(len(POSITIVE_DS), len(methods), figsize=(5 * len(methods), 3.2 * len(POSITIVE_DS)),
                             squeeze=False, sharex="row")
    for i, d in enumerate(POSITIVE_DS):
        ev = event_info.loc[d]
        for j, mth in enumerate(methods):
            ax = axes[i, j]
            frame = ind_by_method[mth][d]
            frame = frame.assign(**{"bearing 2 − 1": frame[T] - frame[MB1]})
            for b, color in zip(list(BEARINGS) + ["bearing 2 − 1"], ("#E4572E", "#4C78A8", "#222222")):
                ax.plot(frame.index, frame[b] - frame[b].iloc[:30].mean(), color=color, lw=1,
                        label=b.replace("_c_avg", ""))
            ax.axvspan(pd.Timestamp(ev["event_start"]), pd.Timestamp(ev["event_end"]), color="0.85")
            ax.axhline(0, color="0.5", lw=0.5)
            ax.set_title(f"{mth}: dataset {d} (asset {int(ev['asset_id'])})", fontsize=10)
            if j == 0:
                ax.set_ylabel("daily indicator − first 30 d [K]")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_event_info(data_path: str) -> pd.DataFrame:
    info = pd.read_csv(os.path.join(data_path, "Wind Farm B", "event_info.csv"), sep=";")
    return info.set_index("event_id")


def run_monitor(data, methods, data_path, out_dir):
    event_info = load_event_info(data_path)
    ind_by_method, results = {}, {}
    for mth in methods:
        t0 = time.time()
        ind_by_method[mth] = indicators(data, mth)
        results[mth] = evaluate_monitor(ind_by_method[mth], event_info)
        results[mth]["time_s"] = round(time.time() - t0, 1)
        print("RESULT", mth, json.dumps(results[mth]), flush=True)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            pd.concat({d: f for d, f in ind_by_method[mth].items()}, names=["dataset", "day"]).to_csv(
                os.path.join(out_dir, f"indicators_{mth}.csv"))
            with open(os.path.join(out_dir, "monitor_results.jsonl"), "a") as f:
                f.write(json.dumps({"method": mth, **results[mth]}) + "\n")
    if out_dir:
        plot_monitor(ind_by_method, event_info, os.path.join(out_dir, "monitor_positives.png"))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["val", "test", "monitor", "monitor-report"])
    ap.add_argument("--methods", default="kf:1,kf,pcrnn,lstm",
                    help="monitor modes: indicators to compare (see indicators())")
    ap.add_argument("--burn-in-days", type=int, default=None,
                    help="monitor modes: no alarms in the first days of each series (default 90)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--data-path", default=os.path.join(REPO_ROOT, "dat",
                                                        "Care_To_Compare"))
    ap.add_argument("--out", default=None, help="folder for results.jsonl and predictions")
    ap.add_argument("--data-cache", default=None, help="pickle of the preprocessed datasets "
                    "(created if missing)")
    args = ap.parse_args()
    if args.burn_in_days is not None:
        global BURN_IN_DAYS
        BURN_IN_DAYS = args.burn_in_days
    if args.mode == "monitor-report":
        report_monitor(args.out, args.methods.split(","), args.data_path)
        return
    if args.data_cache and os.path.exists(args.data_cache):
        with open(args.data_cache, "rb") as f:
            data = pickle.load(f)
    else:
        data = load_datasets(args.data_path, ids=sorted(set((77, 83, 86) + NEGATIVE_DS + POSITIVE_DS)))
        if args.data_cache:
            with open(args.data_cache, "wb") as f:
                pickle.dump(data, f)
    if args.mode == "monitor":
        run_monitor(data, args.methods.split(","), args.data_path, args.out)
        return
    for name in args.variants.split(","):
        res, _ = (run_val if args.mode == "val" else run_test)(data, name, args.out)
        print("RESULT", json.dumps(res), flush=True)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, "results.jsonl"), "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
