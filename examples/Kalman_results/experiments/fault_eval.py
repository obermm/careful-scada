"""Alarm routines on the daily indicators of fault_detection.py, and their evaluation.

A routine maps one daily signal x (K, one turbine) to a one-sided statistic s_t >= 0 per kind;
the threshold of each kind is the maximum of s over the healthy turbines' real series
(leave-one-turbine-out when a healthy turbine is evaluated). Alarm: s_t > threshold, after
BURN_IN days. Signal sets: "bearings" (each bearing's indicator) and "asym" (difference of the
two bearings, both signs).

Metrics:
  FA      healthy turbines with an alarm in their real series (LOO thresholds)
  lead_d  event_start - first alarm (days) on fault dataset d (positive = before the event)
  detXA   (X = A offset / L load-proportional ramp) share of injected ramps (amplitude A K, 6 turbines x 2 bearings x 3 starts) alarmed within
          DET_WINDOW days after the ramp start on the injected bearing's signal, earlier than the same
          signal without the ramp (an alarm the healthy series raises anyway is no detection)
  delXA   median days from ramp start to the first alarm (detected ramps)

    python fault_eval.py --out DIR [--routines cusum,...]
"""

import argparse
import os

import numpy as np
import pandas as pd

from care_scada.detector_factory import cusum
import fault_detection as fd

BURN_IN = 90
RAMP_TYPES = ("A", "L")
MARGIN = 1.0  # threshold = MARGIN * maximum over the (other) healthy turbines
DET_WINDOW = 90
EVENTS = {77: "2023-06-01", 53: "2023-12-26 16:40", 27: "2023-09-01"}
FAULT_BEARING = {77: "mb2", 53: "mb2", 27: None}  # "Rotor Bearing 2" damage (event_info); 27 unspecified


# ---------------------------------------------------------------------------- signals
def signals(f: pd.DataFrame, kind: str, sset: str) -> pd.DataFrame:
    """Daily signals of one turbine/case (index day)."""
    a, b = f[f"{kind}_mb2"], f[f"{kind}_mb1"]
    if sset == "bearings":
        return pd.DataFrame({"mb2": a, "mb1": b})
    if sset == "asym":
        return pd.DataFrame({"mb2": a - b, "mb1": b - a})
    if sset == "xreg":  # each bearing minus alpha * the other (alpha: regression on the baseline days)
        out = {}
        for name, y, x in (("mb2", a, b), ("mb1", b, a)):
            t0 = y.index[0]
            w = (y.index >= t0 + pd.Timedelta(days=30)) & (y.index < t0 + pd.Timedelta(days=90))
            ok = w & y.notna().to_numpy() & x.notna().to_numpy()
            yc, xc = y[ok] - y[ok].mean(), x[ok] - x[ok].mean()
            alpha = float(np.clip((yc * xc).sum() / max((xc ** 2).sum(), 1e-9), 0.0, 1.0))
            out[name] = y - alpha * x
        return pd.DataFrame(out)
    raise ValueError(sset)


# ---------------------------------------------------------------------------- routines
def _daily(x: pd.Series) -> pd.Series:
    return x.asfreq("D")


def r_cusum(x, k_rise=0.02, k_level=0.5, lag=30, window=60):
    """Existing library routine: CUSUM of daily increments (rise) and of the deviation from a
    lagged rolling median (level)."""
    x = _daily(x)
    base = x.rolling(window, min_periods=window // 3).median().shift(lag)
    return {"rise": cusum(x.diff().to_numpy(), k_rise), "level": cusum((x - base).to_numpy(), k_level)}


BASE_DAYS = (30, 90)  # fixed per-turbine baseline: median of these days (after the tracker transient)


def _dev(x):
    """Deviation from the fixed baseline of the turbine (calibration window)."""
    t0 = x.index[0]
    w = x[(x.index >= t0 + pd.Timedelta(days=BASE_DAYS[0])) & (x.index < t0 + pd.Timedelta(days=BASE_DAYS[1]))]
    return x - np.nanmedian(w.to_numpy())


def r_level(x, win=14, persist=1):
    """Trailing win-day mean of the deviation from the fixed baseline; alarm only if it exceeds
    the threshold on persist consecutive days (= rolling minimum)."""
    s = _dev(x).rolling(win, min_periods=win // 2).mean()
    if persist > 1:
        s = s.rolling(persist, min_periods=persist).min()
    return {"level": s.to_numpy()}


def r_level_lag(x, win=14, lag=60, base_win=90):
    """Trailing win-day mean minus the median of the base_win days ending lag days ago (adapts to
    slow drifts, unlike the fixed baseline; the existing CUSUM uses lag 30 / window 60)."""
    x = _daily(x)
    base = x.rolling(base_win, min_periods=base_win // 3).median().shift(lag)
    return {"level": (x.rolling(win, min_periods=win // 2).mean() - base).to_numpy()}


def r_ewma(x, halflife=10.0):
    return {"ewma": _dev(x).ewm(halflife=halflife, ignore_na=True).mean().to_numpy()}


def r_cusum_fix(x, k=1.0):
    """CUSUM of the deviation from the fixed baseline (allowance k K)."""
    return {"level": cusum(_dev(x).to_numpy(), k)}


ROUTINES = {
    "cusum": r_cusum,
    "lvl14": r_level,
    "lag60": r_level_lag,
    "lvl30": lambda x: r_level(x, win=30),
    "lvl14p7": lambda x: r_level(x, win=14, persist=7),
    "ewma10": r_ewma,
    "cusumfix1": r_cusum_fix,
    "cusumfix2": lambda x: r_cusum_fix(x, k=2.0),
}


def statistics(x: pd.Series, routine: str):
    """Statistic per kind after the burn-in (day index)."""
    x = _daily(x)
    stats = ROUTINES[routine](x)
    keep = x.index >= x.index[0] + pd.Timedelta(days=BURN_IN)
    return {k: pd.Series(np.asarray(v, float)[keep], index=x.index[keep]) for k, v in stats.items()}


def first_alarm(stats: dict, h: dict, after=None, before=None):
    days = []
    for k, s in stats.items():
        s = s[(s.index >= after) if after is not None else slice(None)]
        if before is not None:
            s = s[s.index < before]
        hit = s.index[s.to_numpy() > h[k]]
        if len(hit):
            days.append(hit[0])
    return min(days) if days else None


# ---------------------------------------------------------------------------- evaluation
def evaluate(ind: pd.DataFrame, kind: str, sset: str, routine: str) -> dict:
    """One network/mode: ind with columns ds, case, day, <kind>_mb2/mb1."""
    groups = {(d, c): g.set_index("day").sort_index() for (d, c), g in ind.groupby(["ds", "case"])}
    st = {key: {col: statistics(s, routine) for col, s in signals(g, kind, sset).items()}
          for key, g in groups.items()}

    def thresholds(exclude=None):
        h = {}
        for d in fd.HEALTHY:
            if d == exclude:
                continue
            for col_stats in st[(d, "real")].values():
                for k, s in col_stats.items():
                    h[k] = max(h.get(k, 0.0), float(np.nanmax(s.to_numpy(), initial=0.0)))
        return {k: MARGIN * v for k, v in h.items()}

    res = {}
    fa, fa_days = 0, []
    for d in fd.HEALTHY:
        h = thresholds(d)
        days = [first_alarm(cs, h) for cs in st[(d, "real")].values()]
        days = [x for x in days if x is not None]
        fa += bool(days)
        fa_days += [f"{d}:{min(days).date()}"] if days else []
    res["FA"], res["fa_days"] = fa, " ".join(fa_days)
    h = thresholds()
    wrong = []
    for d, ev in EVENTS.items():
        ev = pd.Timestamp(ev).normalize()
        fb = FAULT_BEARING[d]
        days = [first_alarm(cs, h) for col, cs in st[(d, "real")].items() if fb in (None, col)]
        days = [x for x in days if x is not None]
        res[f"lead_{d}"] = (ev - min(days)).days if days else np.nan
        if fb is not None:  # alarm on the healthy bearing before the event (would point to the wrong bearing)
            w = [first_alarm(cs, h, before=ev) for col, cs in st[(d, "real")].items() if col != fb]
            w = [x for x in w if x is not None]
            if w:
                wrong.append(f"{d}:{min(w).date()}")
    res["wrong"] = " ".join(wrong)
    # injected ramps: "A" constant offset, "L" load-proportional
    for typ in RAMP_TYPES:
        for amp in fd.AMPS:
            hits, delays, n = 0, [], 0
            for d in fd.HEALTHY:
                h = thresholds(d)
                start0 = groups[(d, "real")].index[0]
                for b in ("mb2", "mb1"):
                    for s0 in fd.STARTS:
                        key = (d, f"{b}_{typ}{amp:g}_s{s0}")
                        if key not in st:
                            continue
                        n += 1
                        t0 = start0 + pd.Timedelta(days=s0)
                        t1 = t0 + pd.Timedelta(days=DET_WINDOW)
                        a = first_alarm(st[key][b], h, after=t0, before=t1)
                        a0 = first_alarm(st[(d, "real")][b], h, after=t0, before=t1)  # alarm without the ramp
                        if a is not None and (a0 is None or a < a0):
                            hits += 1
                            delays.append((a - t0).days)
            res[f"det{typ}{amp:g}"] = hits / n if n else np.nan
            res[f"del{typ}{amp:g}"] = float(np.median(delays)) if delays else np.nan
    res["h"] = " ".join(f"{k}={v:.2f}" for k, v in h.items())
    return res


def table(ind_all: pd.DataFrame, routines, kinds=("heat", "res", "fric", "off"), ssets=("bearings", "asym", "xreg")) -> pd.DataFrame:
    rows = []
    for (net, mode), ind in ind_all.groupby(["net", "mode"]):
        for kind in kinds:
            for sset in ssets:
                for r in routines:
                    rows.append({"net": net, "mode": mode, "ind": kind, "sig": sset, "routine": r,
                                 **evaluate(ind, kind, sset, r)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=fd.DEFAULT_OUT)
    ap.add_argument("--routines", default=",".join(ROUTINES))
    ap.add_argument("--final", action="store_true", help="only the final table (FINAL detectors)")
    a = ap.parse_args()
    ind = fd.load_indicators(a.out)
    if a.final:
        t = final_table(ind)
        t.to_csv(os.path.join(a.out, "final.csv"), index=False)
        print(fmt_final(t))
        return
    t = table(ind, a.routines.split(","))
    pd.set_option("display.width", 250)
    print(t.drop(columns=["fa_days", "h", "wrong"]).round(2).to_string(index=False))
    t.to_csv(os.path.join(a.out, "eval.csv"), index=False)



# ---------------------------------------------------------------------------- final table
FINAL = [  # (label, indicator, signal set, routine)
    ("D1 existing: heat, per bearing, rise+level CUSUM", "heat", "bearings", "cusum"),
    ("D2 existing: heat, mb2-mb1 +/-, rise+level CUSUM", "heat", "asym", "cusum"),
    ("D3 new: heat, xreg, fixed base, 14d mean, 7d persist", "heat", "xreg", "lvl14p7"),
    ("D4 new: free-run res, xreg, fixed base, 14d, 7d persist", "res", "xreg", "lvl14p7"),
    ("D5 new: free-run res, xreg, 14d mean - lagged base", "res", "xreg", "lag60"),
    ("D6 new: friction-gain tracker, mb2-mb1, lagged base", "fric", "asym", "lag60"),
]


def final_table(ind, mode="free", margins=(1.0, 1.15)):
    global MARGIN
    rows = []
    for m in margins:
        MARGIN = m
        for label, kind, sset, r in FINAL:
            for net in ("demo", "simple", "medium"):
                e = evaluate(ind[(ind.net == net) & (ind["mode"] == mode)], kind, sset, r)
                rows.append({"margin": m, "detector": label, "net": net, "FA": e["FA"],
                             "wrong": len(e["wrong"].split()) if e["wrong"] else 0,
                             **{f"lead{d}": e[f"lead_{d}"] for d in EVENTS},
                             **{f"{t}{a:g}": e[f"det{t}{a:g}"] for t in RAMP_TYPES for a in fd.AMPS},
                             "delA3": e["delA3"], "fa_days": e["fa_days"]})
    MARGIN = 1.0
    return pd.DataFrame(rows)


def fmt_final(t) -> str:
    out = []
    for m, g in t.groupby("margin"):
        out.append(f"threshold = {m:g} x max over the other healthy turbines")
        out.append(f"{'detector':58s} {'net':6s} FA wrong  lead77 lead53 lead27 | "
                   "offset ramp 1/2/3K  | load ramp 1/2/3K | delay3K")
        for r in g.itertuples():
            lead = " ".join(f"{v:6.0f}" if v == v else "     -" for v in (r.lead77, r.lead53, r.lead27))
            out.append(f"{r.detector:58s} {r.net:6s} {r.FA:2d} {r.wrong:4d}  {lead} | "
                       f"{r.A1:4.2f} {r.A2:4.2f} {r.A3:4.2f}      | {r.L1:4.2f} {r.L2:4.2f} {r.L3:4.2f}   | {r.delA3:5.0f}")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
