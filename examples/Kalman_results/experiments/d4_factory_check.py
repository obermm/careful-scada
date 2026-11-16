"""Check the library D4 detector (DetectorFactory "freerun") against the examples/ evaluation.

Same protocol as fault_eval.py (real series only, no ramps): shared fit on the pooled healthy
turbines (leave-one-out for the healthy ones, all six for the fault datasets 27/53/77), refit on
each turbine's first 90 days, thresholds = margin * max over the (other) healthy turbines.

    python d4_factory_check.py fit  [--net simple] [--workers 7]   # indicators -> OUT/ind_*.csv
    python d4_factory_check.py eval [--net simple]                  # FA / leads, compare with r2
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import fault_detection as fd
import network_search as ns

OUT = os.path.expanduser(os.environ.get("D4_OUT", "~/.cache/care_faultdet/d4_factory"))
EVENTS = {77: "2023-06-01", 53: "2023-12-26 16:40", 27: "2023-09-01"}
FAULT_BEARING = {77: fd.B2, 53: fd.B2, 27: None}


def make_detector(net: str, **kw):
    from care_scada.detector_factory import DetectorFactory
    from care_scada.modeling.thermal_network import ThermalNetworkSpec
    p = ns.parse(fd.NETS[net])
    spec = ThermalNetworkSpec(target=fd.B2, neighbours=[], sources=p["sources"], latent=p["latent"],
                              edges=p["edges"], measured_states=[fd.B1])
    return DetectorFactory.create("freerun", network=spec, cooling=sorted(p["cooling"]),
                                  const_heat=p["c"], temp_heat=p["g"], **kw)


def fit_pool(args):
    net, pool = args
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import warnings
    warnings.simplefilter("ignore")
    path = os.path.join(OUT, f"ind_{net}_{pool}.csv")
    if os.path.exists(path):
        return path
    fit_ds = [h for h in fd.HEALTHY if str(h) != str(pool)]
    det = make_detector(net).fit(pd.concat([fd.series(h) for h in fit_ds]))
    targets = fd.FAULT if pool == "all" else (int(pool),)
    frames = [det.indicators(fd.series(d)).assign(ds=d) for d in targets]
    pd.concat(frames).rename_axis("day").to_csv(path)
    return path


def evaluate(net: str, margin: float):
    det = make_detector(net, margin=margin)
    ind = {}
    for pool in ["all", *fd.HEALTHY]:
        f = pd.read_csv(os.path.join(OUT, f"ind_{net}_{pool}.csv"), index_col="day", parse_dates=True)
        for d, g in f.groupby("ds"):
            ind[int(d)] = g.drop(columns="ds")
    fa = []
    for d in fd.HEALTHY:
        det.calibrate([ind[h] for h in fd.HEALTHY if h != d])
        a = det.detect(ind[d])
        if not a.empty:
            fa.append(f"{d}:{a['day'].iloc[0].date()}")
    h = det.calibrate([ind[h] for h in fd.HEALTHY])
    leads = {}
    for d, ev in EVENTS.items():
        a = det.detect(ind[d])
        a = a if FAULT_BEARING[d] is None else a[a["monitored"] == FAULT_BEARING[d]]
        leads[d] = (pd.Timestamp(ev).normalize() - a["day"].iloc[0]).days if not a.empty else None
    return dict(net=net, margin=margin, FA=len(fa), fa_days=" ".join(fa), h=round(h["level"], 3),
                **{f"lead_{d}": v for d, v in leads.items()}), ind


def compare_r2(net: str, ind: dict):
    """Correlation / mean abs difference of the library residuals and the r2 res_* indicators."""
    rows = []
    for pool in ["all", *fd.HEALTHY]:
        path = os.path.join(fd.DEFAULT_OUT, f"ind_{net}_free_{pool}.csv.gz")
        r2 = pd.read_csv(path, parse_dates=["day"])
        r2 = r2[r2["case"] == "real"]
        for d, g in r2.groupby("ds"):
            g = g.set_index("day")
            for col, short in ((fd.B2, "mb2"), (fd.B1, "mb1")):
                a, b = ind[int(d)][col].align(g[f"res_{short}"], join="inner")
                ok = a.notna() & b.notna()
                rows.append(dict(ds=int(d), sensor=short, corr=round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3),
                                 mad=round(float((a[ok] - b[ok]).abs().mean()), 3)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "eval"])
    ap.add_argument("--net", default="simple")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "fit":
        tasks = [(a.net, p) for p in ["all", *map(str, fd.HEALTHY)]]
        with ProcessPoolExecutor(a.workers) as ex:
            for path in ex.map(fit_pool, tasks):
                print(path, flush=True)
    else:
        res = []
        for m in (1.0, 1.15):
            r, ind = evaluate(a.net, m)
            res.append(r)
        print(pd.DataFrame(res).to_string(index=False))
        cmp = compare_r2(a.net, ind)
        print(cmp.to_string(index=False))
        sys.exit(0)
