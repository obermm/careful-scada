"""Main-bearing fault detection on CARE Wind Farm B with Kalman thermal networks.

Stage 1 ("run", SLURM): daily indicators per turbine and network
    * shared physics fitted on pooled healthy turbines (leave-one-turbine-out for every healthy
      turbine, all six for the fault datasets), refitted on each turbine's first CAL_DAYS
      (L2 prior towards the shared fit, as KalmanHeatDetector with calibration_days=90)
    * indicators (daily means, K):
        heat_*  Kalman-tracked steady-state excess (library track_heat, both bearings assimilated)
        res_*   free-run residual (measured - simulated, no bearing assimilated)
        fric_*, off_*  joint Kalman tracking of an extra friction gain (heat ~ rotor speed^2; steady
                excess at full load) and a load-independent constant heat (DetModel.track_load)
    * synthetic faults on the healthy turbines: a ramp of AMP K (reached after RAMP_DAYS, then held)
      added to ONE bearing temperature, starting at START days after the series start; "A" cases
      add a constant offset, "L" cases a load-proportional one (see inject)
Stage 2 ("eval", local): alarm routines on the saved indicators -> tables.

    python fault_detection.py run --tasks 0/5 --out DIR      # one shard of the fit tasks
    python fault_detection.py eval --out DIR
"""

import argparse
import functools
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import network_search as ns

NETS = {
    "demo": "mb2-out mb2-gb1 mb1-mb2 mb1-out mb1-gb1 | rot>mb2,mb1 P>mb2,mb1 | c g",
    "simple": "mb1-gb1 mb1-out mb2-gb1 mb2-out | -ws>mb1,mb2 rot2>mb1,mb2 |",
    "medium": "mb1-gb1 mb1-out mb2-gb1 mb2-out | -ws>mb1,mb2 P>mb1,mb2 rot2>mb1,mb2 |",
}
FIT_MODES = ("free", "kalman")   # physics by free-run MSE (+ noise by NLL) | everything by Kalman NLL
HEALTHY, FAULT = ns.HEALTHY, ns.FAULT
B2, B1 = ns.TARGET, ns.MB1
SHORT = {B2: "mb2", B1: "mb1"}
CAL_DAYS = 90
PRIOR_W = 1.0
AMPS = (1.0, 2.0, 3.0)
STARTS = (120, 200, 280)
RAMP_DAYS = 30
LOAD_TAU_H = 3.0
DEFAULT_OUT = os.path.expanduser("~/.cache/care_faultdet/r2")


# ---------------------------------------------------------------------------- model
@functools.lru_cache(None)
def _det_class():
    import jax
    from care_scada.modeling.kalman_forecaster import KalmanThermalForecaster as Base
    ns._init_worker()
    Free = ns.FreeRunKalman

    class DetModel(Free):
        """Network of network_search with a choice of fit: 'free' = physics by free-run MSE, then
        noise (q, r) by Kalman NLL with frozen physics; 'kalman' = all by NLL (library)."""

        def __init__(self, fit_mode="free", **kw):
            super().__init__(training="kalman", **kw)
            self.fit_mode, self._freeze = fit_mode, False

        def _filter(self, p, u, y, m):
            if self._freeze:
                p = {k: (v if k in ("log_q", "log_r") else jax.lax.stop_gradient(v)) for k, v in p.items()}
            return Base._filter(self, p, u, y, m)

        def _optimize(self, df, params, steps, learning_rate, prior=None):
            if self.fit_mode == "kalman" or self._freeze:
                return Base._optimize(self, df, params, steps, learning_rate, prior)
            return Free._optimize(self, df, params, steps, learning_rate, prior)

        def track_load(self, df, ss_std_c=0.05, ss_std_g=0.05, prior_std=5.0, load_col="rotor_speed_rpm_avg"):
            """Joint Kalman tracking of a constant heat c_i (load independent) and an extra friction
            gain g_i (heat g_i * s(t), s = (rotor speed / its 99th percentile)^2) per bearing, both
            random walks: dT/dt = A T + B u + c + g s(t). Returns the steady-state excess of both
            parts in K (friction part at full load): columns off_<bearing>, fric_<bearing>."""
            import jax.numpy as jnp
            from jax.experimental import enable_x64
            from care_scada.modeling.kalman_forecaster import DT_H, _pad
            with enable_x64():
                p = self.params
                A, B = self._system(p)
                n, m = len(self.states_), len(self.sensors_)
                col_c = len(self.sinks_) + len(self.sources_)
                J = -jnp.linalg.inv(A)
                gain = jnp.abs(jnp.diag(J))[:m]
                F, G = self._discretise(A, B.at[:, col_c].set(0.0))
                # Gamma = int_0^dt exp(A tau) dtau (response to a unit constant heat over one step)
                _, Gam = self._discretise(A, jnp.eye(n)[:, :m])
                N = n + 2 * m
                F0 = jnp.eye(N).at[:n, :n].set(F).at[:n, n:n + m].set(Gam)
                E = jnp.zeros((N, N)).at[:n, n + m:].set(Gam)
                G0 = jnp.zeros((N, G.shape[1])).at[:n].set(G)
                q_c, q_g = (ss_std_c / gain) ** 2 / 24.0, (ss_std_g / gain) ** 2 / 24.0
                Q = jnp.diag(jnp.concatenate([jnp.exp(p["log_q"]), q_c, q_g])) * DT_H
                R = jnp.exp(p["log_r"]) + 1e-6
                if load_col in self.sources_:
                    scale = float(self._src_scale[self.sources_.index(load_col)])
                else:
                    scale = float(np.nanpercentile(df[load_col].to_numpy(float), 99))

                def step(carry, inp):
                    x, P = carry
                    u_t, y_t, m_t, s_t = inp
                    Ft = F0 + s_t * E
                    x, P = Ft @ x + G0 @ u_t, Ft @ P @ Ft.T + Q
                    for j in range(m):
                        sj = P[j, j] + R[j]
                        K = m_t[j] * P[:, j] / sj
                        x, P = x + K * (y_t[j] - x[j]), P - jnp.outer(K, P[j])
                    return (x, 0.5 * (P + P.T)), x

                run = jax.jit(lambda z0, P0, u, y, mk, s: jax.lax.scan(step, (z0, P0), (u, y, mk, s)))
                s_all = np.clip(df[load_col].to_numpy(float) / scale, 0.0, None) ** 2
                c = (p["c"] if self.const_heat else jnp.zeros(m))
                g = jnp.zeros(m)
                P_c = jnp.diag((prior_std / gain) ** 2)
                P_g = jnp.diag((prior_std / gain) ** 2)
                out = np.full((len(df), 2 * m), np.nan)
                Jm = np.asarray(J)[:m, :m]
                for sq in self._sequences(df):
                    y, mk = sq["y"], sq["m"]
                    s = np.full(len(y), np.nan)
                    s[sq["pos"]] = s_all[sq["rows"]]
                    s = pd.Series(s).ffill().fillna(0.0).to_numpy()
                    first = int(np.argmax(mk.all(1)))
                    x0 = -jnp.linalg.solve(A, B.at[:, col_c].set(0.0) @ sq["u"][first] + jnp.zeros(n).at[:m].set(c))
                    x0 = x0.at[:m].set(np.where(mk[first] > 0, y[first], np.mean(y[mk > 0])))
                    P0 = jnp.zeros((N, N)).at[:n, :n].set(jnp.eye(n)).at[n:n + m, n:n + m].set(P_c) \
                        .at[n + m:, n + m:].set(P_g)
                    k, pad = len(y), -len(y) % 2048
                    (z, P), zs = run(jnp.concatenate([x0, c, g]), P0, jnp.asarray(_pad(sq["u"], pad)),
                                     jnp.asarray(_pad(y, pad, 0.0)), jnp.asarray(_pad(mk, pad, 0.0)),
                                     jnp.asarray(_pad(s, pad, 0.0)))
                    zs = np.asarray(zs)[:k]
                    out[sq["rows"]] = np.column_stack([zs[:, n:n + m] @ Jm.T, zs[:, n + m:] @ Jm.T])[sq["pos"]]
                    c, g = jnp.asarray(zs[-1, n:n + m]), jnp.asarray(zs[-1, n + m:])
                    P_c = P[n:n + m, n:n + m] + jnp.diag(q_c) * 24.0
                    P_g = P[n + m:, n + m:] + jnp.diag(q_g) * 24.0
            cols = [f"off_{SHORT[c_]}" for c_ in self.sensors_] + [f"fric_{SHORT[c_]}" for c_ in self.sensors_]
            return pd.DataFrame(out, index=df.index, columns=cols)

        def fit(self, train_df, target_col):
            if self.fit_mode == "kalman":
                self.learning_rate = 0.02
            super().fit(train_df, target_col)
            if self.fit_mode == "free":
                from jax.experimental import enable_x64
                with enable_x64():
                    self._freeze = True
                    self.params = self._optimize(train_df, self.params, 300, 0.03)
                    self._freeze = False
            return self

    return DetModel


def make_model(net_name, fit_mode):
    p = ns.parse(NETS[net_name])
    from care_scada.modeling.thermal_network import ThermalNetworkSpec
    spec = ThermalNetworkSpec(target=B2, neighbours=[], sources=p["sources"], latent=p["latent"],
                              edges=p["edges"], measured_states=[B1])
    steps = 20 if os.environ.get("FD_QUICK") else 800
    return _det_class()(fit_mode=fit_mode, network=spec, const_heat=p["c"], temp_heat=p["g"],
                        cooling=p["cooling"], steps=steps)


# ---------------------------------------------------------------------------- data
def series(d) -> pd.DataFrame:
    tr, te = ns._data()[d]
    return pd.concat([tr, te])


def inject(df, col, amp, start_day, load=False):
    """Ramp 0 -> amp K over RAMP_DAYS from start_day on, then held. load=True: the added temperature
    is proportional to friction heat, (rotor speed)^2 low-passed with LOAD_TAU_H, normalised to mean
    1 over the series (same mean amplitude, but zero at standstill and up to ~2-3 amp at full load)."""
    t0 = df.index[0] + pd.Timedelta(days=start_day)
    frac = ((df.index - t0) / pd.Timedelta(days=RAMP_DAYS)).to_numpy(float)
    add = amp * np.clip(frac, 0.0, 1.0)
    if load:
        s = df["rotor_speed_rpm_avg"].clip(lower=0) ** 2
        s = s.ewm(halflife=pd.Timedelta(hours=LOAD_TAU_H * np.log(2)), times=df.index).mean()
        add = add * (s / s.mean()).fillna(0.0).to_numpy()
    out = df.copy()
    out[col] = out[col] + add
    return out


def daily_indicators(model, df):
    heat = model.track_heat(df)
    res = df[[B2, B1]] - model.predict_all(df)[[B2, B1]]
    ind = pd.DataFrame({f"heat_{SHORT[c]}": heat[c] for c in (B2, B1)} |
                       {f"res_{SHORT[c]}": res[c] for c in (B2, B1)})
    ind = ind.join(model.track_load(df))
    return ind.groupby(ind.index.floor("D")).mean()


def _params(model):
    return {k: {(" ".join(map(str, kk)) if isinstance(kk, tuple) else kk): round(float(v), 5) for kk, v in d.items()}
            for k, d in model.thermal_parameters_.items()}


def run_task(task, out_dir):
    """task = (net, fit_mode, pool): pool 'all' -> fault datasets, pool d -> healthy turbine d (LOO)."""
    import warnings
    warnings.simplefilter("ignore")
    net, mode, pool = task
    path = os.path.join(out_dir, f"ind_{net}_{mode}_{pool}.csv.gz")
    if os.path.exists(path):
        return path
    t0 = time.time()
    fit_ds = [h for h in HEALTHY if str(h) != str(pool)]
    model = make_model(net, mode).fit(pd.concat([series(h) for h in fit_ds]), B2)
    info = {"task": task, "fit_s": round(time.time() - t0, 1),
            "shared": _params(model)}
    targets = FAULT if pool == "all" else (int(pool),)
    frames = []
    for d in targets:
        df = series(d)
        cal = df[df.index < df.index[0] + pd.Timedelta(days=CAL_DAYS)]
        m = model.refit(cal, prior_weight=PRIOR_W)
        info[f"local_{d}"] = _params(m)
        cases = [("real", None, 0.0, 0, False)]
        if d in HEALTHY:
            cases += [(f"{SHORT[c]}_{'L' if ld else 'A'}{a:g}_s{s}", c, a, s, ld)
                      for ld in (False, True) for c in (B2, B1) for a in AMPS for s in STARTS]
            if os.environ.get("FD_QUICK"):
                cases = cases[:2]
        for name, col, amp, start, ld in cases:
            x = df if col is None else inject(df, col, amp, start, ld)
            ind = daily_indicators(m, x)
            ind.insert(0, "case", name)
            ind.insert(0, "ds", d)
            frames.append(ind)
    out = pd.concat(frames)
    out.index.name = "day"
    out.to_csv(path)
    info["total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "fit_info.jsonl"), "a") as f:
        f.write(json.dumps(info, default=str) + "\n")
    return path


def all_tasks():
    return [(n, m, p) for n in NETS for m in FIT_MODES for p in ["all", *map(str, HEALTHY)]]


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    i, n = map(int, a.tasks.split("/"))
    tasks = all_tasks()[i::n]
    with ProcessPoolExecutor(a.workers or len(tasks)) as ex:
        for t, r in zip(tasks, ex.map(functools.partial(_safe, out_dir=a.out), tasks)):
            print(t, r, flush=True)


def _safe(task, out_dir):
    try:
        return run_task(task, out_dir)
    except Exception as e:  # noqa: BLE001
        import traceback
        return "ERROR " + traceback.format_exc()[-1500:]


def load_indicators(out_dir) -> pd.DataFrame:
    frames = []
    for net in NETS:
        for mode in FIT_MODES:
            for pool in ["all", *map(str, HEALTHY)]:
                p = os.path.join(out_dir, f"ind_{net}_{mode}_{pool}.csv.gz")
                if os.path.exists(p):
                    f = pd.read_csv(p, parse_dates=["day"])
                    f["net"], f["mode"] = net, mode
                    frames.append(f)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "one"])
    ap.add_argument("--tasks", default="0/1")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--task", default="simple,free,2")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        os.makedirs(a.out, exist_ok=True)
        net, mode, pool = a.task.split(",")
        print(_safe((net, mode, pool), a.out))


if __name__ == "__main__":
    main()
