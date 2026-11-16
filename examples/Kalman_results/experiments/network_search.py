"""Thermal network search for BOTH main bearings of CARE Wind Farm B (free-run Kalman model).

Model: the library KalmanThermalForecaster network (target = main bearing 2, measured state =
main bearing 1, optional latent nodes), but trained and evaluated as a pure simulation: neither
bearing sensor is assimilated; the states are driven only by the observed boundary temperatures
and the operating signals. At the start of every gap-free sequence (and training chunk) the
bearing states are set to their measurements and the latent nodes to their steady state given
those (one-point initialisation). Loss / metric: RMSE of both bearings.

Protocol: every healthy dataset (2, 21, 52, 74, 82, 83) is fitted on its CARE train part and
evaluated on its test part. Reported: mean test RMSE over datasets per bearing, mean of both.

Network notation (one string, also the log entry):
    "mb2-out mb2-gb1 mb1-mb2 nac-out | rot>mb1,mb2 P2>nac | c g"
  edges        a-b: conductance between two nodes. Measured columns by alias (ALIASES); any
               other name is a latent node (e.g. nac, hub, hous).
  sources      s>n1,n2: source s heats nodes n1, n2 (s2 = squared source); -s>n: s cools n
               (negative gain, e.g. wind cooling).
  flags        c: constant heat per bearing, g: gamma (T_out - 10) per bearing.

Usage:
    python network_search.py run --nets nets.txt --out results.jsonl [--workers 32]
    python network_search.py one "<network>"
"""

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ALIASES = {
    "mb1": "main_bearing_temp1_c_avg", "mb2": "main_bearing_temp2_c_avg",
    "out": "outside_temp_c_avg", "oil": "gearbox_oil_temp1_c_avg",
    "gb1": "gearbox_bearing_temp1_c_avg", "gb2": "gearbox_bearing_temp2_c_avg",
    "gb3": "gearbox_bearing_temp3_c_avg", "gb4": "gearbox_bearing_temp4_c_avg",
    "gen1": "gen_bearing_temp1_c_avg", "gen2": "gen_bearing_temp2_c_avg",
    "P": "active_power_kWh", "rot": "rotor_speed_rpm_avg", "grpm": "gen_rpm_avg",
    "ws": "wind_speed1_ms_avg", "pitch": "pitch_angle_deg_avg",
    "tdev": "outside_dev_c",  # derived: |T_out - 10 C| (tdev2 = (T_out - 10)^2)
}
DERIVED = {"outside_dev_c": lambda df: (df["outside_temp_c_avg"] - 10.0).abs()}


def add_derived(df):
    return df.assign(**{k: f(df) for k, f in DERIVED.items()})
TARGET, MB1 = ALIASES["mb2"], ALIASES["mb1"]
HEALTHY = (2, 21, 52, 74, 82, 83)
FAULT = (27, 53, 77)
DATA_CACHE = os.path.expanduser("~/.cache/care_kalman/data_mon.pkl")


def parse(net: str) -> dict:
    """Network string -> dict(edges, sources, latent, c, g)."""
    parts = [p.strip() for p in net.split("|")] + ["", ""]
    col = lambda a: ALIASES.get(a, a)  # noqa: E731
    edges = [tuple(col(x) for x in e.split("-")) for e in parts[0].split()]
    sources, cooling = {}, set()
    for s in parts[1].split():
        name, nodes = s.split(">")
        neg = name.startswith("-")
        name = name.lstrip("-")
        t = "square" if name.endswith("2") and name[:-1] in ALIASES else "linear"
        name = name[:-1] if t == "square" else name
        for n in nodes.split(","):
            sources.setdefault(col(name), []).append((col(n), t))
            if neg:
                cooling.add((col(name), col(n), t))
    flags = parts[2].split()
    nodes = {n for e in edges for n in e} | {n for v in sources.values() for n, _ in v}
    latent = sorted(n for n in nodes if n not in ALIASES.values())
    return dict(edges=edges, sources=sources, latent=latent, c="c" in flags, g="g" in flags,
                cooling=cooling)


def n_params(net: str) -> int:
    """Free physics parameters: 2 per state-state edge, 1 per sink edge / source link / heat term."""
    p = parse(net)
    states = {TARGET, MB1, *p["latent"]}
    k = sum(2 if (a in states and b in states) else 1 for a, b in p["edges"])
    return k + sum(len(v) for v in p["sources"].values()) + 2 * p["c"] + 2 * p["g"]


def build_model(net: str, **kw):
    from care_scada.modeling.thermal_network import ThermalNetworkSpec
    p = parse(net)
    spec = ThermalNetworkSpec(target=TARGET, neighbours=[], sources=p["sources"], latent=p["latent"],
                              edges=p["edges"], measured_states=[MB1])
    return FreeRunKalman(network=spec, const_heat=p["c"], temp_heat=p["g"], cooling=p["cooling"], **kw)


# ---------------------------------------------------------------------------- model
def _model_class():
    import functools
    import jax
    import jax.numpy as jnp
    import optax
    from jax.experimental import enable_x64
    from care_scada.modeling.kalman_forecaster import KalmanThermalForecaster, _pad

    class FreeRunKalman(KalmanThermalForecaster):
        """Library network simulated without assimilating any bearing sensor (output error fit)."""

        def __init__(self, seed: int = 0, cooling=(), mode: str = "free", chunk_len: int = 1008,
                     steps: int = 800, learning_rate: float = 0.03, **kw):
            """mode: "free" (loss on both bearings simulated freely) or "cross" (loss on each
            bearing simulated with the other one clamped to its measurement)."""
            super().__init__(chunk_len=chunk_len, steps=steps, learning_rate=learning_rate, **kw)
            self.seed, self.cooling, self.mode = seed, set(cooling), mode

        def _resolve_columns(self, df):
            super()._resolve_columns(df)
            idx = {n: i for i, n in enumerate(self.states_)}
            self._sign = np.ones_like(self._m_src)
            for name, node, t in self.cooling:
                if node in idx and (name, t) in self._src:
                    self._sign[idx[node], self._src.index((name, t))] = -1.0

        def _system(self, p):
            A, B = super()._system(p)
            n_s, n_src = len(self.sinks_), len(self._src)
            return A, B.at[:, n_s:n_s + n_src].multiply(self._sign)

        def _init_params(self):
            p = super()._init_params()
            if self.seed:
                rng = np.random.default_rng(self.seed)
                for k in ("log_k_sink", "log_k_couple", "log_beta"):
                    p[k] = p[k] + jnp.asarray(rng.normal(0, 1.0, p[k].shape))
            return p

        def _simulate(self, p, u, y, m, clamp=None):
            """Free-run estimates of all sensors (one-point initialisation from y[0]); sensors with
            clamp = 1 are set to their measurement at every observed step (perfect assimilation)."""
            A, B = self._system(p)
            F, G = self._discretise(A, B)
            ms = len(self.sensors_)
            Bu0 = B @ u[0]
            xss = -jnp.linalg.solve(A, Bu0)
            xM = jnp.where(m[0] > 0, y[0], xss[:ms])
            if A.shape[0] > ms:
                xL = -jnp.linalg.solve(A[ms:, ms:], A[ms:, :ms] @ xM + Bu0[ms:])
                x0 = jnp.concatenate([xM, xL])
            else:
                x0 = xM

            cl = jnp.zeros(ms) if clamp is None else jnp.asarray(clamp)

            def step(x, inp):
                u_t, y_t, m_t = inp
                x = F @ x + G @ u_t
                est = x[:ms]
                x = x.at[:ms].set(jnp.where(cl * m_t > 0, y_t, est))
                return x, est
            _, xs = jax.lax.scan(step, x0, (u[1:], y[1:], m[1:]))
            return jnp.concatenate([x0[None, :ms], xs])

        def _optimize(self, df, params, steps, learning_rate, prior=None):
            L = self.chunk_len
            chunks = [{k: s[k][a:a + L] for k in "uym"} for s in self._sequences(df)
                      for a in range(0, len(s["u"]), L)]
            chunks = [c for c in chunks if c["m"][:, 0].sum() >= self.burn_in + 6]
            U, Y, M = (jnp.asarray(np.stack([_pad(c[k], L - len(c[k]), None if k == "u" else 0.0)
                                             for c in chunks])) for k in "uym")
            W = M * (jnp.arange(L) >= self.burn_in)[None, :, None]
            sim = jax.vmap(self._simulate, in_axes=(None, 0, 0, 0))
            ms = len(self.sensors_)
            clamps = [None] if self.mode == "free" else [tuple(float(j != i) for j in range(ms)) for i in range(ms)]
            sims = [jax.vmap(functools.partial(self._simulate, clamp=c), in_axes=(None, 0, 0, 0)) for c in clamps]

            def loss_fn(p):
                if self.mode == "free":
                    est = sims[0](p, U, Y, M)
                else:  # sensor i from the run in which all other sensors are clamped
                    est = jnp.stack([sims[i](p, U, Y, M)[..., i] for i in range(ms)], axis=-1)
                loss = jnp.sum(W * (est - Y) ** 2) / W.sum()
                if prior is not None:  # L2 pull towards prior parameters (as in the library refit)
                    p0, w = prior
                    sq = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b) ** 2), p, p0))
                    loss = loss + w * sum(sq) / sum(np.size(v) for v in jax.tree_util.tree_leaves(p0))
                return loss

            opt = optax.chain(optax.clip_by_global_norm(10.0),
                              optax.adam(optax.cosine_decay_schedule(learning_rate, steps, alpha=0.02)))
            state = opt.init(params)

            @jax.jit
            def update(params, state):
                loss, g = jax.value_and_grad(loss_fn)(params)
                upd, state = opt.update(g, state, params)
                return optax.apply_updates(params, upd), state, loss

            best = (np.inf, params)
            for i in range(steps):
                params, state, loss = update(params, state)
                loss = float(loss)
                self.train_losses.append(loss)
                if np.isfinite(loss) and loss < best[0]:
                    best = (loss, params)
            return best[1]

        def predict_all(self, df, clamp=None):
            """Free-run estimates of both bearings (clamp: see _simulate), DataFrame with sensor columns."""
            import pandas as pd
            with enable_x64():
                out = np.full((len(df), len(self.sensors_)), np.nan)
                run = jax.jit(functools.partial(self._simulate, clamp=clamp))
                for s in self._sequences(df):
                    n = len(s["u"])
                    pad = -n % 1024
                    est = run(self.params, jnp.asarray(_pad(s["u"], pad)),
                              jnp.asarray(_pad(s["y"], pad, 0.0)), jnp.asarray(_pad(s["m"], pad, 0.0)))
                    out[s["rows"]] = np.asarray(est)[:n][s["pos"]]
            return pd.DataFrame(out, index=df.index, columns=self.sensors_)

    return FreeRunKalman


FreeRunKalman = None


def _init_worker():
    global FreeRunKalman
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    warnings.simplefilter("ignore")
    FreeRunKalman = _model_class()


# ---------------------------------------------------------------------------- evaluation
_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        with open(DATA_CACHE, "rb") as f:
            _DATA = {d: tuple(add_derived(x) for x in v) for d, v in pickle.load(f).items()}
    return _DATA


def fit_eval(net: str, d: int, seed: int = 0, val_frac: float = 0.0, **kw) -> dict:
    """Fit on the train part of dataset d (without its last val_frac, the validation part),
    RMSE per bearing on train, val and test."""
    if FreeRunKalman is None:
        _init_worker()
    tr, te = _data()[d]
    parts = []
    if val_frac:
        cut = int(len(tr) * (1 - val_frac))
        tr, va = tr.iloc[:cut], tr.iloc[cut:]
        parts = [("val", va)]
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_model(net, seed=seed, **kw).fit(tr, TARGET)
    res = {"net": net, "ds": d, "seed": seed, "fit_s": round(time.time() - t0, 1)}
    for part, df in [("train", tr)] + parts + [("test", te)]:
        est = model.predict_all(df)
        for col, a in ((TARGET, "mb2"), (MB1, "mb1")):
            r = (df[col] - est[col]).to_numpy()
            res[f"{part}_{a}"] = round(float(np.sqrt(np.nanmean(r ** 2))), 4)
        if part != "train":  # each bearing with the other one clamped to its measurement
            for col, a, cl in ((TARGET, "mb2", (0.0, 1.0)), (MB1, "mb1", (1.0, 0.0))):
                r = (df[col] - model.predict_all(df, clamp=cl)[col]).to_numpy()
                res[f"{part}_x{a}"] = round(float(np.sqrt(np.nanmean(r ** 2))), 4)
    res["loss"] = round(min(model.train_losses), 5)
    return res


def _task(args):
    net, d, seed, kw = args
    try:
        return fit_eval(net, d, seed, **kw)
    except Exception as e:  # noqa: BLE001
        return {"net": net, "ds": d, "seed": seed, "error": repr(e)[:300]}


def summarise(rows, sort=None):
    """Per network: mean test/train RMSE over datasets (best seed per dataset by train loss)."""
    import pandas as pd
    df = pd.DataFrame([r for r in rows if "error" not in r])
    if df.empty:
        return df
    df = df.sort_values("loss").groupby(["net", "ds"], as_index=False).first()
    df["test"] = (df.test_mb1 + df.test_mb2) / 2
    df["train"] = (df.train_mb1 + df.train_mb2) / 2
    agg = dict(test=("test", "mean"), test_mb1=("test_mb1", "mean"), test_mb2=("test_mb2", "mean"),
               train=("train", "mean"), worst=("test", "max"), n_ds=("ds", "count"), fit_s=("fit_s", "mean"))
    if "test_xmb1" in df:
        df["xtest"] = (df.test_xmb1 + df.test_xmb2) / 2
        agg = dict(xtest=("xtest", "mean"), **agg)
    if "val_mb1" in df:
        df["val"] = (df.val_mb1 + df.val_mb2) / 2
        agg = dict(val=("val", "mean"), val_worst=("val", "max"), **agg)
    if "val_xmb1" in df:
        df["xval"] = (df.val_xmb1 + df.val_xmb2) / 2
        agg = dict(xval=("xval", "mean"), **agg)
    g = df.groupby("net").agg(**agg)
    g["n_par"] = [n_params(n) for n in g.index]
    return g.sort_values(sort or ("val" if "val" in g else "test"))


def run(nets, out, workers, seeds=(0,), datasets=HEALTHY, kw=None):
    tasks = [(n, d, s, kw or {}) for n in nets for d in datasets for s in seeds]
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        for r in ex.map(_task, tasks, chunksize=1):
            rows.append(r)
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "one", "summary"])
    ap.add_argument("net", nargs="?")
    ap.add_argument("--nets", help="file with one network per line")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--datasets", default=",".join(map(str, HEALTHY)))
    ap.add_argument("--kw", default="{}", help="JSON model kwargs (steps, learning_rate, chunk_len)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    ds = [int(s) for s in a.datasets.split(",")]
    if a.mode == "summary":
        import pandas as pd
        pd.set_option("display.width", 250, "display.max_colwidth", 140, "display.max_rows", 500)
        rows = [json.loads(l) for f in a.out.split(",") for l in open(f)]
        print(summarise(rows).round(3).to_string())
        return
    nets = [a.net] if a.mode == "one" else [l.strip() for l in open(a.nets) if l.strip() and not l.startswith("#")]
    rows = run(nets, a.out, a.workers, seeds, ds, json.loads(a.kw))
    import pandas as pd
    pd.set_option("display.width", 250, "display.max_colwidth", 140)
    print(summarise(rows).round(3).to_string())


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
    main()


def log_table(rows, title, path, top=None, note=""):
    """Append a concise per-network summary to the results log."""
    s = summarise(rows)
    s = s[s.n_ds == len(HEALTHY)]
    if top:
        s = s.head(top)
    with open(path, "a") as f:
        f.write(f"\n## {title}\n")
        if note:
            f.write(note.rstrip() + "\n")
        f.write(f"{'val':>6} {'test':>6} {'t_mb1':>6} {'t_mb2':>6} {'npar':>4}  network\n")
        for n, r in s.iterrows():
            f.write(f"{r.val:6.3f} {r.test:6.3f} {r.test_mb1:6.3f} {r.test_mb2:6.3f} {int(r.n_par):4d}  {n}\n")
