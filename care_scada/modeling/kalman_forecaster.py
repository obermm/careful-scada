"""Kalman-filter thermal network forecaster (JAX, optional dependencies ``jax``, ``optax``).

The network is a ``ThermalNetworkSpec`` (the same as for the PCRNN). Its state nodes are latent
true temperatures; the target and the ``measured_states`` are observed through noisy sensors,
latent nodes are not. Every state i follows

    dT_i/dt = sum_j k_ij (T_j - T_i) + sum_l beta_il u_l + c_i + gamma_i (T_out - 10)

over the edges and source links of the spec, with rates k = G / C_i per unit heat capacity, load
sources u (scaled to their 99th percentile, optionally squared) and, for measured states, a
constant heat c_i and an outside-temperature heat term gamma_i. The system is discretised exactly
on the 10-min grid and run as a Kalman filter; the target sensor is never assimilated at
prediction time, so a fault shows up as a residual.
"""

import copy
import functools
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax.experimental import enable_x64
from jax.scipy.linalg import expm
import optax

from care_scada.model_factory import TimeSeriesModel
from care_scada.modeling.neural_forecasters import _ensure_jax_backend
from care_scada.modeling.thermal_network import ThermalNetworkSpec

DT_H = 1.0 / 6.0
STEP_NS = 10 * 60 * 10**9


def _x64(method):
    """Run a method in double precision (needed by expm and the filter recursions)."""
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        with enable_x64():
            return method(*args, **kwargs)
    return wrapper


def _kalman(F, G, Q, R, x0, P0, u, y, m):
    """Kalman filter; the first len(R) states are observed, sequential scalar updates masked by m."""
    def step(carry, inp):
        x, P = carry
        u_t, y_t, m_t = inp
        x, P = F @ x + G @ u_t, F @ P @ F.T + Q
        nll = 0.0
        for j in range(R.shape[0]):
            s = P[j, j] + R[j]
            e = y_t[j] - x[j]
            K = m_t[j] * P[:, j] / s
            x, P = x + K * e, P - jnp.outer(K, P[j])
            nll = nll + m_t[j] * 0.5 * (jnp.log(2 * jnp.pi * s) + e ** 2 / s)
        return (x, 0.5 * (P + P.T)), (nll, x)
    return jax.lax.scan(step, (x0, P0), (u, y, m))


def _pad(a: np.ndarray, n: int, value: Optional[float] = None) -> np.ndarray:
    """Append n rows: copies of the last row, or a constant value."""
    tail = np.repeat(a[-1:], n, 0) if value is None else np.full((n,) + a.shape[1:], value)
    return np.concatenate([a, tail])


class KalmanThermalForecaster(TimeSeriesModel):
    """
    Latent-state lumped heat network fitted by Kalman-filter maximum likelihood ("kalman", the
    target is assimilated while fitting) or by the free-run MSE of the unassimilated target
    ("observer").

    The network is defined as for the PCRNN: pass a complete ``ThermalNetworkSpec`` as
    ``network``, or override the ``thermal_networks`` config entry of the target with
    ``neighbours``, ``sources`` (dict, or a list of linear sources), ``latent``, ``edges`` and
    ``measured_states`` (measured columns modelled as states that are coupled to the target and
    share its neighbours and sources). Sensors unavailable in the training data are dropped.

    :param const_heat: constant heat c per measured state
    :param temp_heat: heat term gamma * (T_out - 10) per measured state
    :param training: "observer" or "kalman"
    :param chunk_len: training sequences are cut into chunks of this many 10-min steps
    :param burn_in: steps per chunk excluded from the observer loss
    """

    def __init__(self, network: Optional[ThermalNetworkSpec] = None,
                 neighbours: Optional[Sequence[str]] = None, sources=None,
                 latent: Optional[Sequence[str]] = None, edges: Optional[Sequence[tuple]] = None,
                 measured_states: Optional[Sequence[str]] = None, const_heat: bool = True,
                 temp_heat: bool = True, outside: str = "outside_temp_c_avg",
                 training: str = "observer", chunk_len: int = 432, burn_in: int = 36,
                 steps: int = 600, learning_rate: float = 0.02, max_gap_minutes: float = 60.0,
                 verbose: bool = False):
        if training not in ("kalman", "observer"):
            raise ValueError("training must be 'kalman' or 'observer'")
        overrides = dict(neighbours=neighbours, sources=sources, latent=latent, edges=edges,
                         measured_states=measured_states)
        if network is not None and any(v is not None for v in overrides.values()):
            raise ValueError("Pass either a ThermalNetworkSpec or neighbours/sources/latent/edges/"
                             "measured_states, not both.")
        self.network, self._overrides, self.outside = network, overrides, outside
        self.const_heat, self.temp_heat, self.training = const_heat, temp_heat, training
        self.chunk_len, self.burn_in, self.steps = chunk_len, burn_in, steps
        self.learning_rate, self.max_gap_minutes, self.verbose = learning_rate, max_gap_minutes, verbose
        self.reset()

    def reset(self):
        self.params = None
        self.target_col = None
        self.network_ = None
        self.train_losses = []

    # ------------------------------------------------------------------ data
    def _resolve_columns(self, df: pd.DataFrame):
        """Network for the target (spec or config + overrides), restricted to df, as masks."""
        if self.network is not None and self.network.target != self.target_col:
            raise ValueError(f"Network target '{self.network.target}' differs from target_col "
                             f"'{self.target_col}'.")
        spec = self.network or ThermalNetworkSpec.from_config(self.target_col, **self._overrides)
        spec = self.network_ = spec.restrict_to(df.columns)
        self.states_ = spec.states                                  # measured first, latent last
        self.sensors_ = [spec.target] + spec.measured_states
        self.sinks_ = spec.observed
        links = spec.source_links
        self._src = list(dict.fromkeys((name, t) for name, _, t in links))  # input columns
        self.sources_ = [name for name, _ in self._src]
        self._src_names = [f"{name}²" if t == "square" else name for name, t in self._src]
        self._square = np.array([t == "square" for _, t in self._src], dtype=bool)
        idx = {n: i for i, n in enumerate(self.states_)}
        n = len(self.states_)
        self._m_sink, self._m_couple = np.zeros((n, len(self.sinks_))), np.zeros((n, n))
        self._m_src = np.zeros((n, len(self._src)))
        for a, b in spec.all_edges:
            if a in idx and b in idx:
                self._m_couple[idx[a], idx[b]] = self._m_couple[idx[b], idx[a]] = 1.0
            elif a in idx or b in idx:
                st, o = (a, b) if a in idx else (b, a)
                self._m_sink[idx[st], self.sinks_.index(o)] = 1.0
        for name, node, t in links:
            self._m_src[idx[node], self._src.index((name, t))] = 1.0

    def _inputs(self, df: pd.DataFrame) -> np.ndarray:
        """u = [sinks, scaled sources (squared where specified), 1, T_out - 10]."""
        n = len(df)
        src = np.maximum(df[self.sources_].to_numpy(float) / self._src_scale, 0.0)
        src[:, self._square] **= 2
        t_out = df[self.outside].to_numpy(float) - 10.0 if self.outside in df else np.zeros(n)
        return np.column_stack([df[self.sinks_].to_numpy(float), src, np.ones(n), t_out])

    def _sequences(self, df: pd.DataFrame):
        """Gap-free sequences on a regular 10-min grid (missing rows = masked observations)."""
        t = pd.DatetimeIndex(df.index).asi8
        dt = np.diff(t)
        breaks = np.flatnonzero((dt <= 0) | (dt > self.max_gap_minutes * 60e9)) + 1
        u_all = self._inputs(df)
        y_all = df.reindex(columns=self.sensors_).to_numpy(float)
        for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(df)]):
            pos = np.round((t[a:b] - t[a]) / STEP_NS).astype(int)
            if np.any(np.diff(pos) <= 0):  # not on the 10-min grid: consecutive steps
                pos = np.arange(b - a)
            u = np.full((pos[-1] + 1, u_all.shape[1]), np.nan)
            y = np.full((pos[-1] + 1, y_all.shape[1]), np.nan)
            u[pos], y[pos] = u_all[a:b], y_all[a:b]
            u = pd.DataFrame(u).ffill().fillna(pd.Series(self._u_fill)).to_numpy()
            yield dict(u=u, y=np.nan_to_num(y), m=(~np.isnan(y)).astype(float),
                       rows=np.arange(a, b), pos=pos)

    # ------------------------------------------------------------------ model
    def _init_params(self):
        n, m, n_s, n_src = len(self.states_), len(self.sensors_), len(self.sinks_), len(self.sources_)
        return {"log_k_sink": jnp.full((n, n_s), np.log(0.05)),
                "log_k_couple": jnp.full((n, n), np.log(0.05)),
                "log_beta": jnp.full((n, n_src), np.log(0.1)),
                "c": jnp.zeros(m), "gamma": jnp.zeros(m),
                "log_q": jnp.full(n, np.log(0.05)),   # process noise, K^2/h
                "log_r": jnp.full(m, np.log(0.01))}   # sensor noise, K^2

    def _system(self, p):
        """Continuous-time dx/dt = A x + B u (rates and gains outside the spec are masked out)."""
        n, m = len(self.states_), len(self.sensors_)
        ks = jnp.exp(p["log_k_sink"]) * self._m_sink
        kc = jnp.exp(p["log_k_couple"]) * self._m_couple
        A = kc - jnp.diag(ks.sum(1) + kc.sum(1))
        c = jnp.zeros(n).at[:m].set(p["c"]) if self.const_heat else jnp.zeros(n)
        gamma = jnp.zeros(n).at[:m].set(p["gamma"]) if self.temp_heat else jnp.zeros(n)
        B = jnp.column_stack([ks, jnp.exp(p["log_beta"]) * self._m_src, c, gamma])
        return A, B

    @staticmethod
    def _discretise(A, B):
        n, m = B.shape
        E = expm(jnp.zeros((n + m, n + m)).at[:n, :n].set(A).at[:n, n:].set(B) * DT_H)
        return E[:n, :n], E[:n, n:]

    def _filter(self, p, u, y, m):
        """Per-step NLL and target estimate over one sequence (steady-state start)."""
        A, B = self._system(p)
        F, G = self._discretise(A, B)
        x0 = -jnp.linalg.solve(A, B @ u[0])
        Q, R = jnp.diag(jnp.exp(p["log_q"])) * DT_H, jnp.exp(p["log_r"]) + 1e-6
        _, (nll, x) = _kalman(F, G, Q, R, x0, jnp.eye(len(x0)) * 4.0, u, y, m)
        return nll, x[:, 0]

    # ------------------------------------------------------------------ API
    @_x64
    def fit(self, train_df: pd.DataFrame, target_col: str):
        _ensure_jax_backend()
        self.reset()
        self.target_col = target_col
        self._resolve_columns(train_df)
        scale = np.nanpercentile(np.abs(train_df[self.sources_].to_numpy(float)), 99, axis=0)
        self._src_scale = np.where(scale > 1e-8, scale, 1.0)
        self._u_fill = np.nan_to_num(np.nanmean(self._inputs(train_df), axis=0))
        self.params = self._optimize(train_df, self._init_params(), self.steps, self.learning_rate)
        return self

    @_x64
    def refit(self, df: pd.DataFrame, prior_weight: float = 1.0, steps: int = 300,
              learning_rate: float = 0.005) -> "KalmanThermalForecaster":
        """Copy re-optimised on df (e.g. one turbine's first months), L2-pulled to the current fit."""
        if self.params is None:
            raise RuntimeError("Model must be fitted before calling refit.")
        new = copy.copy(self)
        new.train_losses = []
        new.params = new._optimize(df, self.params, steps, learning_rate,
                                   prior=(self.params, prior_weight))
        return new

    def _optimize(self, df, params, steps, learning_rate, prior=None):
        L = self.chunk_len
        chunks = [{k: s[k][a:a + L] for k in "uym"} for s in self._sequences(df)
                  for a in range(0, len(s["u"]), L)]
        chunks = [c for c in chunks if c["m"][:, 0].sum() >= self.burn_in + 6]
        if not chunks:
            raise ValueError("Not enough gap-free data to fit the model.")
        U, Y, M = (jnp.asarray(np.stack([_pad(c[k], L - len(c[k]), None if k == "u" else 0.0)
                                         for c in chunks])) for k in "uym")
        loss_mask = M[:, :, 0] * (jnp.arange(L) >= self.burn_in)
        filt = jax.vmap(self._filter, in_axes=(None, 0, 0, 0))

        def loss_fn(p):
            if self.training == "kalman":
                nll, _ = filt(p, U, Y, M)
                loss = nll.sum() / M.sum()
            else:  # the target is never assimilated
                _, est = filt(p, U, Y, M.at[:, :, 0].set(0.0))
                loss = jnp.sum(loss_mask * (est - Y[:, :, 0]) ** 2) / loss_mask.sum()
            loss = loss + self._penalty(p, U)
            if prior is not None:
                p0, w = prior
                sq = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b) ** 2), p, p0))
                loss = loss + w * sum(sq) / sum(np.size(v) for v in jax.tree_util.tree_leaves(p0))
            return loss

        opt = optax.chain(optax.clip_by_global_norm(10.0),
                          optax.adam(optax.cosine_decay_schedule(learning_rate, steps, alpha=0.05)))
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
            if self.verbose and (i % 100 == 0 or i == steps - 1):
                print(f"step {i:4d} loss {loss:.4f}", flush=True)
        return best[1]

    def _penalty(self, p, U):
        """Extra loss term on the parameters and training inputs (hook for subclasses)."""
        return 0.0

    @_x64
    def predict(self, test_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if self.params is None:
            raise RuntimeError("Model must be fitted before calling predict.")
        out = np.full(len(test_df), np.nan)
        run = jax.jit(self._filter)
        for s in self._sequences(test_df):
            n = len(s["u"])
            pad = -n % 2048  # few distinct shapes -> few compilations
            m = s["m"].copy()
            m[:, 0] = 0.0  # never assimilate the target
            _, est = run(self.params, jnp.asarray(_pad(s["u"], pad)),
                         jnp.asarray(_pad(s["y"], pad, 0.0)), jnp.asarray(_pad(m, pad, 0.0)))
            out[s["rows"]] = np.asarray(est)[s["pos"]]
        return pd.DataFrame(out, index=test_df.index, columns=[target_col])

    @_x64
    def track_heat(self, df: pd.DataFrame, ss_std_per_day: float = 0.05,
                   prior_ss_std: float = 5.0) -> pd.DataFrame:
        """
        Steady-state excess temperature (-A^-1) c of every measured state in K, with all sensors
        assimilated and the constant heats c tracked as random walks (std ss_std_per_day of
        the excess per sqrt(day), persisting across gaps). Extra heat from a fault shows up
        as a rise of the excess.
        """
        p = self.params
        A, B = self._system(p)
        n, m, col_c = len(self.states_), len(self.sensors_), len(self.sinks_) + len(self.sources_)
        J = -jnp.linalg.inv(A)
        gain = jnp.abs(jnp.diag(J))[:m]
        q_c = (ss_std_per_day / gain) ** 2 / 24.0  # per hour
        # augmented state z = [T, c] (c of the measured states)
        Az = jnp.zeros((n + m, n + m)).at[:n, :n].set(A).at[:m, n:].set(jnp.eye(m))
        F, G = self._discretise(Az, jnp.zeros((n + m, B.shape[1])).at[:n].set(B.at[:, col_c].set(0.0)))
        Q = jnp.diag(jnp.concatenate([jnp.exp(p["log_q"]), q_c])) * DT_H
        R = jnp.exp(p["log_r"]) + 1e-6
        run = jax.jit(lambda z0, P0, u, y, m_: _kalman(F, G, Q, R, z0, P0, u, y, m_))

        c = p["c"] if self.const_heat else jnp.zeros(m)
        P_c = jnp.diag((prior_ss_std / gain) ** 2)
        out = np.full((len(df), m), np.nan)
        for s in self._sequences(df):
            y, mk = s["y"], s["m"]
            first = int(np.argmax(mk.all(1)))
            x0 = -jnp.linalg.solve(A, B @ s["u"][first])  # latent nodes start in steady state
            x0 = x0.at[:m].set(np.where(mk[first] > 0, y[first], np.mean(y[mk > 0])))
            P0 = jnp.zeros((n + m, n + m)).at[:n, :n].set(jnp.eye(n)).at[n:, n:].set(P_c)
            k, pad = len(y), -len(y) % 2048
            (z, P), (_, zs) = run(jnp.concatenate([x0, c]), P0, jnp.asarray(_pad(s["u"], pad)),
                                  jnp.asarray(_pad(y, pad, 0.0)), jnp.asarray(_pad(mk, pad, 0.0)))
            cs = np.asarray(zs)[:k, n:]
            out[s["rows"]] = (cs @ np.asarray(J)[:m, :m].T)[s["pos"]]
            c, P_c = jnp.asarray(cs[-1]), P[n:, n:] + jnp.diag(q_c) * 24.0  # one day of drift per gap
        return pd.DataFrame(out, index=df.index, columns=self.sensors_)

    @property
    @_x64
    def thermal_parameters_(self) -> dict:
        """
        Fitted network as rates per unit heat capacity of the receiving state:

        * ``couplings``: {(state_i, state_j): k_ij in 1/h} (heating of i by j, both directions)
        * ``sink_rates``: {(state, observed node): k in 1/h}
        * ``source_gains``: {(source, state): K/h at full load (99th percentile of the source;
          squared sources are named "column²")}
        * ``const_heat``, ``outside_heat``: {measured state: c in K/h}, {measured state: gamma}
        * ``time_constants_h``: {state: -1 / A_ii}
        * ``steady_state_rise_K``: {(source, state): steady temperature rise at full load}
        * ``sensor_noise_K``: {measured state: standard deviation of its sensor noise}
        """
        if self.params is None:
            raise RuntimeError("Model must be fitted first.")
        A, B = (np.asarray(M) for M in self._system(self.params))
        S, M, n_s, n_src = self.states_, self.sensors_, len(self.sinks_), len(self.sources_)
        idx = {a: i for i, a in enumerate(S)}
        rise = np.linalg.solve(-A, B)
        links = [(self._src_names[j], node, n_s + j) for name, node, t in self.network_.source_links
                 for j in [self._src.index((name, t))]]
        return {
            "couplings": {(a, b): float(A[i, j]) for i, a in enumerate(S) for j, b in enumerate(S)
                          if self._m_couple[i, j]},
            "sink_rates": {(a, o): float(B[i, k]) for i, a in enumerate(S) for k, o in enumerate(self.sinks_)
                           if self._m_sink[i, k]},
            "source_gains": {(name, a): float(B[idx[a], k]) for name, a, k in links},
            "const_heat": {a: float(B[i, n_s + n_src]) for i, a in enumerate(M)},
            "outside_heat": {a: float(B[i, n_s + n_src + 1]) for i, a in enumerate(M)},
            "time_constants_h": {a: float(-1 / A[i, i]) for i, a in enumerate(S)},
            "steady_state_rise_K": {(name, a): float(rise[idx[a], k]) for name, a, k in links},
            "sensor_noise_K": {a: float(np.sqrt(np.exp(self.params["log_r"][i]))) for i, a in enumerate(M)},
        }

    @_x64
    def summary(self) -> Dict[str, float]:
        """Time constants (h), conductances (1/h) and heat terms (K/h) of the target node."""
        A, B = (np.asarray(M) for M in self._system(self.params))
        names = self.sinks_ + self._src_names + ["const", "outside_temp"]
        res = {f"tau_h[{s}]": float(-1 / A[i, i]) for i, s in enumerate(self.states_)}
        res.update({f"k[{s}]": float(A[0, i]) for i, s in enumerate(self.states_) if i})
        res.update({f"B[{s}]": float(B[0, j]) for j, s in enumerate(names)})
        return res
