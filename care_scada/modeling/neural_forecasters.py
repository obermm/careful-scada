"""Neural forecasting models (LSTM and physics-constrained RNN) implemented in JAX/Flax.

Both models are normal-behaviour models in the same spirit as the random forest in the
model factory: the target sensor at time t is predicted from a window of the *other* sensors
(t-window+1, ..., t). The target itself is never used as an input, so predictions do not
depend on ground truth in the test set.

Requires the optional dependencies ``jax``, ``flax`` and ``optax`` (``pip install .[deep]``).
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from care_scada.model_factory import TimeSeriesModel, feature_columns
from care_scada.modeling.thermal_network import TRANSFORMS, ThermalNetworkSpec


def _ensure_jax_backend():
    """Fall back to CPU if the default (e.g. CUDA) backend cannot be initialised."""
    try:
        jax.devices()
    except RuntimeError:
        jax.config.update("jax_platforms", "cpu")


class LSTMNet(nn.Module):
    """LSTM over a window of sensor values, followed by a linear read-out of the last step."""
    hidden_size: int

    @nn.compact
    def __call__(self, x):  # x: (batch, window, features)
        h = nn.RNN(nn.OptimizedLSTMCell(self.hidden_size))(x)
        return nn.Dense(1)(h[:, -1, :])[:, 0], 0.0


class _WindowedNeuralForecaster(TimeSeriesModel):
    """
    Shared training / prediction logic for window-based JAX models.

    Subclasses define the per-row input matrix (``_fit_inputs`` / ``_make_inputs``) and a flax
    module mapping windows (batch, window, n_inputs) to ``(standardised prediction, penalty)``;
    the training loss is the standardised MSE plus the penalty.

    By default, the columns from ``feature_columns`` (numeric, without the target, its other
    statistics and metadata) are standardised with training statistics
    and NaNs are replaced by the training mean (0 after standardisation). Rows at the start of a
    series are padded by repeating the first row, so every row receives a prediction.
    """

    def __init__(self, window: int = 36, hidden_size: int = 32, epochs: int = 5,
                 batch_size: int = 256, learning_rate: float = 1e-3, seed: int = 0,
                 verbose: bool = False):
        self.window = window
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.verbose = verbose
        self.reset()

    @abstractmethod
    def _build_module(self) -> nn.Module:
        """Return the flax module mapping (batch, window, n_inputs) -> (pred, penalty)."""

    def reset(self):
        self.params = None
        self.feature_cols: Optional[list] = None
        self.target_col: Optional[str] = None
        self._x_mean = self._x_std = None
        self._y_mean = self._y_std = None
        self.train_losses: list = []
        self._module = None

    # ---------------------------------------------------------------- input hooks
    def _fit_inputs(self, train_df: pd.DataFrame):
        """Determine input columns and their statistics from the training data."""
        self.feature_cols = feature_columns(train_df, self.target_col)
        if not self.feature_cols:
            raise ValueError("No numeric feature columns available besides the target.")
        feats = train_df[self.feature_cols].to_numpy(dtype=np.float32)
        self._x_mean = np.nan_to_num(np.nanmean(feats, axis=0))
        x_std = np.nan_to_num(np.nanstd(feats, axis=0))
        self._x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)

    def _required_columns(self) -> List[str]:
        return list(self.feature_cols)

    def _make_inputs(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row model inputs (n_rows, n_inputs) without NaNs."""
        return _standardise(df, self.feature_cols, self._x_mean, self._x_std)

    # ---------------------------------------------------------------- helpers
    def _padded(self, x: np.ndarray) -> np.ndarray:
        pad = np.repeat(x[:1], self.window - 1, axis=0)
        return np.concatenate([pad, x], axis=0)

    def _windows(self, x_padded, idx):
        """Gather windows ending at rows idx (indices refer to the unpadded array)."""
        return x_padded[idx[:, None] + jnp.arange(self.window)[None, :]]

    # ---------------------------------------------------------------- API
    def fit(self, train_df: pd.DataFrame, target_col: str):
        if train_df.empty:
            raise ValueError("Training data is empty.")
        if target_col not in train_df.columns:
            raise ValueError(f"target_col '{target_col}' not found in train_df columns.")
        _ensure_jax_backend()
        self.reset()
        self.target_col = target_col

        y = train_df[target_col].to_numpy(dtype=np.float32)
        if np.all(np.isnan(y)):
            raise ValueError("Target column contains only NaNs.")
        self._y_mean = float(np.nanmean(y))
        self._y_std = float(max(np.nanstd(y), 1e-8))
        self._fit_inputs(train_df)

        x_padded = jnp.asarray(self._padded(self._make_inputs(train_df)))
        y_std = (y - self._y_mean) / self._y_std
        train_idx = np.flatnonzero(~np.isnan(y_std))  # skip rows without a target
        y_std = jnp.asarray(np.nan_to_num(y_std))

        self._module = self._build_module()
        rng = np.random.default_rng(self.seed)
        params = self._module.init(jax.random.PRNGKey(self.seed),
                                   self._windows(x_padded, jnp.arange(1)))
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)
        module, window = self._module, self.window

        @jax.jit
        def train_step(params, opt_state, idx):
            def loss_fn(p):
                pred, penalty = module.apply(p, x_padded[idx[:, None] + jnp.arange(window)[None, :]])
                return jnp.mean((pred - y_std[idx]) ** 2) + penalty
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        batch = min(self.batch_size, len(train_idx))
        n_batches = len(train_idx) // batch
        for epoch in range(self.epochs):
            perm = rng.permutation(train_idx)[: n_batches * batch].reshape(n_batches, batch)
            losses = []
            for idx in perm:
                params, opt_state, loss = train_step(params, opt_state, jnp.asarray(idx))
                losses.append(loss)
            self.train_losses.append(float(jnp.mean(jnp.stack(losses))))
            if self.verbose:
                print(f"epoch {epoch + 1}/{self.epochs} - standardised train loss "
                      f"{self.train_losses[-1]:.4f}")
        self.params = params

    def predict(self, test_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if self.params is None:
            raise RuntimeError("Model must be fitted before calling predict.")
        if target_col != self.target_col:
            raise ValueError(f"Model was fitted on '{self.target_col}', not '{target_col}'.")
        missing = [c for c in self._required_columns() if c not in test_df.columns]
        if missing:
            raise ValueError(f"Feature columns missing in test_df: {missing}")

        x_padded = jnp.asarray(self._padded(self._make_inputs(test_df)))
        module = self._module
        apply = jax.jit(lambda p, x: module.apply(p, x)[0])
        n = len(test_df)
        out = []
        for start in range(0, n, 4096):
            idx = jnp.arange(start, min(start + 4096, n))
            out.append(np.asarray(apply(self.params, self._windows(x_padded, idx))))
        preds = np.concatenate(out) * self._y_std + self._y_mean if n else np.empty(0)
        return pd.DataFrame(preds, index=test_df.index, columns=[target_col])


def _standardise(df: pd.DataFrame, cols: Sequence[str], mean, std) -> np.ndarray:
    if len(cols) == 0:
        return np.zeros((len(df), 0), dtype=np.float32)
    x = df.reindex(columns=list(cols)).to_numpy(dtype=np.float32)
    return np.nan_to_num((x - mean) / std, nan=0.0).astype(np.float32)


class LSTMForecaster(_WindowedNeuralForecaster):
    """
    LSTM normal-behaviour model: target_t from a window of the other sensors.

    Defaults were tuned by random search on chronological validation splits (last 20% of the
    training data) of the normal-behaviour datasets 83 and 86 of Wind Farm B
    (main_bearing_temp2_c_avg), with the best configurations checked over three seeds. The
    window length mattered most: all top configurations use 12 h (72 steps), which matches the
    thermal time constants of hours; a small network was as accurate as larger ones and the
    most stable across seeds.
    """

    def __init__(self, window: int = 72, hidden_size: int = 8, epochs: int = 30,
                 batch_size: int = 256, learning_rate: float = 3e-3, seed: int = 0,
                 verbose: bool = False):
        super().__init__(window=window, hidden_size=hidden_size, epochs=epochs,
                         batch_size=batch_size, learning_rate=learning_rate, seed=seed,
                         verbose=verbose)

    def _build_module(self) -> nn.Module:
        return LSTMNet(hidden_size=self.hidden_size)


# =============================================================================== PCRNN
class ThermalNetworkNet(nn.Module):
    """
    Lumped thermal network as a recurrent flax module.

    Input rows are laid out as ``[observed temperatures (°C) | heat sources | dt (h) |
    residual features]``. With state vector s (target first, then latent nodes), observed
    temperatures b and transformed sources u, the heat balance is

        C ds/dt = -A s + G_b b + B u (+ e_target q_NN(x) in hard mode)

    where A is built from the positive conductances (a weighted graph Laplacian plus the
    conductances to observed nodes), so A is an M-matrix whenever every state is connected
    to an observed temperature. Conductances, source gains and the latent heat capacities are
    positive (log-parametrised, so learning is scale-free); the target heat capacity is fixed to 1 (only ratios are identifiable),
    so G and beta are rates in 1/h and K/h.

    * ``mode="hard"``: the network *is* the model. It is integrated over the window with the
      implicit Euler scheme (C + dt A) s_t = C s_{t-1} + dt (G_b b_t + B u_t + q_t), which is
      unconditionally stable, starting from the steady state of the first window step.
      An optional MLP residual heat source q_NN(x_t) on the target absorbs unmodelled effects;
      its magnitude is L2-penalised.
    * ``mode="soft"``: an LSTM predicts the target at every window step; the heat balance with
      learnable coefficients enters only as a soft constraint (physics loss) on consecutive
      predictions, as in Exenberger et al. (2024). Requires a network without latent nodes.
    """
    n_states: int
    n_obs: int
    n_src: int
    state_edges: tuple          # ((i, j), ...) between state nodes
    obs_edges: tuple            # ((i, b), ...) state i - observed b
    src_links: tuple            # ((i, k), ...) source k heats state i
    y_mean: float
    y_std: float
    mode: str = "hard"
    residual: bool = True
    hidden_size: int = 16
    residual_penalty: float = 1.0
    physics_weight: float = 0.25
    init_conductance: float = 0.3   # 1/h
    init_gain: float = 1.0          # K/h at full load

    def _matrices(self):
        ns, no, nu = self.n_states, self.n_obs, self.n_src

        def positive(name, n, init):
            log_value = self.param(name, lambda key: jnp.full((n,), np.log(init), jnp.float32))
            return jnp.exp(log_value)

        g_obs = positive("conductance_obs", len(self.obs_edges), self.init_conductance)
        beta = positive("source_gain", len(self.src_links), self.init_gain)
        A = jnp.zeros((ns, ns))
        oi, ob = (np.array(v, dtype=int) for v in zip(*self.obs_edges))
        A = A.at[oi, oi].add(g_obs)
        G_b = jnp.zeros((ns, no)).at[oi, ob].add(g_obs)
        B = jnp.zeros((ns, nu))
        if self.src_links:
            li, lk = (np.array(v, dtype=int) for v in zip(*self.src_links))
            B = B.at[li, lk].add(beta)
        if self.state_edges:
            g_state = positive("conductance_state", len(self.state_edges), self.init_conductance)
            si, sj = (np.array(v, dtype=int) for v in zip(*self.state_edges))
            A = A.at[si, si].add(g_state).at[sj, sj].add(g_state)
            A = A.at[si, sj].add(-g_state).at[sj, si].add(-g_state)
        C = jnp.ones((1,))
        if ns > 1:
            C = jnp.concatenate([C, positive("capacity_latent", ns - 1, 1.0)])
        return A, G_b, B, C

    @nn.compact
    def __call__(self, x):  # x: (batch, window, n_inputs)
        no, nu = self.n_obs, self.n_src
        b, u, dt, feats = x[..., :no], x[..., no:no + nu], x[..., no + nu], x[..., no + nu + 1:]
        A, G_b, B, C = self._matrices()
        self.sow("intermediates", "matrices", (A, G_b, B, C))
        forcing = b @ G_b.T + u @ B.T                      # (batch, window, n_states)

        if self.mode == "soft":
            return self._soft(feats, forcing, dt, A)

        penalty = 0.0
        if self.residual and feats.shape[-1] > 0:
            h = nn.tanh(nn.Dense(self.hidden_size)(feats))
            h = nn.tanh(nn.Dense(self.hidden_size)(h))
            q = nn.Dense(1, kernel_init=nn.initializers.zeros)(h)[..., 0] * self.y_std  # K/h
            forcing = forcing.at[..., 0].add(q)
            penalty = self.residual_penalty * jnp.mean((q / self.y_std) ** 2)

        if self.n_states == 1:
            a, c = A[0, 0], C[0]
            temp0 = forcing[:, 0, 0] / a

            def step(temp, inp):
                f, d = inp
                return (c * temp + d * f) / (c + d * a), None
            temp, _ = jax.lax.scan(step, temp0, (forcing[:, 1:, 0].T, dt[:, 1:].T))
        else:
            s0 = jnp.linalg.solve(A, forcing[:, 0, :].T).T

            def step(s, inp):
                f, d = inp
                M = jnp.diag(C)[None] + d[:, None, None] * A[None]
                rhs = C * s + d[:, None] * f
                return jnp.linalg.solve(M, rhs[..., None])[..., 0], None
            s, _ = jax.lax.scan(step, s0, (jnp.swapaxes(forcing[:, 1:], 0, 1), dt[:, 1:].T))
            temp = s[:, 0]
        return (temp - self.y_mean) / self.y_std, penalty

    def _soft(self, feats, forcing, dt, A):
        h = nn.RNN(nn.OptimizedLSTMCell(self.hidden_size))(feats)
        z = nn.Dense(1)(h)[..., 0]                          # (batch, window), standardised
        temp = self.y_mean + self.y_std * z
        a = A[0, 0]
        d = dt[:, 1:]
        temp_phys = (temp[:, :-1] + d * forcing[:, 1:, 0]) / (1.0 + d * a)
        mask = (d > 0).astype(jnp.float32)
        err = ((temp[:, 1:] - temp_phys) / self.y_std) ** 2
        penalty = self.physics_weight * jnp.sum(err * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        return z[:, -1], penalty


class PCRNNForecaster(_WindowedNeuralForecaster):
    """
    Physics-constrained RNN based on a lumped thermal network (see ``ThermalNetworkNet``).

    The target temperature exchanges heat with observed temperatures (``neighbours``), receives
    heat from operating signals (``sources``) and may be coupled to unmeasured ``latent`` nodes.
    All heat-transfer coefficients are learned. Without arguments, the network for the target
    is taken from the ``thermal_networks`` section of sensor_config.yaml; ``neighbours``,
    ``sources``, ``latent``, ``edges`` and ``measured_states`` override the configured values, or
    pass a complete ``ThermalNetworkSpec`` as ``network`` (the same spec works for the Kalman
    model). The PCRNN has no observation model, so measured states are used as exact boundary
    temperatures. Sensors unavailable in the training data are dropped.

    Time steps are taken from a DatetimeIndex (gaps are integrated with the actual elapsed time,
    capped at ``max_gap_hours``); otherwise ``default_dt_hours`` is used.

    :param mode: "hard" (network is the model, optional residual) or "soft" (LSTM + physics loss)
    :param residual: hard mode only; add a learned residual heat source q_NN(x_t) on the target
    :param residual_features: inputs of the residual MLP / soft-mode LSTM. Default: all numeric
        columns except the target, other statistics of the target sensor and metadata columns.
    :param residual_penalty: weight of the L2 penalty on the (standardised) residual. The default
        (1.0) and the window (72 steps = 12 h) were chosen on chronological validation splits of the
        normal-behaviour datasets 83 and 86 of Wind Farm B (main_bearing_temp2_c_avg); smaller
        penalties let the residual absorb heat that the network should explain, larger ones
        underfit.
    :param physics_weight: soft mode only; weight of the physics loss
    """

    def __init__(self, network: Optional[ThermalNetworkSpec] = None,
                 neighbours: Optional[List[str]] = None,
                 sources: Optional[Dict[str, Union[str, tuple]]] = None,
                 latent: Optional[List[str]] = None, edges: Optional[List[tuple]] = None,
                 measured_states: Optional[List[str]] = None, mode: str = "hard", residual: bool = True,
                 residual_features: Optional[List[str]] = None,
                 residual_penalty: float = 1.0, physics_weight: float = 0.25,
                 max_gap_hours: float = 24.0, default_dt_hours: float = 1 / 6,
                 window: int = 72, hidden_size: int = 16, epochs: int = 20,
                 batch_size: int = 256, learning_rate: float = 1e-2, seed: int = 0,
                 verbose: bool = False):
        if mode not in ("hard", "soft"):
            raise ValueError("mode must be 'hard' or 'soft'.")
        self._overrides = dict(neighbours=neighbours, sources=sources, latent=latent, edges=edges,
                               measured_states=measured_states)
        if network is not None and any(v is not None for v in self._overrides.values()):
            raise ValueError("Pass either a ThermalNetworkSpec or neighbours/sources/latent/"
                             "edges/measured_states, not both.")
        self.network = network
        self.mode = mode
        self.residual = residual
        self.residual_features = residual_features
        self.residual_penalty = residual_penalty
        self.physics_weight = physics_weight
        self.max_gap_hours = max_gap_hours
        self.default_dt_hours = default_dt_hours
        super().__init__(window=window, hidden_size=hidden_size, epochs=epochs,
                         batch_size=batch_size, learning_rate=learning_rate, seed=seed,
                         verbose=verbose)

    def reset(self):
        super().reset()
        self.network_: Optional[ThermalNetworkSpec] = None

    # ---------------------------------------------------------------- inputs
    def _resolve_network(self, train_df) -> ThermalNetworkSpec:
        spec = self.network
        if spec is None:
            spec = ThermalNetworkSpec.from_config(self.target_col, **self._overrides)
        elif spec.target != self.target_col:
            raise ValueError(f"Network target '{spec.target}' differs from target_col "
                             f"'{self.target_col}'.")
        # no observation model: measured states are exact boundary temperatures
        spec = spec.restrict_to(train_df.columns).as_boundary()
        if self.mode == "soft" and spec.latent:
            raise ValueError("Soft mode does not support latent nodes.")
        return spec

    def _fit_inputs(self, train_df: pd.DataFrame):
        spec = self.network_ = self._resolve_network(train_df)
        self._obs_cols = spec.observed
        self._src_links = spec.source_links
        self._src_cols = [s for s, _, _ in self._src_links]
        obs = train_df[self._obs_cols].to_numpy(dtype=np.float64)
        self._obs_fill = np.nan_to_num(np.nanmean(obs, axis=0)) if self._obs_cols else np.zeros(0)
        src = train_df[self._src_cols].to_numpy(dtype=np.float64)
        scale = np.nan_to_num(np.nanpercentile(np.abs(src), 99, axis=0)) if self._src_cols \
            else np.zeros(0)
        self._src_scale = np.where(scale < 1e-8, 1.0, scale)
        self._src_fill = np.nan_to_num(np.nanmean(self._transform_sources(src, fill=False),
                                                  axis=0)) if self._src_cols else np.zeros(0)

        if self.residual_features is not None:
            res = list(self.residual_features)
        else:
            res = feature_columns(train_df, self.target_col)
        if self.mode == "soft" and not res:
            raise ValueError("Soft mode needs at least one residual feature as LSTM input.")
        self.feature_cols = res
        feats = train_df[res].to_numpy(dtype=np.float32) if res else np.zeros((1, 0))
        self._x_mean = np.nan_to_num(np.nanmean(feats, axis=0))
        x_std = np.nan_to_num(np.nanstd(feats, axis=0))
        self._x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)

    def _transform_sources(self, src: np.ndarray, fill: bool = True) -> np.ndarray:
        u = np.maximum(src / self._src_scale, 0.0)  # heat generation is non-negative
        squares = np.array([t == "square" for _, _, t in self._src_links], dtype=bool)
        u[:, squares] = u[:, squares] ** 2
        if fill:
            u = np.where(np.isnan(u), self._src_fill, u)
        return u

    def _time_steps(self, df: pd.DataFrame) -> np.ndarray:
        index = df.index
        if isinstance(index, pd.MultiIndex):
            levels = [index.get_level_values(i) for i in range(index.nlevels)]
            index = next((lv for lv in levels if isinstance(lv, pd.DatetimeIndex)), None)
        dt = np.full(len(df), self.default_dt_hours)
        if isinstance(index, pd.DatetimeIndex) and len(df) > 1:
            diff = np.diff(index.asi8) / 3.6e12  # ns -> h
            dt[1:] = np.where(np.isfinite(diff), np.clip(diff, 0.0, self.max_gap_hours),
                              self.default_dt_hours)
        if len(dt):
            dt[0] = 0.0  # no previous step; also makes the start padding a no-op
        return dt

    def _required_columns(self) -> List[str]:
        return list(dict.fromkeys(self._obs_cols + self._src_cols + self.feature_cols))

    def _make_inputs(self, df: pd.DataFrame) -> np.ndarray:
        obs = df.reindex(columns=self._obs_cols).to_numpy(dtype=np.float64)
        obs = np.where(np.isnan(obs), self._obs_fill, obs)
        src = self._transform_sources(df.reindex(columns=self._src_cols).to_numpy(dtype=np.float64))
        dt = self._time_steps(df)[:, None]
        feats = _standardise(df, self.feature_cols, self._x_mean, self._x_std)
        x = np.concatenate([obs, src, dt, feats], axis=1).astype(np.float32)
        self._n_inputs = x.shape[1]
        return x

    def _build_module(self) -> nn.Module:
        spec = self.network_
        states, obs = spec.states, self._obs_cols
        state_edges, obs_edges = [], []
        for a, b in spec.all_edges:
            if a in states and b in states:
                state_edges.append((states.index(a), states.index(b)))
            else:
                s, o = (a, b) if a in states else (b, a)
                obs_edges.append((states.index(s), obs.index(o)))
        src_links = tuple((states.index(node), k) for k, (_, node, _) in enumerate(self._src_links))
        return ThermalNetworkNet(
            n_states=len(states), n_obs=len(obs), n_src=len(self._src_links),
            state_edges=tuple(state_edges), obs_edges=tuple(obs_edges), src_links=src_links,
            y_mean=self._y_mean, y_std=self._y_std, mode=self.mode, residual=self.residual,
            hidden_size=self.hidden_size, residual_penalty=self.residual_penalty,
            physics_weight=self.physics_weight)

    # ---------------------------------------------------------------- interpretation
    @property
    def thermal_parameters_(self) -> dict:
        """
        Learned physical parameters of the fitted network (target heat capacity = 1):

        * ``conductances``: {(node_a, node_b): G in 1/h}
        * ``source_gains``: {(source, node): K/h at full load (99th percentile of the source)}
        * ``capacities``: {state: C relative to the target}
        * ``time_constants_h``: {state: C_i / sum of its conductances, in hours}
        * ``steady_state_rise_K``: {(source, node): steady temperature rise at full load}
        """
        if self.params is None:
            raise RuntimeError("Model must be fitted first.")
        dummy = jnp.zeros((1, self.window, self._n_inputs), jnp.float32)
        _, state = self._module.apply(self.params, dummy, mutable=["intermediates"])
        A, G_b, B, C = (np.asarray(m) for m in state["intermediates"]["matrices"][0])
        spec, states = self.network_, self.network_.states
        conductances = {}
        for a, b in spec.all_edges:
            if a in states and b in states:
                conductances[(a, b)] = float(-A[states.index(a), states.index(b)])
            else:
                s, o = (a, b) if a in states else (b, a)
                conductances[(a, b)] = float(G_b[states.index(s), self._obs_cols.index(o)])
        gains, rises = {}, {}
        for k, (name, node, _) in enumerate(self._src_links):
            i = states.index(node)
            gains[(name, node)] = float(B[i, k])
            rises[(name, node)] = float(np.linalg.solve(A, B[:, k])[i])
        return {
            "conductances": conductances,
            "source_gains": gains,
            "capacities": {s: float(C[i]) for i, s in enumerate(states)},
            "time_constants_h": {s: float(C[i] / A[i, i]) for i, s in enumerate(states)},
            "steady_state_rise_K": rises,
        }
