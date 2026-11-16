"""Tests for the LSTM and PCRNN forecasters and the thermal network specification."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from care_scada.modeling.thermal_network import ThermalNetworkSpec, plot_thermal_network

TARGET = "temp"


# --------------------------------------------------------------------------- feature selection
def test_feature_columns_exclude_target_statistics_and_metadata():
    from care_scada.model_factory import ModelFactory, feature_columns
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, 7)),
                      columns=["bearing_avg", "bearing_max", "bearing_std", "bearing2_avg",
                               "power", "id", "status_type_id"])
    df["label"] = "train"
    assert feature_columns(df, "bearing_avg") == ["bearing2_avg", "power"]

    model = ModelFactory.create("random_forest")
    model.fit(df, "bearing_avg")
    assert model.feature_cols == ["bearing2_avg", "power"]
    # predict only needs the columns used in fit
    assert len(model.predict(df[["bearing2_avg", "power"]], "bearing_avg")) == len(df)


# --------------------------------------------------------------------------- spec (no JAX)
def test_spec_structure():
    spec = ThermalNetworkSpec(target="T", neighbours=["amb", "nb"],
                              sources={"rpm": "linear", "power": ("air", "square")},
                              latent=["air"], edges=[("air", "amb"), ("T", "air"), ("amb", "T")])
    assert spec.states == ["T", "air"]
    assert spec.observed == ["amb", "nb"]
    assert len(spec.all_edges) == 4  # duplicate (amb, T) removed
    assert spec.source_links == [("rpm", "T", "linear"), ("power", "air", "square")]
    assert "G·(T[amb] − T)" in spec.equation()


@pytest.mark.parametrize("kwargs", [
    dict(neighbours=[]),                                               # no heat sink
    dict(neighbours=["amb"], latent=["air"]),                          # isolated latent node
    dict(neighbours=["amb"], sources={"rpm": "cubic"}),                # unknown transform
    dict(neighbours=["amb"], sources={"rpm": ("nowhere", "linear")}),  # source to non-state
    dict(neighbours=["amb"], sources={"amb": "linear"}),               # source is a temperature
])
def test_spec_validation_errors(kwargs):
    with pytest.raises(ValueError):
        ThermalNetworkSpec(target="T", **kwargs)


def test_spec_restrict_and_config_defaults():
    spec = ThermalNetworkSpec.from_config("main_bearing_temp2_c_avg")
    assert "outside_temp_c_avg" in spec.neighbours
    assert "main_bearing_temp1_c_avg" in spec.neighbours
    assert spec.sources["rotor_speed_rpm_avg"] == "linear"
    with pytest.warns(UserWarning, match="main_bearing_temp3_c_avg"):
        restricted = spec.restrict_to(["outside_temp_c_avg", "main_bearing_temp1_c_avg",
                                       "gearbox_bearing_temp1_c_avg", "rotor_speed_rpm_avg",
                                       "active_power_kWh"])
    assert "main_bearing_temp3_c_avg" not in restricted.observed
    # unknown targets use the default entry; explicit arguments override the config
    default = ThermalNetworkSpec.from_config("some_temp", neighbours=["x"])
    assert default.neighbours == ["x"] and "active_power_kWh" in default.sources


def test_spec_measured_states():
    spec = ThermalNetworkSpec.from_config("T", neighbours=["amb", "nb", "gb"], sources=["rpm"],
                                          measured_states=["nb"])
    # nb becomes a state coupled to T, sharing T's observed neighbours and sources
    assert spec.states == ["T", "nb"] and spec.observed == ["amb", "gb"]
    assert {frozenset(e) for e in spec.all_edges} == {frozenset(e) for e in [
        ("T", "amb"), ("T", "gb"), ("nb", "T"), ("nb", "amb"), ("nb", "gb")]}
    assert spec.source_links == [("rpm", "T", "linear"), ("rpm", "nb", "linear")]
    assert spec.sources == {"rpm": [("T", "linear"), ("nb", "linear")]}
    # without an observation model nb is a boundary temperature again
    boundary = spec.as_boundary()
    assert boundary.states == ["T"] and boundary.observed == ["amb", "gb", "nb"]
    assert boundary.source_links == [("rpm", "T", "linear")]
    # an unavailable measured state is dropped with its edges and source links
    with pytest.warns(UserWarning, match="nb"):
        restricted = spec.restrict_to(["amb", "gb", "rpm"])
    assert restricted.states == ["T"] and restricted.source_links == [("rpm", "T", "linear")]
    with pytest.raises(ValueError):  # a measured state needs a connection
        ThermalNetworkSpec(target="T", neighbours=["amb"], measured_states=["nb"])


def test_plot_spec_with_measured_state():
    spec = ThermalNetworkSpec(target="T", neighbours=["amb"], sources={"rpm": "linear"}).with_measured_states(["nb"])
    fig = plot_thermal_network(spec)
    nodes = {t.name: list(t.text) for t in fig.data if t.mode == "markers+text"}
    assert nodes["predicted, measured"] == ["T<sub>0</sub>"] and "T<sub>1</sub>" in nodes["helper, measured"]
    assert "T<sub>1</sub> = nb (measured state)" in " ".join(a.text or "" for a in fig.layout.annotations)


def test_plot_latent_network_symbols():
    spec = ThermalNetworkSpec(target="T", neighbours=["amb", "nb"], latent=["air"],
                              edges=[("air", "T"), ("air", "amb"), ("air", "air_sensor")],
                              sources={"rpm": "linear", "power": ("air", "square")})
    fig = plot_thermal_network(spec)
    text = [a.text or "" for a in fig.layout.annotations]
    # one heat balance per state node, symbols explained below (T0 target, T1 latent)
    equations = next(t for t in text if "dT<sub>0</sub>/dt" in t)
    assert "dT<sub>1</sub>/dt" in equations and "u<sub>1</sub>²" in equations
    key = " ".join(text)
    assert "T<sub>0</sub> = T (predicted)" in key
    assert "T<sub>1</sub> = air (latent)" in key and "T<sub>2</sub> = air_sensor" in key
    assert "u<sub>1</sub> = power (squared)" in key
    nodes = {t.name: t for t in fig.data if t.mode == "markers+text"}
    assert list(nodes["helper, latent"].text) == ["T<sub>1</sub>"]
    # the latent node sits left of the target; its source and its sensor are drawn on that side
    x = {sym: x for t in nodes.values() for sym, x in zip(t.text, t.x)}
    assert x["T<sub>1</sub>"] < 0 and x["u<sub>1</sub>"] < 0 and x["T<sub>2</sub>"] < 0


def test_plot_spec():
    spec = ThermalNetworkSpec(target="T", neighbours=["amb"], sources={"rpm": "linear"})
    fig = plot_thermal_network(spec)
    assert isinstance(fig, go.Figure)
    # colour = role, fill = observation; no observation model: the target itself is predicted and measured
    classes = {t.name: (list(t.text), t.marker.color) for t in fig.data if t.mode == "markers+text"}
    assert set(classes) == {"predicted, measured", "helper, measured"}
    assert classes["predicted, measured"][0] == ["T<sub>0</sub>"]
    assert classes["predicted, measured"][1] != classes["helper, measured"][1] != "white"


# --------------------------------------------------------------------------- models (JAX)
jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from care_scada.model_factory import ModelFactory  # noqa: E402

TRUE_G = {"amb": 0.5, "nb": 0.3}  # 1/h
TRUE_BETA = 4.0                   # K/h at the 99th percentile of rpm


def _make_df(n, seed, t0=0):
    """Temperature following a first-order lag of a noisy load signal."""
    rng = np.random.default_rng(seed)
    t = np.arange(t0, t0 + n)
    load = np.sin(t / 30.0) + 0.1 * rng.standard_normal(n)
    temp = np.zeros(n)
    for i in range(1, n):
        temp[i] = temp[i - 1] + 0.2 * (20 * load[i] + 40 - temp[i - 1])
    idx = pd.date_range("2022-01-01", periods=n, freq="10min")
    return pd.DataFrame({"load": load, "const": 1.0, TARGET: temp,
                         "label": "train"}, index=idx)


def _make_thermal_df(n=2500, seed=0):
    """Target generated by a known lumped network (implicit Euler, dt = 10 min)."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    amb = 10 + 5 * np.sin(k / 144 * 2 * np.pi) + rng.normal(0, 0.3, n)
    rpm = np.clip(8 + 4 * np.sin(k / 50) + rng.normal(0, 1, n), 0, None)
    nb = amb + 8 + rng.normal(0, 0.3, n)
    scale = np.percentile(rpm, 99)
    g_sum, dt = sum(TRUE_G.values()), 1 / 6
    temp = np.empty(n)
    temp[0] = (TRUE_G["amb"] * amb[0] + TRUE_G["nb"] * nb[0] + TRUE_BETA * rpm[0] / scale) / g_sum
    for i in range(1, n):
        forcing = TRUE_G["amb"] * amb[i] + TRUE_G["nb"] * nb[i] + TRUE_BETA * rpm[i] / scale
        temp[i] = (temp[i - 1] + dt * forcing) / (1 + dt * g_sum)
    idx = pd.date_range("2022-01-01", periods=n, freq="10min")
    return pd.DataFrame({"amb": amb, "nb": nb, "rpm": rpm,
                         TARGET: temp + rng.normal(0, 0.05, n)}, index=idx)


NETWORK = dict(neighbours=["amb", "nb"], sources={"rpm": "linear"})
FAST = dict(epochs=8, batch_size=64, window=24)


@pytest.fixture(scope="module")
def data():
    return _make_df(600, 0), _make_df(200, 1, t0=600)


@pytest.fixture(scope="module")
def thermal_data():
    df = _make_thermal_df()
    return df.iloc[:2000], df.iloc[2000:]


def _rmse(preds, test):
    return float(np.sqrt(np.mean((preds[TARGET] - test[TARGET]) ** 2)))


def _check_fit_predict_reset(model, train, test):
    model.fit(train, TARGET)
    preds = model.predict(test, TARGET)
    assert list(preds.columns) == [TARGET]
    assert preds.index.equals(test.index)
    assert np.isfinite(preds[TARGET]).all()
    # clearly better than predicting the training mean
    baseline = np.sqrt(np.mean((train[TARGET].mean() - test[TARGET]) ** 2))
    assert _rmse(preds, test) < baseline
    model.reset()
    with pytest.raises(RuntimeError):
        model.predict(test, TARGET)


def test_lstm_fit_predict_reset(data):
    model = ModelFactory.create("lstm", epochs=30, hidden_size=8, window=12, batch_size=64,
                                learning_rate=5e-3)
    _check_fit_predict_reset(model, *data)


@pytest.mark.parametrize("kwargs", [
    dict(residual=False),
    dict(),                                                        # hard mode with residual
    dict(mode="soft", hidden_size=8),
    dict(residual=False, latent=["air"], edges=[("temp", "air"), ("air", "amb")]),
])
def test_pcrnn_modes_fit_predict_reset(thermal_data, kwargs):
    model = ModelFactory.create("pcrnn", **NETWORK, **FAST, **kwargs)
    _check_fit_predict_reset(model, *thermal_data)


def test_pcrnn_recovers_true_parameters(thermal_data):
    train, test = thermal_data
    model = ModelFactory.create("pcrnn", **NETWORK, residual=False, epochs=30, batch_size=64,
                                window=24)
    model.fit(train, TARGET)
    params = model.thermal_parameters_
    assert params["conductances"][(TARGET, "amb")] == pytest.approx(TRUE_G["amb"], rel=0.1)
    assert params["conductances"][(TARGET, "nb")] == pytest.approx(TRUE_G["nb"], rel=0.1)
    assert params["source_gains"][("rpm", TARGET)] == pytest.approx(TRUE_BETA, rel=0.1)
    assert params["time_constants_h"][TARGET] == pytest.approx(1 / sum(TRUE_G.values()), rel=0.1)
    assert _rmse(model.predict(test, TARGET), test) < 0.2


def test_pcrnn_cooling_source(thermal_data):
    # negated temperatures: the same network with rpm as a cooling source (gain -TRUE_BETA)
    train, test = (df.assign(**{c: -df[c] for c in (TARGET, "amb", "nb")}) for df in thermal_data)
    model = ModelFactory.create("pcrnn", **NETWORK, cooling=[("rpm", TARGET)], residual=False, epochs=30,
                                batch_size=64, window=24)
    model.fit(train, TARGET)
    params = model.thermal_parameters_
    assert params["source_gains"][("rpm", TARGET)] == pytest.approx(-TRUE_BETA, rel=0.1)
    assert params["steady_state_rise_K"][("rpm", TARGET)] < 0
    assert _rmse(model.predict(test, TARGET), test) < 0.2


def test_spec_retarget():
    spec = ThermalNetworkSpec(target=TARGET, neighbours=["amb"], sources={"rpm": "linear"}).with_measured_states(["nb"])
    flipped = spec.retarget("nb")
    assert flipped.target == "nb" and flipped.measured_states == [TARGET]
    assert set(map(frozenset, flipped.all_edges)) == set(map(frozenset, spec.all_edges))
    assert sorted(flipped.source_links) == sorted(spec.source_links)
    assert flipped.retarget(TARGET).target == TARGET and spec.retarget(TARGET) is spec
    with pytest.raises(ValueError):
        spec.retarget("amb")


def test_pcrnn_time_steps_follow_index(thermal_data):
    train, _ = thermal_data
    model = ModelFactory.create("pcrnn", **NETWORK, max_gap_hours=2.0)
    stamps = pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:30",
                             "2022-01-02 00:00"])
    dt = model._time_steps(pd.DataFrame(index=stamps))
    np.testing.assert_allclose(dt, [0.0, 1 / 6, 1 / 3, 2.0])


def test_pcrnn_network_from_config_drops_missing_sensors(thermal_data):
    train, test = thermal_data
    df = train.rename(columns={"amb": "outside_temp_c_avg", "rpm": "rotor_speed_rpm_avg"})
    model = ModelFactory.create("pcrnn", residual=False, **FAST)
    with pytest.warns(UserWarning, match="active_power_kWh"):
        model.fit(df, TARGET)  # 'temp' has no entry -> default network
    assert model.network_.observed == ["outside_temp_c_avg"]
    assert list(model.network_.sources) == ["rotor_speed_rpm_avg"]


def test_pcrnn_rejects_inconsistent_arguments(thermal_data):
    spec = ThermalNetworkSpec(target="other", neighbours=["amb"])
    with pytest.raises(ValueError):
        ModelFactory.create("pcrnn", network=spec, neighbours=["amb"])
    with pytest.raises(ValueError):
        ModelFactory.create("pcrnn", network=spec).fit(thermal_data[0], TARGET)
    with pytest.raises(ValueError):
        ModelFactory.create("pcrnn", mode="soft", latent=["air"], neighbours=["amb"],
                            edges=[("temp", "air"), ("air", "amb")]).fit(thermal_data[0], TARGET)


def test_plot_fitted_pcrnn(thermal_data):
    model = ModelFactory.create("pcrnn", **NETWORK, **FAST)
    model.fit(thermal_data[0], TARGET)
    fig = plot_thermal_network(model)
    text = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "G=" in text and " K" in text
    assert "q<sub>NN</sub>(x)" in text  # heat balance above the network
    assert "q<sub>NN</sub>(x) = residual heat" in text and "C<sub>0</sub> = 1" in text  # key below


def test_neural_models_handle_nans(data, thermal_data):
    train, test = data
    train = train.copy()
    train.iloc[10:15, train.columns.get_loc("load")] = np.nan
    train.iloc[30:33, train.columns.get_loc(TARGET)] = np.nan
    model = ModelFactory.create("lstm", epochs=1, hidden_size=4, window=8)
    model.fit(train, TARGET)
    assert np.isfinite(model.predict(test, TARGET)[TARGET]).all()

    train, test = (df.copy() for df in thermal_data)
    train.iloc[10:15, train.columns.get_loc("amb")] = np.nan
    train.iloc[30:33, train.columns.get_loc(TARGET)] = np.nan
    test.iloc[5:8, test.columns.get_loc("rpm")] = np.nan
    model = ModelFactory.create("pcrnn", **NETWORK, epochs=1, window=8)
    model.fit(train, TARGET)
    assert np.isfinite(model.predict(test, TARGET)[TARGET]).all()


def test_random_forest_kwargs_survive_reset():
    from care_scada.model_factory import ModelFactory
    assert ModelFactory.create("random_forest").model.n_estimators == 10
    model = ModelFactory.create("random_forest", n_estimators=50, max_depth=5)
    model.reset()
    assert (model.model.n_estimators, model.model.max_depth) == (50, 5)


# --------------------------------------------------------------------------- Kalman model (JAX)
# Two coupled nodes: target T and an observed neighbour "nb", ambient sink, rpm source (scaled
# to its 99th percentile = 12), simulated exactly (zero-order hold, dt = 10 min).
K_A = np.array([[-0.8, 0.3], [0.1, -0.3]])         # 1/h: T<-amb 0.5, T<-nb 0.3, nb<-amb 0.2, nb<-T 0.1
K_B = np.array([[0.5, 4.0, 1.0], [0.2, 1.0, 0.0]])  # columns: amb, rpm / 12, extra heat on T (K/h)


def _make_network_df(n=3000, seed=0, extra_heat=0.0, extra_from=0):
    from scipy.linalg import expm
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    amb = 10 + 5 * np.sin(k / 144 * 2 * np.pi) + rng.normal(0, 0.3, n)
    rpm = np.clip(8 + 6 * np.sin(k / 50) + rng.normal(0, 1, n), 0, 12)
    u = np.column_stack([amb, rpm / 12, np.where(k >= extra_from, extra_heat, 0.0)])
    E = expm(np.block([[K_A, K_B], [np.zeros((3, 5))]]) / 6)
    x = np.linalg.solve(-K_A, K_B @ u[0])
    states = []
    for u_t in u:
        x = E[:2, :2] @ x + E[:2, 2:] @ u_t
        states.append(x)
    states = np.array(states) + rng.normal(0, 0.05, (n, 2))
    idx = pd.date_range("2022-01-01", periods=n, freq="10min")
    return pd.DataFrame({"amb": amb, "rpm": rpm, TARGET: states[:, 0], "nb": states[:, 1]}, index=idx)


KALMAN = dict(neighbours=["amb", "nb"], sources=["rpm"])


@pytest.fixture(scope="module")
def network_data():
    df = _make_network_df()
    return df.iloc[:2000], df.iloc[2000:]


def test_kalman_fit_predict_reset(network_data):
    _check_fit_predict_reset(ModelFactory.create("kalman", **KALMAN, steps=200), *network_data)


@pytest.mark.parametrize("training, steps", [("observer", 600), ("kalman", 1500)])
def test_kalman_recovers_true_parameters(network_data, training, steps):
    train, test = network_data
    model = ModelFactory.create("kalman", **KALMAN, training=training, steps=steps).fit(train, TARGET)
    params = model.summary()
    assert params["B[amb]"] == pytest.approx(0.5, rel=0.1)
    assert params["B[nb]"] == pytest.approx(0.3, rel=0.1)
    assert params["B[rpm]"] == pytest.approx(4.0, rel=0.1)
    assert params[f"tau_h[{TARGET}]"] == pytest.approx(1 / 0.8, rel=0.1)
    assert _rmse(model.predict(test, TARGET), test) < 0.2


def test_kalman_prediction_ignores_target_and_handles_gaps(network_data):
    train, test = network_data
    model = ModelFactory.create("kalman", **KALMAN, steps=50).fit(train, TARGET)
    test = test.drop(test.index[100:130])           # 5 h gap -> new sequence
    test.iloc[200:205, test.columns.get_loc("amb")] = np.nan
    preds = model.predict(test, TARGET)
    assert preds.index.equals(test.index) and np.isfinite(preds[TARGET]).all()
    pd.testing.assert_frame_equal(model.predict(test.assign(**{TARGET: np.nan}), TARGET), preds)


def test_kalman_network_from_config(network_data):
    df = network_data[0].rename(columns={"amb": "outside_temp_c_avg", "nb": "main_bearing_temp1_c_avg",
                                         "rpm": "rotor_speed_rpm_avg", TARGET: "main_bearing_temp2_c_avg"})
    model = ModelFactory.create("kalman", steps=5).fit(df, "main_bearing_temp2_c_avg")
    # unavailable config sensors (bearing 3, gearbox bearing, power) are dropped
    assert model.sinks_ == ["outside_temp_c_avg", "main_bearing_temp1_c_avg"]
    assert model.sources_ == ["rotor_speed_rpm_avg"]
    model = ModelFactory.create("kalman", measured_states=["main_bearing_temp1_c_avg"], steps=5)
    model.fit(df, "main_bearing_temp2_c_avg")
    assert model.states_ == ["main_bearing_temp2_c_avg", "main_bearing_temp1_c_avg"]
    assert model.sinks_ == ["outside_temp_c_avg"]
    with pytest.raises(ValueError):
        ModelFactory.create("kalman", training="other")
    with pytest.raises(ValueError):  # no heat sink left
        ModelFactory.create("kalman", neighbours=["missing"]).fit(df, "main_bearing_temp2_c_avg")
    spec = ThermalNetworkSpec(target="other", neighbours=["amb"])
    with pytest.raises(ValueError):
        ModelFactory.create("kalman", network=spec, neighbours=["amb"])
    with pytest.raises(ValueError):
        ModelFactory.create("kalman", network=spec).fit(network_data[0], TARGET)


def test_kalman_custom_network(network_data):
    train, test = network_data
    spec = ThermalNetworkSpec(target=TARGET, neighbours=["amb"], latent=["air"],
                              edges=[("air", TARGET), ("air", "amb")],
                              sources={"rpm": [(TARGET, "linear"), ("air", "square")]}).with_measured_states(["nb"])
    model = ModelFactory.create("kalman", network=spec, steps=30).fit(train, TARGET)
    assert model.states_ == [TARGET, "nb", "air"] and model.sensors_ == [TARGET, "nb"]
    p = model.thermal_parameters_
    # only the edges and source links of the spec; c, gamma and sensor noise only on measured states
    assert set(p["couplings"]) == {(TARGET, "nb"), ("nb", TARGET), (TARGET, "air"), ("air", TARGET)}
    assert set(p["sink_rates"]) == {(TARGET, "amb"), ("nb", "amb"), ("air", "amb")}
    assert set(p["source_gains"]) == {("rpm", TARGET), ("rpm", "nb"), ("rpm²", "air")}
    assert set(p["const_heat"]) == set(p["sensor_noise_K"]) == {TARGET, "nb"}
    assert np.isfinite(model.predict(test, TARGET)[TARGET]).all()
    assert list(model.track_heat(test).columns) == [TARGET, "nb"]


def test_same_spec_for_pcrnn_and_kalman(network_data):
    spec = ThermalNetworkSpec(target=TARGET, neighbours=["amb"], sources={"rpm": "linear"}).with_measured_states(["nb"])
    kalman = ModelFactory.create("kalman", network=spec, steps=5).fit(network_data[0], TARGET)
    pcrnn = ModelFactory.create("pcrnn", network=spec, epochs=1, window=8)
    pcrnn.fit(network_data[0], TARGET)
    # the Kalman model observes nb through a sensor; the PCRNN uses it as a boundary temperature
    assert kalman.states_ == [TARGET, "nb"] and kalman.sinks_ == ["amb"]
    assert pcrnn.network_.states == [TARGET] and pcrnn.network_.observed == ["amb", "nb"]


def test_kalman_refit(network_data):
    train, _ = network_data
    model = ModelFactory.create("kalman", **KALMAN).fit(train, TARGET)
    shifted = _make_network_df(1000, seed=1, extra_heat=0.4)
    params = {k: np.array(v) for k, v in model.params.items()}
    free = model.refit(shifted, prior_weight=0.0)
    assert free.summary()["B[const]"] == pytest.approx(0.4, abs=0.1)
    bias = {m: float((shifted[TARGET] - m.predict(shifted, TARGET)[TARGET]).iloc[100:].mean())
            for m in (model, free)}
    assert bias[model] > 0.3 and abs(bias[free]) < 0.05
    # the original model is unchanged; a strong prior keeps the refit at the original fit
    assert all(np.array_equal(params[k], model.params[k]) for k in params)
    stiff = model.refit(shifted, prior_weight=1e6)
    assert stiff.summary()["B[const]"] == pytest.approx(model.summary()["B[const]"], abs=0.02)


def test_kalman_track_heat(network_data):
    train, _ = network_data
    model = ModelFactory.create("kalman", neighbours=["amb"], sources=["rpm"], measured_states=["nb"],
                                training="kalman").fit(train, TARGET)
    fault = _make_network_df(3000, seed=1, extra_heat=0.4, extra_from=1500)
    excess = model.track_heat(fault)
    assert list(excess.columns) == [TARGET, "nb"] and excess.index.equals(fault.index)
    rise = excess.iloc[-144:].mean() - excess.iloc[1000:1500].mean()
    # extra heat on T: steady-state excess (-A^-1) [0.4, 0] of both nodes
    expected = np.linalg.solve(-K_A, [0.4, 0.0])
    assert rise[TARGET] == pytest.approx(expected[0], rel=0.2)
    assert rise["nb"] == pytest.approx(expected[1], abs=0.1)


def test_plot_kalman_network(network_data):
    model = ModelFactory.create("kalman", neighbours=["amb"], sources=["rpm"], measured_states=["nb"])
    with pytest.raises(ValueError):
        plot_thermal_network(model)
    model.steps = 20
    model.fit(network_data[0], TARGET)
    fig = plot_thermal_network(model)
    text = " ".join(a.text or "" for a in fig.layout.annotations)
    # one balance per state with constant and outside heat; both coupling rates on the state edge
    assert "dT<sub>0</sub>/dt" in text and "dT<sub>1</sub>/dt" in text and "c<sub>1</sub>" in text
    assert "k<sub>0,1</sub>=" in text and "k<sub>1,0</sub>=" in text
    # latent true temperatures with their sensors; latent constant heats as heat inputs
    assert "T<sub>1</sub> = true temperature of nb" in text and "(sensor of T<sub>0</sub>, fit only)" in text
    assert "y<sub>1</sub> = nb (sensor of T<sub>1</sub>)" in text and "c<sub>0</sub> = constant heat" in text
    nodes = {t.name: list(t.text) for t in fig.data if t.mode == "markers+text"}
    assert nodes["predicted, latent"] == ["T<sub>0</sub>"]
    assert set(nodes["helper, latent"]) == {"T<sub>1</sub>", "c<sub>0</sub>", "c<sub>1</sub>"}
