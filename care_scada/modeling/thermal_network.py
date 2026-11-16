"""Lumped thermal network specification and plotting.

One ``ThermalNetworkSpec`` describes the heat network of a target temperature; the same spec can
be passed to the PCRNN and the Kalman model (``network=spec``) and plotted with
``plot_thermal_network``. It consists of

* **state nodes** with their own heat balance: the target (predicted), optional
  ``measured_states`` (measured temperature columns modelled as states, e.g. a neighbouring
  bearing) and optional ``latent`` (unmeasured) nodes, e.g. the nacelle air,
* **observed nodes**: measured temperatures acting as boundary conditions (heat sinks), e.g. the
  outside temperature,
* **edges**: thermal conductances between nodes (learned, positive),
* **heat sources**: operating signals (rotor speed, active power, ...) heating one or more state
  nodes with learned, positive gains.

For every state node i the network encodes the lumped heat balance

    C_i dT_i/dt = sum_j G_ij (T_j - T_i) + sum_k beta_ik f(u_k)

Each model adds its own assumptions: the Kalman model observes the target and the measured states
through noisy sensors (and adds constant / outside-temperature heat terms), while the PCRNN has no
observation model and uses measured states as exact boundary temperatures (``as_boundary``).

This module has no JAX dependency, so specifications can be built and plotted without the
optional ``deep`` extra.
"""

from __future__ import annotations

import math
import re
import os
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import yaml

TRANSFORMS = ("linear", "square")
SourceSpec = Union[str, Tuple[str, str], List[Tuple[str, str]]]

_DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "sensor_config.yaml")


@dataclass
class ThermalNetworkSpec:
    """
    Structure of a lumped thermal network around a target temperature.

    :param target: Predicted temperature column (a state node).
    :param neighbours: Observed temperature columns directly connected to the target.
    :param sources: Heat sources ``{column: transform}`` feeding the target,
        ``{column: (node, transform)}`` feeding another state node, or
        ``{column: [(node, transform), ...]}`` feeding several. ``transform`` is ``"linear"``
        (heat ~ u) or ``"square"`` (heat ~ u^2).
    :param latent: Names of unmeasured state nodes (e.g. ``"nacelle"``).
    :param edges: Additional conductances ``(node_a, node_b)`` between any nodes. Endpoints that
        are not state nodes are observed temperature columns.
    :param measured_states: Measured temperature columns modelled as state nodes (connected only
        through ``edges`` / ``sources``; see ``with_measured_states`` for a quick default).
    """
    target: str
    neighbours: List[str] = field(default_factory=list)
    sources: Dict[str, SourceSpec] = field(default_factory=dict)
    latent: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    measured_states: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.neighbours = list(dict.fromkeys(self.neighbours or []))
        self.sources = dict(self.sources or {})
        self.latent = list(dict.fromkeys(self.latent or []))
        self.edges = [tuple(e) for e in (self.edges or [])]
        self.measured_states = list(dict.fromkeys(self.measured_states or []))
        self.validate()

    # ------------------------------------------------------------------ structure
    @property
    def states(self) -> List[str]:
        """State nodes: the target first, then measured states, then latent nodes."""
        return [self.target] + self.measured_states + self.latent

    @property
    def all_edges(self) -> List[Tuple[str, str]]:
        """Unique undirected edges (neighbour edges first)."""
        out, seen = [], set()
        for a, b in [(self.target, n) for n in self.neighbours] + self.edges:
            key = frozenset((a, b))
            if key not in seen:
                seen.add(key)
                out.append((a, b))
        return out

    @property
    def observed(self) -> List[str]:
        """Observed temperature columns (boundary nodes), in order of appearance."""
        states = set(self.states)
        nodes = [n for e in self.all_edges for n in e if n not in states]
        return list(dict.fromkeys(nodes))

    @property
    def source_links(self) -> List[Tuple[str, str, str]]:
        """``(source column, receiving state node, transform)`` triples."""
        links = []
        for name, spec in self.sources.items():
            if isinstance(spec, str):
                links.append((name, self.target, spec))
            elif spec and isinstance(spec[0], (list, tuple)):
                links += [(name, node, transform) for node, transform in spec]
            else:
                node, transform = spec
                links.append((name, node, transform))
        return links

    @property
    def required_columns(self) -> List[str]:
        return list(dict.fromkeys(self.measured_states + self.observed
                                  + [s for s, _, _ in self.source_links]))

    def validate(self):
        states = self.states
        if len(set(states)) != len(states):
            raise ValueError("Latent and measured state names must be unique and differ from the target.")
        for a, b in self.all_edges:
            if a == b:
                raise ValueError(f"Self-loop edge on '{a}'.")
        for name, node, transform in self.source_links:
            if transform not in TRANSFORMS:
                raise ValueError(f"Unknown source transform '{transform}' for '{name}' "
                                 f"(use one of {TRANSFORMS}).")
            if node not in states:
                raise ValueError(f"Source '{name}' feeds '{node}', which is not a state node "
                                 f"(target, measured or latent).")
            if name in states or name in self.observed:
                raise ValueError(f"'{name}' cannot be both a heat source and a temperature node.")
        # Every state must be connected (through states) to at least one observed temperature,
        # otherwise the steady state is undefined (the network has no heat sink).
        adjacency = {n: set() for n in states}
        for a, b in self.all_edges:
            for x, y in ((a, b), (b, a)):
                if x in adjacency:
                    adjacency[x].add(y)
        for start in states:
            seen, stack, grounded = {start}, [start], False
            while stack and not grounded:
                for nxt in adjacency[stack.pop()]:
                    if nxt not in adjacency:
                        grounded = True
                        break
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            if not grounded:
                raise ValueError(f"State node '{start}' is not connected to any observed "
                                 f"temperature; add a neighbour such as 'outside_temp_c_avg'.")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, target: str, config_path: Optional[str] = None,
                    **overrides) -> "ThermalNetworkSpec":
        """
        Default network for ``target`` from the ``thermal_networks`` section of
        sensor_config.yaml (falls back to the ``default`` entry). Keyword arguments
        (``neighbours``, ``sources``, ``latent``, ``edges``, ``measured_states``) that are not
        None replace the configured values. ``sources`` may also be a list (linear, into the
        target); ``measured_states`` are added with ``with_measured_states``.
        """
        with open(config_path or _DEFAULT_CONFIG, "r") as f:
            networks = (yaml.safe_load(f) or {}).get("thermal_networks", {}) or {}
        entry = dict(networks.get(target) or networks.get("default") or {})
        entry.update({k: v for k, v in overrides.items() if v is not None})
        sources = entry.get("sources", {})
        if isinstance(sources, (list, tuple)):
            sources = {name: "linear" for name in sources}
        spec = cls(target=target, neighbours=entry.get("neighbours", []), sources=sources,
                   latent=entry.get("latent", []), edges=entry.get("edges", []))
        return spec.with_measured_states(entry.get("measured_states", []))

    def restrict_to(self, columns: Sequence[str], warn: bool = True) -> "ThermalNetworkSpec":
        """Drop observed temperatures, measured states and sources that are not in ``columns``."""
        columns = set(columns)
        missing = [c for c in self.required_columns if c not in columns]
        if warn and missing:
            warnings.warn(f"Thermal network for '{self.target}': dropping unavailable "
                          f"sensors {missing}.", stacklevel=2)
        measured = [n for n in self.measured_states if n in columns]
        states = {self.target, *measured, *self.latent}
        keep = lambda n: n in states or (n in columns and n not in self.measured_states)  # noqa: E731
        return ThermalNetworkSpec(
            target=self.target,
            neighbours=[n for n in self.neighbours if keep(n)],
            sources=_links_to_sources([l for l in self.source_links if l[0] in columns and l[1] in states]),
            latent=self.latent,
            edges=[e for e in self.edges if keep(e[0]) and keep(e[1])],
            measured_states=measured,
        )

    def with_measured_states(self, columns: Sequence[str]) -> "ThermalNetworkSpec":
        """
        Copy in which ``columns`` are measured states coupled to the target and to each other and
        sharing the target's observed neighbours and heat sources (e.g. a second bearing in the
        same housing). A column that was a neighbour of the target becomes a state.
        """
        new = [c for c in columns if c != self.target and c not in self.measured_states]
        if not new:
            return self
        states = self.states + new
        env = [b if a == self.target else a for a, b in self.all_edges
               if self.target in (a, b) and (b if a == self.target else a) not in states]
        edges = list(self.edges)
        for k, c in enumerate(new):
            edges += [(c, self.target)] + [(c, o) for o in env] + [(c, d) for d in self.measured_states + new[:k]]
        links = self.source_links + [(name, c, t) for name, node, t in self.source_links
                                     if node == self.target for c in new]
        return replace(self, neighbours=[n for n in self.neighbours if n not in new], edges=edges,
                       sources=_links_to_sources(links), measured_states=self.measured_states + new)

    def retarget(self, target: str) -> "ThermalNetworkSpec":
        """
        The same network with the measured state ``target`` as target; the old target becomes a
        measured state (e.g. to predict the second bearing of a two-bearing network).
        """
        if target == self.target:
            return self
        if target not in self.measured_states:
            raise ValueError(f"'{target}' is not a measured state of the network.")
        return replace(self, target=target, neighbours=[],
                       edges=[(self.target, n) for n in self.neighbours] + list(self.edges),
                       sources=_links_to_sources(self.source_links),
                       measured_states=[self.target] + [m for m in self.measured_states if m != target])

    def as_boundary(self) -> "ThermalNetworkSpec":
        """Copy in which the measured states are observed boundary temperatures (no own balance,
        their heat sources are dropped); used by models without an observation model (PCRNN)."""
        if not self.measured_states:
            return self
        dropped = [l for l in self.source_links if l[1] in self.measured_states]  # no own balance
        states = {self.target, *self.latent}
        to_target = [m for m in self.measured_states if any({m, self.target} == set(e) for e in self.edges)]
        return replace(self, measured_states=[], neighbours=self.neighbours + to_target,
                       edges=[e for e in self.edges if states & set(e) and set(e) - {self.target} - set(to_target)],
                       sources=_links_to_sources([l for l in self.source_links if l not in dropped]))

    def equation(self, residual: bool = False) -> str:
        """Heat balance of the target node as a readable string."""
        terms = []
        for a, b in self.all_edges:
            if self.target in (a, b):
                other = b if a == self.target else a
                terms.append(f"G·(T[{other}] − T)")
        for name, node, transform in self.source_links:
            if node == self.target:
                terms.append(f"β·{name}" + ("²" if transform == "square" else ""))
        if residual:
            terms.append("q_NN(x)")
        return "C dT/dt = " + " + ".join(terms) if terms else "C dT/dt = 0"


def _links_to_sources(links) -> Dict[str, SourceSpec]:
    """Inverse of ``source_links``: {column: [(node, transform), ...]}."""
    out: Dict[str, list] = {}
    for name, node, transform in links:
        out.setdefault(name, []).append((node, transform))
    return out


# ---------------------------------------------------------------------------- plotting
# Colour = role (red: predicted temperatures and their sensors, blue: helpers), fill = observation
# (filled: measured, open: latent), shape: circle = temperature, square = heat input.
_RED, _BLUE, _EDGE, _ARROW = "#E4572E", "#4C78A8", "#BDBDBD", "#7F7F7F"
_SCHEME = ("red: predicted temperatures (and their sensors), blue: helpers; filled: measured, open: latent; "
           "circle: temperature, square: heat input")
_SIDE_ANGLES = (180, 0, 205, 335, 155, 25)
_PX = 80  # nominal pixels per data unit (sets the figure height)


@dataclass
class _Node:
    group: str                 # state | sensor | sink | source | attached (heat input of one state)
    label: str                 # text in the key; "{of}" is replaced by the symbol of ``of``
    predicted: bool = False
    measured: bool = False
    of: Optional[str] = None   # state of a sensor / attached input
    symbol: str = ""           # fixed symbol, or a template with "{i}" (index of ``of``)


@dataclass
class _Drawing:
    title: str
    nodes: Dict[str, _Node]                           # predicted states first
    heat: list = field(default_factory=list)          # (a, b, width value or None, label or label(idx))
    inputs: list = field(default_factory=list)        # (input, state, label)
    sensors: list = field(default_factory=list)       # (state, sensor, label)
    taus: Dict[str, float] = field(default_factory=dict)
    equations: Optional[callable] = None              # (idx, sym) -> lines
    footer: List[str] = field(default_factory=list)


def _seg_dist(p, q, x) -> float:
    """Distance of point x from the segment p-q."""
    p, q, x = (np.asarray(v, float) for v in (p, q, x))
    t = np.clip(np.dot(x - p, q - p) / max(np.dot(q - p, q - p), 1e-12), 0, 1)
    return float(np.linalg.norm(p + t * (q - p) - x))


def _free_arc(names, links, pos, lo, hi, r=2.9):
    """Points on an arc (left to right) whose links to their states keep clear of other nodes."""
    if not names:
        return []
    cand = np.linspace(lo, hi, 61)
    targets = {s for k in names for s in links[k] if s in pos}

    def clearance(a):
        p = (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
        return min((_seg_dist(p, pos[s], q) for s in targets for o, q in pos.items() if o != s), default=9.0)

    score = np.array([clearance(a) for a in cand])
    for c in (0.55, 0.4, 0.25, -1.0):  # prefer clear links, then at least 1.1 units between nodes
        ok = cand[score >= c]
        if len(ok) < len(names):
            continue
        picks = ok[np.round(np.linspace(0, len(ok) - 1, len(names))).astype(int)] if len(names) > 1 else [ok[len(ok) // 2]]
        if len(names) == 1 or np.min(np.abs(np.diff(picks))) * math.pi / 180 * r >= 1.1:
            break
    return [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for a in picks]


def _radius(d: _Drawing) -> float:
    """Arc radius: wider when there are several state nodes."""
    n_states = sum(n.group == "state" for n in d.nodes.values())
    return 2.9 + 0.7 * max(n_states - 1, 0)


def _place(d: _Drawing) -> Dict[str, Tuple[float, float]]:
    """Predicted states side by side in the centre, helper states beside them, sensors below and
    attached inputs above their state, sinks on the upper and sources on the lower arc."""
    N, r = d.nodes, _radius(d)
    main = [k for k, n in N.items() if n.group == "state" and n.predicted]
    other = [k for k, n in N.items() if n.group == "state" and not n.predicted]
    pos = {k: ((j - (len(main) - 1) / 2) * 2.8, 0.0) for j, k in enumerate(main)}
    for j, k in enumerate(other):
        a, rs = math.radians(_SIDE_ANGLES[j % 6]), r - 1.2 + 0.5 * (j // 6)
        pos[k] = (rs * math.cos(a), rs * math.sin(a))
    for k, n in N.items():
        if n.group in ("sensor", "attached"):
            x, y = pos[n.of]
            pos[k] = (x, y + (-1.15 if n.group == "sensor" else 1.15))
    links = {k: [] for k in N}
    for a, b, *_ in d.heat + d.inputs:
        links[a].append(b)
        links[b].append(a)
    for group, lo, hi in (("sink", 165, 15), ("source", 200, 340)):
        names = [k for k, n in N.items() if n.group == group]
        names.sort(key=lambda k: np.mean([pos[s][0] for s in links[k] if s in pos] or [0.0]))
        pos.update(zip(names, _free_arc(names, links, pos, lo, hi, r)))
    return pos


def _wrap(lhs: str, terms: List[str], per_line: int = 4) -> List[str]:
    def join(ts):  # "a + −b" -> "a − b"
        return "".join(t if not j else (f" − {t[1:]}" if t.startswith("−") else f" + {t}") for j, t in enumerate(ts))
    chunks = [join(terms[j:j + per_line]) for j in range(0, len(terms), per_line)] or ["0"]
    return [f"{lhs} = {chunks[0]}"] + [("        − " + c[1:]) if c.startswith("−") else "        + " + c
                                       for c in chunks[1:]]


def _from_spec(obj) -> _Drawing:
    """PCRNN forecaster or ThermalNetworkSpec."""
    if isinstance(obj, ThermalNetworkSpec):
        spec, params, residual = obj, None, False
    else:
        spec = getattr(obj, "network_", None)
        params = obj.thermal_parameters_ if spec is not None else None
        spec = spec or getattr(obj, "network", None)
        if spec is None:
            raise ValueError("The model has no network yet: fit it first or pass a ThermalNetworkSpec.")
        residual = bool(getattr(obj, "residual", False)) and getattr(obj, "mode", "hard") == "hard"
    T = spec.target
    key = {n: f"s:{n}" for n in spec.states}
    key.update({n: f"o:{n}" for n in spec.observed})
    squared = {name for name, _, t in spec.source_links if t == "square"}
    # no observation model: the simulated T0 is the prediction of the sensor, fitted to it directly
    note = "" if isinstance(obj, ThermalNetworkSpec) else "; its measurement is only the fit target"
    N = {key[T]: _Node("state", f"{T} (predicted{note})", True, True)}
    N.update({key[n]: _Node("state", f"{n} (measured state)", measured=True) for n in spec.measured_states})
    N.update({key[n]: _Node("state", f"{n} (latent)") for n in spec.latent})
    N.update({key[n]: _Node("sink", n, measured=True) for n in spec.observed})
    N.update({f"u:{n}": _Node("source", n + (" (squared)" if n in squared else ""), measured=True)
              for n, _, _ in spec.source_links})
    if residual:
        N[f"q:{T}"] = _Node("attached", "residual heat into {of} (see below)", of=key[T], symbol="q<sub>NN</sub>")
    g = params["conductances"] if params else {}
    rises = params["steady_state_rise_K"] if params else {}
    cooled = {(c[0], c[1]) for c in getattr(obj, "cooling", ())}
    d = _Drawing(
        title=f"Lumped thermal network of {T}", nodes=N,
        heat=[(key[a], key[b], g.get((a, b)), f"G={g[(a, b)]:.2g}/h" if (a, b) in g else "G")
              for a, b in spec.all_edges],
        inputs=[(f"u:{n}", key[s], (f"{rises[(n, s)]:+.2g} K" if (n, s) in rises else "β")
                 + (" (u²)" if n in squared else "")) for n, s, _ in spec.source_links]
        + ([(f"q:{T}", key[T], "")] if residual else []),
        taus={key[n]: v for n, v in (params["time_constants_h"] if params else {}).items()})

    def equations(idx, sym):
        lines = []
        for n in spec.states:
            i, k = idx[key[n]], key[n]
            others = sorted((key[b if a == n else a] for a, b in spec.all_edges if n in (a, b)), key=idx.get)
            terms = [f"G<sub>{i},{idx[o]}</sub>({sym[o]} − {sym[k]})" for o in others]
            terms += [("−" if (name, s) in cooled else "") + f"β<sub>{i},{idx['u:' + name]}</sub>{sym['u:' + name]}"
                      + ("²" if name in squared else "") for name, s, _ in spec.source_links if s == n]
            terms += ["q<sub>NN</sub>(x)"] if residual and n == T else []
            lines += _wrap(f"d{sym[k]}/dt" if n == T else f"C<sub>{i}</sub> d{sym[k]}/dt", terms)
        return lines

    d.equations = equations
    if residual:
        n_feat = len(getattr(obj, "feature_cols", None) or [])
        d.footer.append(f"q<sub>NN</sub>(x) = residual heat in K/h, learned by an MLP from {n_feat or 'the other'} "
                        "sensor features x (L2-penalised)")
    d.footer.append("G, β per unit heat capacity of the predicted temperature (C<sub>0</sub> = 1)"
                    + ("; C<sub>i</sub> relative to it" if spec.latent else ""))
    return d


def _from_kalman(model, monitored: Optional[List[str]] = None, tracking: bool = True) -> _Drawing:
    """Fitted KalmanThermalForecaster; ``monitored`` = monitored temperatures of a detector
    (``tracking``: KalmanHeatDetector, else free-run residual detector)."""
    free = getattr(model, "training", None) == "free"
    tracking = bool(monitored) and tracking
    if getattr(model, "params", None) is None:
        raise ValueError("The Kalman model is not fitted yet.")
    p, spec = model.thermal_parameters_, model.network_
    pred = list(monitored) if monitored else [spec.target]
    states = pred + [a for a in model.states_ if a not in pred]
    measured = set(model.sensors_)
    N = {}
    for a in states:
        role = " (monitored)" if monitored and a in pred else " (predicted)" if a in pred else ""
        N[f"s:{a}"] = _Node("state", (f"true temperature of {a}" if a in measured else f"{a} (latent)") + role,
                            predicted=a in pred)
        if a in measured:
            fit_only = not monitored and a == spec.target
            N[f"y:{a}"] = _Node("sensor", f"{a} (sensor of {{of}}{', fit only' if fit_only else ''})",
                                a in pred, True, f"s:{a}", "y<sub>{i}</sub>")
    N.update({f"o:{o}": _Node("sink", o, measured=True) for o in model.sinks_})
    gains = p["steady_state_rise_K"]
    N.update({f"u:{u}": _Node("source", u.replace("²", " (squared)"), measured=True) for u, _ in gains})
    heated = [a for a in states if a in measured] if model.const_heat else []
    N.update({f"c:{a}": _Node("attached", "constant heat into {of}" + (" (random walk)" if tracking else " (fitted)"),
                              of=f"s:{a}", symbol="c<sub>{i}</sub>") for a in heated})
    heat = [(f"s:{a}", f"o:{o}", k, f"k={k:.2g}/h") for (a, o), k in p["sink_rates"].items()]
    for i, a in enumerate(states):
        for b in states[i + 1:]:
            if (a, b) in p["couplings"]:
                kab, kba = p["couplings"][(a, b)], p["couplings"][(b, a)]
                heat.append((f"s:{a}", f"s:{b}", max(kab, kba), lambda idx, a=a, b=b, kab=kab, kba=kba:
                             f"k<sub>{idx['s:' + a]},{idx['s:' + b]}</sub>={kab:.2g}<br>"
                             f"k<sub>{idx['s:' + b]},{idx['s:' + a]}</sub>={kba:.2g} /h"))
    inputs = [(f"u:{u}", f"s:{a}", f"{rise:+.2g} K") for (u, a), rise in gains.items()]  # < 0: cooling
    inputs += [(f"c:{a}", f"s:{a}", f"{p['const_heat'][a]:+.2g} K/h") for a in heated]
    d = _Drawing(
        title=("Kalman heat detector network" if tracking else "Free-run residual detector network"
               if monitored else f"Kalman heat network of {spec.target}"),
        nodes=N, heat=heat, inputs=inputs,
        sensors=[(f"s:{a}", f"y:{a}", "not assimilated" if free else f"σ={p['sensor_noise_K'][a]:.2g} K")
                 for a in states if a in measured],
        taus={f"s:{a}": v for a, v in p["time_constants_h"].items()})

    def equations(idx, sym):
        out = sym.get(f"o:{model.outside}", "T<sub>out</sub>")
        lines = []
        for a in states:
            k, i = f"s:{a}", idx[f"s:{a}"]
            others = sorted((("s:" if b in model.states_ else "o:") + b
                             for e in spec.all_edges if a in e for b in e if b != a), key=idx.get)
            terms = [f"k<sub>{i},{idx[o]}</sub>({sym[o]} − {sym[k]})" for o in others]
            terms += [("−" if p["source_gains"][(u, b)] < 0 else "") + f"β<sub>{i},{idx['u:' + u]}</sub>{sym['u:' + u]}"
                      for u, b in gains if b == a]
            if a in measured:
                terms += [f"c<sub>{i}</sub>"] * model.const_heat + [f"γ<sub>{i}</sub>({out} − 10)"] * model.temp_heat
            lines += _wrap(f"d{sym[k]}/dt", terms)
        lines.append("y<sub>i</sub> = T<sub>i</sub> + noise" + (", c<sub>i</sub>: random walks" if tracking else "")
                     + (" (free run: y used only as initial value)" if free else ""))
        return lines

    d.equations = equations
    d.footer.append("k, β, c, γ per unit heat capacity of the receiving state (1/h, K/h)"
                    + ("; alarm indicator: steady-state excess −A<sup>−1</sup>c" if tracking else
                       "; alarm indicator: y − free-run simulation of T" if monitored else ""))
    return d


def plot_thermal_network(obj, title: Optional[str] = None) -> go.Figure:
    """
    Draw the heat network of a ``ThermalNetworkSpec``, a ``PCRNNForecaster``, a fitted
    ``KalmanThermalForecaster`` or a fitted ``KalmanHeatDetector`` / ``FreeRunResidualDetector``.

    Same scheme for all models: colour = role (red: predicted temperatures and their sensors,
    blue: helpers), fill = observation (filled: measured, open: latent), circle = temperature,
    square = heat input; solid lines exchange heat, dashed lines link a temperature to its sensor,
    arrows are heat inputs. Symbols (T = temperatures, y = sensors, u = loads, c/q = heat inputs)
    are explained below the network and the heat balances are written above it. For fitted models
    the lines show the learned conductances or rates, the arrows the steady-state rise at full load
    (or the constant heat), and the predicted states their time constants.

    :param obj: ThermalNetworkSpec, PCRNNForecaster, KalmanThermalForecaster or KalmanHeatDetector
    :param title: Optional figure title
    :return: plotly Figure
    """
    if hasattr(getattr(obj, "model", None), "track_heat"):  # KalmanHeatDetector / FreeRunResidualDetector
        d = _from_kalman(obj.model, monitored=list(obj.monitored), tracking=hasattr(obj, "ss_std_per_day"))
    elif hasattr(obj, "track_heat"):                         # KalmanThermalForecaster
        d = _from_kalman(obj)
    else:
        d = _from_spec(obj)
    N, pos, r = d.nodes, _place(d), _radius(d)

    # symbols: T for states then sinks, y / c with the index of their state, u for sources
    temps = [k for k, n in N.items() if n.group == "state"]
    temps += sorted((k for k, n in N.items() if n.group == "sink"), key=lambda k: pos[k][0])
    idx = {k: i for i, k in enumerate(temps)}
    idx.update({k: j + 1 for j, k in enumerate(sorted((k for k, n in N.items() if n.group == "source"),
                                                       key=lambda k: pos[k][0]))})
    sym = {}
    for k, n in N.items():
        if n.group in ("state", "sink"):
            sym[k] = f"T<sub>{idx[k]}</sub>"
        elif n.group == "source":
            sym[k] = f"u<sub>{idx[k]}</sub>"
        else:
            sym[k] = n.symbol.format(i=idx[n.of])
    fig = go.Figure()
    annotations, requests = [], []  # requests: (candidate positions, text, colour), placed below
    request = lambda cands, text, color="#555555": requests.append((cands, text, color))  # noqa: E731

    def along(p, q):  # candidate label positions along a line, middle first
        return [(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])) for t in (0.5, 0.4, 0.6, 0.3, 0.7, 0.22, 0.78)]

    def beside(x, y):
        return [(x + dx, y + dy) for dy in (0, 0.22, -0.22) for dx in (-0.5, 0.5)]

    # time constants of predicted states, sensor noise and attached inputs next to their state
    for k, tau in d.taus.items():
        if N[k].predicted:
            x, y = pos[k]
            request([(x + 0.6, y - 0.32), (x - 0.6, y - 0.32), (x + 0.6, y + 0.32), (x - 0.6, y + 0.32)],
                    f"τ={tau:.2g} h", "#222222")

    # heat exchange (solid, width ~ value) and sensor links (dashed)
    w_max = max((w for _, _, w, _ in d.heat if w is not None), default=1.0) or 1.0
    for a, b, w, text in d.heat:
        text = text(idx) if callable(text) else text
        (x0, y0), (x1, y1) = pos[a], pos[b]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="text", showlegend=False,
                                 line=dict(color=_EDGE, width=1.5 + 7 * w / w_max if w is not None else 2.0),
                                 text=f"{N[a].label.split(' (')[0]} ↔ {N[b].label.split(' (')[0]}<br>{text}"))
        request(along(pos[a], pos[b]), text)
    for s_, y_, text in d.sensors:
        (x0, y0), (x1, y1) = pos[s_], pos[y_]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=_ARROW, width=1.5, dash="dash")))
        if text:
            request(beside(x0, (y0 + y1) / 2), text)

    # heat inputs as arrows into their state
    for u, s_, text in d.inputs:
        (xs, ys), (xn, yn) = pos[u], pos[s_]
        annotations.append(dict(x=xn, y=yn, ax=xs, ay=ys, xref="x", yref="y", axref="x", ayref="y",
                                showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor=_ARROW,
                                text="", standoff=24, startstandoff=18))  # stop at the marker edges
        if text:
            request(beside(xn, (ys + yn) / 2) if N[u].group == "attached" else along(pos[u], pos[s_]), text)

    # greedy placement: first candidate free of nodes and already placed labels (sizes at _PX px/unit)
    boxes = [(x - 0.3, y - 0.3, x + 0.3, y + 0.3) for x, y in pos.values()]
    overlap = lambda b, c: max(0.0, min(b[2], c[2]) - max(b[0], c[0])) * max(0.0, min(b[3], c[3]) - max(b[1], c[1]))  # noqa: E731
    for cands, text, color in sorted(requests, key=lambda q: len(q[0])):  # least freedom first
        lines = text.split("<br>")
        w = (6.0 * max(len(re.sub("<[^>]+>", "", t)) for t in lines) + 8) / _PX / 2
        h = 15.0 * len(lines) / _PX / 2
        cost = [sum(overlap((x - w, y - h, x + w, y + h), b) for b in boxes) for x, y in cands]
        x, y = cands[int(np.argmin(cost))]
        boxes.append((x - w, y - h, x + w, y + h))
        annotations.append(dict(x=x, y=y, showarrow=False, text=text, font=dict(size=10, color=color),
                                bgcolor="white", borderpad=1))

    # nodes: four colour / fill classes, symbol inside the marker, time constant beside predicted states
    classes = {"predicted, latent": (True, False), "predicted, measured": (True, True),
               "helper, measured": (False, True), "helper, latent": (False, False)}
    for name, (predicted, measured) in classes.items():
        ks = [k for k, n in N.items() if (n.predicted, n.measured) == (predicted, measured)]
        if not ks:
            continue
        color = _RED if predicted else _BLUE
        hover = [f"<b>{N[k].label.format(of=sym.get(N[k].of, ''))}</b><br>{name}"
                 + (f"<br>τ = {d.taus[k]:.3g} h" if k in d.taus else "") for k in ks]
        fig.add_trace(go.Scatter(
            x=[pos[k][0] for k in ks], y=[pos[k][1] for k in ks], mode="markers+text", name=name,
            marker=dict(symbol=["square" if N[k].group in ("source", "attached") else "circle" for k in ks],
                        size=[34 if N[k].group in ("source", "attached") else 44 for k in ks],
                        color=color if measured else "white", line=dict(color=color, width=2 if measured else 3)),
            text=[sym[k] for k in ks], textposition="middle center",
            textfont=dict(size=13, color="white" if measured else color),
            hovertext=hover, hoverinfo="text", showlegend=False))

    # equations above; key (temperatures | heat inputs) and notes below
    def entry(k):
        n = N[k]
        mark = "●" if n.measured else "○"
        return (f"<span style='color:{_RED if n.predicted else _BLUE}'>{mark}</span> {sym[k]} = "
                + n.label.format(of=sym.get(n.of, "")))

    left = []
    for k in temps:
        left.append(entry(k))
        left += [entry(s) for s, n in N.items() if n.group == "sensor" and n.of == k]
    right = [entry(k) for k in sorted((k for k, n in N.items() if n.group == "source"), key=idx.get)]
    right += [entry(k) for k, n in N.items() if n.group == "attached"]
    eq_lines = d.equations(idx, sym)
    footer = d.footer + [_SCHEME]
    n_key = max(len(left), len(right))
    top, bottom = 60 + 20 * len(eq_lines), 40 + 19 * (n_key + len(footer))
    annotations.append(dict(x=0, y=1, xref="paper", yref="paper", yanchor="bottom", xanchor="left",
                            align="left", showarrow=False, text="<br>".join(eq_lines),
                            font=dict(size=13), yshift=8))
    for x, lines, shift in ((0, left, 0), (0.52, right, 0), (0, footer, 19 * n_key + 8)):
        annotations.append(dict(x=x, y=0, xref="paper", yref="paper", yanchor="top", xanchor="left",
                                align="left", showarrow=False, text="<br>".join(lines),
                                font=dict(size=12), yshift=-8 - shift))
    fig.update_layout(
        title=dict(text=title or d.title, y=0.99, yanchor="top"),
        annotations=annotations, plot_bgcolor="white", paper_bgcolor="white",
        height=int(2 * (r + 0.6) * _PX) + top + bottom, showlegend=False,
        xaxis=dict(visible=False, range=[-r - 0.8, r + 0.8]),
        yaxis=dict(visible=False, range=[-r - 0.6, r + 0.6], scaleanchor="x"),
        margin=dict(l=20, r=20, t=top, b=bottom),
    )
    return fig
