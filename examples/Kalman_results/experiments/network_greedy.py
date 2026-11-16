"""Greedy forward / backward network search on SLURM (driver for network_search.py).

A network is a set of tokens of the network_search notation: edges "a-b", source links
"s>n" / "s2>n" / "-s>n" and flags "c", "g". Every forward step evaluates current + one element
of the pool (an element may be a group of tokens, e.g. a latent node with its links), every
backward step current - one token. Selection by the mean validation RMSE of both bearings
(last 25 % of each healthy train part); test RMSE is only recorded.

    python network_greedy.py --name g1 --start "<net>" --pool pool.txt [--min-gain 0.005]

pool.txt: one element per line (tokens separated by spaces; a pool element that mentions a
latent node only becomes available once that node exists, unless it also grounds it).
"""

import argparse
import json
import os
import subprocess
import time

import network_search as ns

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "02_network_search", "raw")
RUNNERS = os.path.join(HERE, "..", "runners")
PARTS = os.path.expanduser("~/.cache/care_netsearch/parts")


def tokens(net: str) -> list:
    parts = [p.split() for p in (net.split("|") + ["", ""])[:3]]
    src = [f"{s.split('>')[0]}>{n}" for s in parts[1] for n in s.split(">")[1].split(",")]
    return parts[0] + src + parts[2]


def to_net(toks) -> str:
    toks = list(dict.fromkeys(toks))
    src = [t for t in toks if ">" in t]
    flags = [t for t in toks if t in ("c", "g")]
    edges = [t for t in toks if t not in src and t not in flags]
    key = lambda t: t  # noqa: E731
    return f"{' '.join(sorted(edges, key=key))} | {' '.join(sorted(src, key=key))} | {' '.join(sorted(flags))}"


def valid(net: str) -> bool:
    from care_scada.modeling.thermal_network import ThermalNetworkSpec
    p = ns.parse(net)
    if not p["edges"]:
        return False
    try:
        ThermalNetworkSpec(target=ns.TARGET, sources=p["sources"], latent=p["latent"],
                           edges=p["edges"], measured_states=[ns.MB1])
    except ValueError:
        return False
    states = {ns.TARGET, ns.MB1, *p["latent"]}
    linked = {n for e in p["edges"] for n in e}
    if not all(s in linked for s in states):  # every state has at least one edge
        return False
    # every latent node must reach a bearing through state-state edges (else it is irrelevant)
    reach, grew = {ns.TARGET, ns.MB1}, True
    while grew:
        new = {b if a in reach else a for a, b in p["edges"] if (a in reach) != (b in reach)
               and (a in states and b in states)}
        grew = bool(new - reach)
        reach |= new
    return set(p["latent"]) <= reach


def evaluate(nets, stage, n_jobs=5, seeds="0,1,2"):
    """Run nets on SLURM (cached: nets already in RES/*.jsonl are not rerun); summary per net."""
    done = {}
    rows = []
    for f in os.listdir(RES):
        if f.endswith(".jsonl"):
            for l in open(os.path.join(RES, f)):
                r = json.loads(l)
                if ("val_mb1" in r or "error" in r) and str(r.get("seed")) in seeds.split(","):
                    rows.append(r)
                    done[r["net"]] = done.get(r["net"], set()) | {(r["ds"], r["seed"])}
    todo = [n for n in nets if len(done.get(n, ())) < 6 * len(seeds.split(","))]
    if todo:
        os.makedirs(os.path.join(PARTS, stage), exist_ok=True)
        k = min(n_jobs, len(todo))
        procs = []
        for i in range(k):
            part = os.path.join(PARTS, stage, f"part{i}")
            with open(part, "w") as f:
                f.write("\n".join(todo[i::k]) + "\n")
            procs.append(subprocess.Popen(
                ["sbatch", "--wait", "-p", "l40,rtx8000", os.path.join(RUNNERS, "netsearch_job.sh"), part,
                 os.path.join(RES, f"{stage}_part{i}.jsonl"), seeds, "2,21,52,74,82,83",
                 json.dumps({"val_frac": 0.25})], cwd=HERE, stdout=subprocess.DEVNULL))
        for p in procs:
            p.wait()
        for i in range(k):
            path = os.path.join(RES, f"{stage}_part{i}.jsonl")
            if os.path.exists(path):
                rows += [json.loads(l) for l in open(path)
                         if str(json.loads(l).get("seed")) in seeds.split(",")]
    s = ns.summarise([r for r in rows if r["net"] in set(nets)])
    return s[s.n_ds == 6]


def log(msg, path):
    print(msg, flush=True)
    with open(path, "a") as f:
        f.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--min-gain", type=float, default=0.015, help="K of mean val RMSE to accept an addition")
    ap.add_argument("--drop-tol", type=float, default=0.008, help="K of val RMSE a removal may cost")
    ap.add_argument("--max-steps", type=int, default=30)
    a = ap.parse_args()
    pool = [l.split() for l in open(a.pool) if l.strip() and not l.startswith("#")]
    trace = os.path.join(RES, f"{a.name}_trace.txt")
    cur = tokens(a.start)
    base = evaluate([to_net(cur)], f"{a.name}_s0")
    best = float(base.val.iloc[0])
    log(f"[{a.name}] start val={best:.3f} test={base.test.iloc[0]:.3f} npar={base.n_par.iloc[0]} :: {to_net(cur)}", trace)
    for step in range(1, a.max_steps + 1):
        # backward: remove one token if it costs < drop_tol
        cands = {to_net([t for t in cur if t != x]): ("-", x) for x in cur}
        cands = {n: v for n, v in cands.items() if valid(n)}
        # forward: add one pool element
        fw = {}
        for el in pool:
            if all(t in cur for t in el):
                continue
            n = to_net(cur + el)
            if valid(n):
                fw[n] = ("+", " ".join(el))
        res = evaluate(list(cands) + list(fw), f"{a.name}_s{step}")
        res["move"] = [(cands.get(n) or fw.get(n)) for n in res.index]
        rm = res[[n in cands for n in res.index]].sort_values("val")
        ad = res[[n in fw for n in res.index]].sort_values("val")
        for n, r in res.sort_values("val").head(8).iterrows():
            log(f"  s{step} {r.move[0]}{r.move[1]:<28s} val={r.val:.3f} test={r.test:.3f} npar={r.n_par}", trace)
        if len(rm) and rm.val.iloc[0] <= best + a.drop_tol:
            cur, best, why = tokens(rm.index[0]), min(best, float(rm.val.iloc[0])), f"drop {rm.move.iloc[0][1]}"
        elif len(ad) and ad.val.iloc[0] < best - a.min_gain:
            cur, best, why = tokens(ad.index[0]), float(ad.val.iloc[0]), f"add {ad.move.iloc[0][1]}"
        else:
            log(f"[{a.name}] converged at step {step}: val={best:.3f} :: {to_net(cur)}", trace)
            break
        r = res.loc[to_net(cur)]
        log(f"[{a.name}] step {step}: {why} -> val={r.val:.3f} test={r.test:.3f} npar={r.n_par} :: {to_net(cur)}", trace)


if __name__ == "__main__":
    main()
