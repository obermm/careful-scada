# Kalman thermal network studies (CARE Wind Farm B, main bearings)

Research experiments behind the Kalman model and the detectors of `care_scada`. New users should
start with `examples/basic_usage.ipynb`; this folder documents how the defaults used there were found.

```
Kalman_results/
├── results/        what was found (reports, tables, figures, raw logs) - read these
├── experiments/    the Python scripts that produced them
└── runners/        SLURM batch scripts that run the experiments on a cluster
```

## Studies

| | study | report | experiment scripts | runners |
|---|---|---|---|---|
| 01 | **Kalman monitoring**: Kalman heat tracker vs PCRNN / LSTM residuals with CUSUM alarms on the faulty turbines 27, 53, 77 | `results/01_kalman_monitoring/report.txt` (+ `monitor_summary.csv`, `monitor_positives.png`) | `experiments/kalman_monitoring.py` | – |
| 02 | **Thermal network search**: structure search for a free-run network of both main bearings over the 6 healthy turbines; result: networks *simple* (8 parameters) and *medium* (10) | `results/02_network_search/thermal_network_search.txt` | `experiments/network_search.py` (model, evaluation), `experiments/network_greedy.py` (greedy forward/backward driver) | `runners/netsearch_job.sh`, `runners/netsearch_submit.sh` |
| 03 | **Fault detection**: 3 networks (demo/baseline, simple, medium) × indicators × alarm routines on 27, 53, 77, false alarms on the healthy turbines (leave-one-out) and injected 1/2/3 K ramps; result: detector D4 = `DetectorFactory.create("freerun", ...)` | `results/03_fault_detection/fault_detection.txt` (+ `fault_detection_positives.png`) | `experiments/fault_detection.py` (fits, indicators), `experiments/fault_eval.py` (alarm routines, tables), `experiments/d4_factory_check.py` (library D4 = study result) | `runners/fd_job.sh`, `runners/d4check_job.sh` |

`results/02_network_search/raw/` holds the per-network results (`*.jsonl`, one line per network,
dataset and seed) and the traces of the greedy searches (`*_trace.txt`).

## Running

The experiment scripts import the installed package (`pip install -e ".[deep]"` in the repository
root) and import each other, so run them from `experiments/`:

```bash
cd examples/Kalman_results/experiments
python network_search.py one "mb1-gb1 mb1-out mb2-gb1 mb2-out | -ws>mb1,mb2 rot2>mb1,mb2 |"   # one network
python fault_eval.py --out ~/.cache/care_faultdet/r2 --final                                   # tables of study 03
```

The runners are SLURM scripts (CPU only). Set `CARE_REPO` (repository root) and `CARE_PYTHON` (Python
of the environment) if they differ from the defaults at the top of each script; logs go to
`<job name>_<job id>.out` in the submit directory. Example (study 03, 42 fit tasks in 5 shards):

```bash
for i in 0 1 2 3 4; do sbatch runners/fd_job.sh $i/5 ~/.cache/care_faultdet/r2; done
python experiments/fault_eval.py --out ~/.cache/care_faultdet/r2 --final
```

Intermediate data (preprocessed series, indicators) are cached under `~/.cache/care_*`; the paths
are set at the top of the experiment scripts.
