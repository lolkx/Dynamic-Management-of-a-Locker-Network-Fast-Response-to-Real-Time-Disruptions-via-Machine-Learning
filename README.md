# Dynamic Management of a Locker Network: Fast Response to Real-Time Disruptions via Machine Learning

TFM — Vehicle Routing Problem with Simultaneous Pickup and Delivery (VRPSPD) to a
parcel-locker network, extended with capacity constraints and ML-augmented
routing heuristics that anticipate locker saturation.

Four heuristics are compared:

| Heuristic | File | Savings formula |
|---|---|---|
| Standard (baseline) | `heuristic.py` | pure distance (Clarke-Wright + biased-random GRASP) |
| Dynamic | `heuristic_dynamic.py` | distance + **live** accumulated route time (no ML) |
| **Option C** | `heuristic_c.py` | distance + time + ML penalty on **static locker overflow risk** |
| **Option D** | `heuristic_d.py` | distance + time + ML penalty on **stochastic release risk** (does the locker empty out before the vehicle arrives?) |

Options C and D share the same formula shape:

```
s(i,j) = alpha · s_dist(i,j) + (1-alpha) · s_time(i,j) − beta · penalty(j, arrival_hour) · d_fallback(j)
```

differing only in what `penalty(j, hour)` is trained to predict. See `CLAUDE.md`
for the full design history, formulas, bugs found/fixed, and experimental
results (§1–§16) — that file is the authoritative record of *why* things are
built the way they are, and should be consulted before making further changes.

## Project layout

```
.
├── CLAUDE.md              # Full design log: decisions, formulas, bugs, experiment results
├── README.md              # This file
├── Articulos/              # Reference papers (Rudy 2025, etc.)
├── code/                  # All Python source (see below)
└── data/                  # Instances, generated datasets, trained models, results
```

### `code/`

**Core model & I/O**
- `model.py` — `Node`, `Edge`, `Route`, `Solution` data structures.
- `instance_reader.py` — parses Rudy (2025)-format `.txt` instances; feature-column
  schemas (`FEATURE_COLS_B/C/D`); `MAX_COMPARTMENTS`.
- `speed_profile.py` — shared 24h time-dependent speed table.
- `simulator.py` — post-hoc locker-capacity/fallback simulation.
  `simulate_solution`/`simulate_solution_with_overflow` (deterministic, fixed
  capacity all day) and `simulate_solution_stochastic` (Option D's background
  self-collection release model — censored Normal release-hour sampling).

**Heuristics**
- `heuristic.py` — standard BR-CWS + GRASP baseline.
- `heuristic_learn.py` — shared time/ML utilities reused by Options C/D
  (`_compute_arrival_times`, `_fallback_distances`, `load_model`, `expected_fallback_km`).
- `heuristic_dynamic.py` — adds live accumulated route time (no ML); also used
  as the route-bootstrap for Option C's label generation.
- `heuristic_c.py` — Option C (static overflow-risk penalty).
- `heuristic_d.py` — Option D (stochastic release-risk penalty); reuses
  Option C's merge core (`br_CWS_c`) unchanged.

**ML training pipeline (per option)**
- `ml_common.py` — shared model factories (RF/GBM/LR/XGBoost) and CV scoring.
- `generate_instances_B.py` — generates high-saturation synthetic instances
  (`data/instances_B/`), porting `instance.h`'s (reference-only, not compiled)
  random-generation algorithm to Python.
- `generate_labels_C.py` / `train_model_C.py` — real-route overflow labels
  (bootstrapped via `heuristic_dynamic`) → Option C model. Also hosts
  `split_all_instances`, the shared 80/20 train/test split used by **both**
  Option C and Option D.
- `generate_labels_D.py` / `train_model_D.py` — Monte-Carlo `delivered_ok`
  labels (via `simulate_solution_stochastic`, one fixed route × N replicas)
  → Option D model.
- `run_experiments_C.py` / `run_experiments_D.py` — 3-way isolated comparison
  (standard / time-only / time+ML) on the held-out 20% test split, low- and
  high-occupancy phases. `run_experiments_D.py` additionally reports a
  **stochastic** evaluation (matching the training assumption) alongside the
  deterministic one — see `CLAUDE.md` §15 for why one metric alone is
  misleading.
- `run_sensitivity.py` — OFAT sensitivity sweeps (p_bias, departure_h, n_iter,
  and Option C's `sweep_alpha_c`) for a single instance.

**Reference (not executed)**
- `instance.h` — original C++ instance generator being ported by
  `generate_instances_B.py`; kept for provenance, not compiled in this repo.
- `main.tex` — thesis manuscript source. (Lives here rather than a separate
  `doc/` folder — worth relocating if you reorganize further.)

### `data/`

```
data/
├── instances/          # 320 real Rudy (2025) benchmark instances (low occupancy)
├── instances_B/        # 324 synthetic high-saturation instances (generate_instances_B.py)
├── datasets_C/          # labels_C.csv, labels_C_round2.csv (Option C training labels)
├── datasets_D/          # labels_D.csv (Option D training labels, Monte-Carlo)
├── models_C/            # trained Option C bundles (rf/gbm/lr + best)
├── models_D/            # trained Option D bundles (rf/gbm/lr/xgb + best)
├── results_C_best_alpha0.5_beta{0.1,0.3}.csv   # Option C held-out test results
├── results_D_best_alpha0.5_beta0.3.csv          # Option D held-out test results
└── sensitivity_alpha_c.csv                      # Option C alpha ablation
```

Model A/B (percentile-label and static-overflow-label variants explored early
in the project) were retired entirely — see `CLAUDE.md` §14 for why, and §16
for the general cleanup that removed their now-dead code/data.

## Usage

```bash
# Option C: generate labels -> train -> evaluate on held-out test set
python code/generate_labels_C.py
python code/train_model_C.py --compare
python code/run_experiments_C.py --alpha 0.5 --beta 0.3

# Option D: same shape, stochastic release-risk model
python code/generate_labels_D.py --n-replicas 10
python code/train_model_D.py --compare
python code/run_experiments_D.py --alpha 0.5 --beta 0.3

# Sensitivity sweep on one instance
python code/run_sensitivity.py data/instances/<file>.txt --sweep alpha_c
```

Note: `data/*.csv` files are working artifacts. Per the convention at the top
of `CLAUDE.md`, don't read them directly for context — the accompanying
prose in `CLAUDE.md` already summarizes what each result means.
