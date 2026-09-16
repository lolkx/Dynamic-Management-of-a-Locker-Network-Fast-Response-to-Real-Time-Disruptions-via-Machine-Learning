# Dynamic Management of a Locker Network — Fast Response to Real-Time Disruptions via Machine Learning

**Master's Thesis (TFM) · Guillermo Gracia Rebullida · 2025–2026**

This repository contains the full implementation of a **learnheuristic** for the
*Parcel Locker-Based Delivery Problem* (PLBDP): a Vehicle Routing Problem with
Simultaneous Pickup and Delivery where each customer node is an automated locker
station subject to heterogeneous compartment capacities and stochastic occupancy.

Benchmark: [Rudy (2025)](https://doi.org/10.24425/acs.2025.156309) — 320 instances,
70–7 177 orders, eight Polish cities.

---

## Problem Overview

Classical routing heuristics plan a **static route** at the start of the day,
ignoring that locker availability changes as customers collect their parcels.
When a courier arrives and the locker is full, the vehicle must divert to the
nearest alternative (**fallback redirection**), increasing total travel distance.

This work addresses that gap with three increasingly sophisticated approaches:

| Method | Savings formula | ML signal |
|--------|----------------|-----------|
| **Standard** (BR-CWS) | distance only | — |
| **HSR** (Heuristic with Static Risk) | dist + time − β·P(overflow\|j)·d_fallback | trained on simulated overflow labels |
| **SRA** (Stochastic Release Aware) | dist + time − β·P(fail\|j, hour)·d_fallback | trained on Monte-Carlo customer-pickup replicas |

All three are wrapped in a **GRASP** multi-start framework (100 iterations, geometric
bias parameter p = 0.25, K = 20 nearest neighbours).

---

## Repository Structure

```
code/
├── model.py                  # Data classes: Node, Edge, Route, Solution
├── instance_reader.py        # Parser for Rudy .txt instances + feature extraction
├── speed_profile.py          # Time-dependent speed profile (Table 5, Rudy 2025)
├── heuristic.py              # Standard BR-CWS + GRASP baseline
├── heuristic_dynamic.py      # Time-aware variant (real accumulated route times)
├── heuristic_c.py            # HSR learnheuristic (Option C)
├── heuristic_d.py            # SRA learnheuristic (Option D)
├── heuristic_learn.py        # Legacy Model A/B learnheuristic (archived)
├── simulator.py              # Discrete-event simulator: fallback routing +
│                             #   stochastic customer-pickup events (Monte-Carlo)
├── ml_common.py              # Shared ML utilities (overflow label, bin-packing)
├── generate_labels_C.py      # Training data for HSR (real overflow from simulation)
├── generate_labels_D.py      # Training data for SRA (Monte-Carlo release replicas)
├── train_model_C.py          # Train RF/LR classifier for HSR; saves models_C/
├── train_model_D.py          # Train RF classifier for SRA; saves models_D/
├── run_sensitivity.py        # One-factor-at-a-time sensitivity sweeps (§6.1)
├── run_experiments_C.py      # Held-out comparison: Standard vs HSR (§6.2)
├── run_experiments_D.py      # Held-out comparison: Standard vs SRA (§6.2)
├── compare_dynamic_priority.py  # α × β grid + dynamic-priority mechanism
├── compare_rudy.py           # HVI comparison vs Rudy baseline (§6.2.4)
└── sensitivity_release_distribution.py  # Release-distribution robustness study

data/
├── instances/                # Rudy benchmark instances (320 .txt files)
├── instances_real_highocc/   # Synthetic high-saturation instances
├── datasets_C/               # HSR training data (CSV, one row per locker visit)
├── datasets_D/               # SRA training data (Monte-Carlo replicas)
├── models_C/                 # Serialised HSR classifiers (.pkl)
├── models_D/                 # Serialised SRA classifiers (.pkl)
└── results_*.csv             # Experiment outputs

Tfm/                          # LaTeX source of the dissertation
```

---

## Setup

**Requirements:** Python 3.10+, `numpy`, `scikit-learn`, `pandas`.

```bash
pip install numpy scikit-learn pandas
```

The OSRM distance/time matrices are pre-baked into the instance files by Rudy's
generator; no external routing API is needed at runtime.

---

## Quickstart

### 1 — Generate training labels

```bash
# HSR labels (real overflow from one simulated route per instance)
python code/generate_labels_C.py

# SRA labels (Monte-Carlo customer-pickup replicas, slower)
python code/generate_labels_D.py --n_replicas 30
```

### 2 — Train classifiers

```bash
python code/train_model_C.py   # saves data/models_C/model_C_best.pkl
python code/train_model_D.py   # saves data/models_D/model_D_best.pkl
```

### 3 — Run experiments

```bash
# Sensitivity sweeps (p_bias, departure_h, n_iter, alpha, ablation)
python code/run_sensitivity.py

# Held-out comparison (Standard vs HSR)
python code/run_experiments_C.py --alpha 0.5 --beta 0.3

# Held-out comparison (Standard vs SRA)
python code/run_experiments_D.py --alpha 0.5 --beta 0.3

# α × β grid + dynamic-priority mechanism
python code/compare_dynamic_priority.py

# HVI comparison against Rudy (2025) greedy/GA baselines
python code/compare_rudy.py
```

---

## Key Design Decisions

### Time-dependent savings

Both learnheuristics compute the time saving **live** at merge-decision time
using the route's real accumulated `tail_time`, avoiding the static pre-route
estimate that caused circular bias in earlier iterations:

```
s(i,j) = α · s_dist(i,j) + (1−α) · s_time(i,j)
        − β · risk(j, arrival_hour) · d_fallback(j)
```

where `s_time` uses the **known speed at node i** (not a fixed average) and
`risk(j, hour)` is looked up from a precomputed 24-bucket table, keeping
the merge-decision loop at O(1) per arc.

### Two learnheuristic variants

| | HSR (Option C) | SRA (Option D) |
|---|---|---|
| **Label** | Did delivery overflow? (1 simulated route) | Did delivery fail? (N Monte-Carlo pickup replicas) |
| **Key feature** | `initial_occupancy_ratio` + `arrival_hour` | same + stochastic release signal |
| **Training cost** | low | higher (N replicas × instances) |
| **Advantage** | faster to train | captures time-of-day release patterns |

### Dynamic-priority mechanism

The merge rule includes a **dynamic-priority** check: if the ML score for a
candidate arc is negative (i.e. the penalty outweighs the saving), the merge
is skipped regardless of the biased-random draw. This prevents the heuristic
from actively worsening routes under strong saturation signals.

### Stochastic simulator

`simulator.py` implements a discrete-event simulation where:
- Pre-occupied compartments are released stochastically during the day
  (Poisson process, rate calibrated to a configurable pickup distribution)
- Vehicle visits are processed in route order with real accumulated times
- Fallback redirections are resolved to the nearest locker with remaining capacity

This simulator serves both as the **evaluation oracle** (effective distance =
route distance + fallback detour) and as the **label generator** for SRA training.

---

## Results Summary

On the 20-instance held-out set (10 low-occupancy + 10 high-saturation):

| Metric | Standard | HSR (dynamic-priority) | SRA (dynamic-priority) |
|--------|----------|------------------------|------------------------|
| effective_km (high-sat) | 342.5 | **341.3** | 341.8 |
| n_fallbacks (high-sat) | 65.1 | **63.4** | 64.2 |
| effective_km (low-occ) | 1909.3 | 1901.7 | 1902.1 |

HSR with `α=0.5, β=0.3` and the dynamic-priority mechanism achieves the best
effective distance and fewest fallbacks on high-saturation instances while
matching the standard heuristic on low-occupancy instances where saturation
never occurs.

---

## Citation

If you use the benchmark instances, please cite:

```bibtex
@article{rudy2025,
  author  = {Rudy, Jarosław},
  title   = {Multi-criteria parcel locker-based vehicle routing with pickup and delivery},
  journal = {Archives of Control Sciences},
  volume  = {35},
  number  = {3},
  pages   = {505--560},
  year    = {2025},
  doi     = {10.24425/acs.2025.156309}
}
```

---

## License

This code is released for academic purposes. The Rudy (2025) benchmark instances
are distributed under the terms of the original publication.
