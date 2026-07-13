"""
generate_labels_D.py
----------------------
Generates training labels for Option D (heuristic_d.py): for each locker
actually visited in a solved route, across many stochastic replicas of the
background self-collection release process (simulator.simulate_solution_
stochastic), did every delivery order at that visit find a compartment?

Unlike Option C's label (real overflow from ONE deterministic route, no
notion of a locker's already-occupied compartments emptying over the day),
Option D's label is Monte-Carlo: the route is fixed (solved ONCE per
instance, plain heuristic.run_grasp -- no ML, no time-blend, just "real
timestamps" per the user's own spec), and n_replicas independent stochastic
replicas of the SAME route vary only the background release outcomes. Each
replica contributes one row per locker visit:
    (13 FEATURE_COLS_B columns, initial_occupancy_ratio, arrival_hour) -> delivered_ok

The model trained on this (train_model_D.py) learns
P(delivered_ok | initial_occupancy_ratio, arrival_hour) directly -- letting
the replicas vary the LABEL for a fixed input row, not an extra input
feature computed differently at train vs. serve time (see simulator.py's
module docstring and CLAUDE.md for the full rationale).

Train/test split: reuses generate_labels_C.split_all_instances with the
SAME seed/default split as Option C, so both are held out against the
IDENTICAL 20% test set -- directly comparable results (Option C's and
Option D's run_experiments_*.py both import DATA_DIR_LOW/DATA_DIR_HIGH/
TEST_SPLIT/RANDOM_SEED from generate_labels_C.py, unchanged here).

Usage
-----
    python generate_labels_D.py
    python generate_labels_D.py --max-train 20 --n-replicas 10   # quick test
    python generate_labels_D.py --mean-hour 16 --std-hour 2      # sensitivity sweep
"""

from __future__ import annotations
import os
import sys
import csv
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from instance_reader import read_full_instance, extract_features_capacity, FEATURE_COLS_D
from heuristic import run_grasp
from simulator import simulate_solution_stochastic
from generate_labels_C import split_all_instances, DATA_DIR_LOW, DATA_DIR_HIGH, TEST_SPLIT, RANDOM_SEED

_HERE     = os.path.dirname(__file__)
OUT_PATH  = os.path.join(_HERE, '..', 'data', 'datasets_D', 'labels_D.csv')

N_ITER_DEFAULT     = 20     # one representative fixed route is enough for labelling
N_REPLICAS_DEFAULT = 50
MEAN_HOUR_DEFAULT  = 14.0
STD_HOUR_DEFAULT   = 3.0
START_H_DEFAULT    = 6
END_H_DEFAULT       = 22


# =============================================================================
# PER-INSTANCE LABEL EXTRACTION
# =============================================================================

def extract_labels(filepath: str, n_iter: int, n_replicas: int, source_pool: str,
                   mean_hour: float, std_hour: float, start_h: int, end_h: int,
                   base_seed: int, verbose: bool = False) -> list[dict]:
    """
    Solve one instance ONCE with the plain standard heuristic (fixed real
    route/timestamps), then run n_replicas independent stochastic release
    replicas on that SAME route. Each replica/visit pair becomes one row:
    FEATURE_COLS_D values + 'delivered_ok' + metadata.
    """
    params, nodes, dist_matrix, orders, locker_cap = read_full_instance(filepath)
    if not locker_cap:
        return []   # no capacity data -- nothing to label

    dep_h = params.departure
    best_sol, _elapsed = run_grasp(
        nodes, dist_matrix, params.capacity, n_iter=n_iter, verbose=False)
    if best_sol is None:
        return []

    feats = extract_features_capacity(params, orders, locker_cap)

    rows: list[dict] = []
    n_delivered_ok = 0
    for rep in range(n_replicas):
        rng = random.Random(base_seed + rep)
        visit_log = simulate_solution_stochastic(
            best_sol, orders, locker_cap, dist_matrix,
            departure_h=dep_h, mean_hour=mean_hour, std_hour=std_hour,
            start_h=start_h, end_h=end_h, rng=rng)
        for visit in visit_log:
            node_id = visit['location']
            if node_id not in feats:
                continue
            row = dict(feats[node_id])
            row['initial_occupancy_ratio'] = visit['initial_occupancy_ratio']
            row['arrival_hour']            = visit['arrival_hour']
            row['delivered_ok']            = visit['delivered_ok']
            row['instance']                = params.name
            row['source_pool']             = source_pool
            row['replica']                 = rep
            rows.append(row)
            n_delivered_ok += visit['delivered_ok']

    if verbose:
        pct = 100 * n_delivered_ok / len(rows) if rows else 0.0
        print(f"  {params.name:<30} [{source_pool:<4}]  rows={len(rows):>5}  "
              f"delivered_ok={pct:.1f}%")
    return rows


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description='Generate stochastic-release labels for Option D')
    p.add_argument('--max-train', type=int, default=None,
                   help='Cap on training instances used (default: all of the 80%% split -- '
                        'use this only for a quick smoke test)')
    p.add_argument('--n-iter', type=int, default=N_ITER_DEFAULT,
                   help='GRASP iterations for the ONE fixed route per instance')
    p.add_argument('--n-replicas', type=int, default=N_REPLICAS_DEFAULT,
                   help='Stochastic release replicas per instance')
    p.add_argument('--mean-hour', type=float, default=MEAN_HOUR_DEFAULT)
    p.add_argument('--std-hour',  type=float, default=STD_HOUR_DEFAULT)
    p.add_argument('--start-h',   type=int,   default=START_H_DEFAULT)
    p.add_argument('--end-h',     type=int,   default=END_H_DEFAULT)
    p.add_argument('--data-low',  default=DATA_DIR_LOW)
    p.add_argument('--data-high', default=DATA_DIR_HIGH)
    p.add_argument('--test-split', type=float, default=TEST_SPLIT)
    p.add_argument('--seed', type=int, default=RANDOM_SEED)
    p.add_argument('--out', default=OUT_PATH)
    args = p.parse_args()

    train_items, test_items = split_all_instances(
        args.data_low, args.data_high, args.test_split, args.seed)
    if args.max_train:
        train_items = train_items[:args.max_train]

    print(f"\n{'='*70}")
    print(f"  GENERATE LABELS — Option D (stochastic pickup-release model)")
    print(f"  Train split: {len(train_items)} instances "
          f"({sum(1 for _, p in train_items if p=='low')} low + "
          f"{sum(1 for _, p in train_items if p=='high')} high)")
    print(f"  Test split (held out, NOT used here): {len(test_items)} instances")
    print(f"  n_iter (fixed route)  : {args.n_iter}")
    print(f"  n_replicas (per inst) : {args.n_replicas}")
    print(f"  Release model         : Normal(mean={args.mean_hour}h, std={args.std_hour}h) "
          f"censored to [{args.start_h}h, {args.end_h}h]")
    print(f"{'='*70}\n")

    all_rows: list[dict] = []
    t0 = time.perf_counter()
    for idx, (path, pool_name) in enumerate(train_items, 1):
        print(f"[{idx}/{len(train_items)}] ", end='')
        base_seed = args.seed * 1_000_003 + idx   # distinct, reproducible per instance
        try:
            all_rows.extend(extract_labels(
                path, args.n_iter, args.n_replicas, pool_name,
                args.mean_hour, args.std_hour, args.start_h, args.end_h,
                base_seed, verbose=True))
        except Exception as exc:
            print(f"  ERROR on {path}: {exc}")
    elapsed = time.perf_counter() - t0

    if not all_rows:
        print("No rows generated -- check that data/instances/ and/or data/instances_B/ exist.")
        return

    n_ok = sum(r['delivered_ok'] for r in all_rows)
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed:.1f}s: {len(all_rows)} rows, "
          f"{n_ok} delivered_ok ({100*n_ok/len(all_rows):.1f}%)")
    print(f"{'='*70}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = FEATURE_COLS_D + ['delivered_ok', 'instance', 'source_pool', 'replica']
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Labels -> {args.out}")


if __name__ == '__main__':
    main()
