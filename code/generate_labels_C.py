"""
generate_labels_C.py
----------------------
Generates REAL, time-conditioned training labels for Option C
(heuristic_c.py): for each locker actually visited in a solved route, was
it ACTUALLY overflowing given the hour it was really visited?

Unlike Model A (percentile label, time-independent) and Model B (binary
overflow, also time-independent -- a pure bin-packing check against total
instance demand), this label is grounded in an actual constructed route:

  1. Solve the instance with heuristic_dynamic.run_grasp_dynamic(bundle=None)
     -- the dynamic heuristic, no ML (no Option-C model exists yet; this is
     the bootstrap the user explicitly asked for, since Option C's own
     savings formula already blends distance+time the same way).
  2. Replay heuristic_learn._compute_arrival_times on the finished solution
     to get each node's REAL arrival hour (solution-agnostic, exact -- no
     new tracking needed).
  3. Run simulator.simulate_solution_with_overflow on the SAME solution to
     get REAL per-locker overflow, accounting for the actual route order and
     the shared, cascading compartment state across all routes (exactly the
     mechanism the user described: "this locker overflows because base
     occupancy is already high at that hour").
  4. Pair each node's (13 capacity-aware features, real arrival hour) with
     its (real overflowed 0/1) label.

Draws instances from BOTH data/instances/ (low occupancy, mostly-negative
labels expected) and data/instances_B/ (high saturation, meaningfully
positive labels expected) so Option C's model sees the full risk spectrum.

Train/test split (IMPORTANT): split_all_instances() below performs ONE
80/20 shuffle-split over the COMBINED pool of both directories -- the exact
same pattern as run_experiments_A.py/_B.py's split_instances -- and this
script only ever generates labels from the TRAIN 80%. run_experiments_C.py
imports the SAME function (same seed) to get the TEST 20%, so the two
scripts can never disagree about which instances are "seen" vs "held out".
This replaces an earlier version that sampled ~40+40 instances independently
in each script -- a real train/test contamination risk, since nothing
guaranteed the two independent samples didn't overlap -- and it also used
far fewer training instances than are actually available (all ~515 of the
80% split are used now, not a fixed subsample).

Usage
-----
    python generate_labels_C.py
    python generate_labels_C.py --max-train 20 --n-iter 5   # quick test

    # Round 2 (bootstrap iteration): generate training routes with the
    # CURRENT best Option C model instead of the plain dynamic heuristic,
    # so training routes better match what the model is actually scored
    # against at inference time (reduces train/serve skew on arrival_hour):
    python generate_labels_C.py --bootstrap-model ../data/models_C/saturation_C_best.pkl \
        --bootstrap-beta 0.1 --out ../data/datasets_C/labels_C_round2.csv
"""

from __future__ import annotations
import os
import sys
import csv
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from instance_reader import read_full_instance, extract_features_capacity, FEATURE_COLS_C
from heuristic_dynamic import run_grasp_dynamic
from heuristic_c import run_grasp_c, ALPHA_DEFAULT, BETA_DEFAULT, load_model
from heuristic_learn import _compute_arrival_times
from simulator import simulate_solution_with_overflow

_HERE           = os.path.dirname(__file__)
DATA_DIR_LOW    = os.path.join(_HERE, '..', 'data', 'instances')
DATA_DIR_HIGH   = os.path.join(_HERE, '..', 'data', 'instances_B')
OUT_PATH        = os.path.join(_HERE, '..', 'data', 'datasets_C', 'labels_C.csv')

N_ITER_DEFAULT  = 20   # one representative solution is enough for labelling
TEST_SPLIT      = 0.20
RANDOM_SEED     = 42


# =============================================================================
# SHARED TRAIN/TEST SPLIT  (also imported by run_experiments_C.py)
# =============================================================================

def split_all_instances(data_dir_low: str = DATA_DIR_LOW, data_dir_high: str = DATA_DIR_HIGH,
                        test_split: float = TEST_SPLIT, seed: int = RANDOM_SEED
                        ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    ONE 80/20 shuffle-split over the COMBINED low+high instance pools --
    same shuffle-then-slice pattern as run_experiments_A.py/_B.py's
    split_instances, just over both directories at once instead of one.

    Returns (train_items, test_items), each a list of (filepath, pool)
    tuples where pool is 'low' or 'high'. Both generate_labels_C.py and
    run_experiments_C.py call this with the same default seed, so training
    labels and evaluation instances never overlap.
    """
    items: list[tuple[str, str]] = []
    for pool, d in (('low', data_dir_low), ('high', data_dir_high)):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.txt'):
                items.append((os.path.join(d, f), pool))

    rng = random.Random(seed)
    rng.shuffle(items)
    n_test = max(1, int(len(items) * test_split))
    return items[n_test:], items[:n_test]   # train, test


# =============================================================================
# PER-INSTANCE LABEL EXTRACTION
# =============================================================================

def extract_labels(filepath: str, n_iter: int, source_pool: str,
                   verbose: bool = False,
                   bootstrap_bundle: dict | None = None,
                   bootstrap_alpha: float = ALPHA_DEFAULT,
                   bootstrap_beta: float = BETA_DEFAULT) -> list[dict]:
    """
    Solve one instance, extract one training row per active node:
    FEATURE_COLS_C values + 'overflowed' + metadata.

    bootstrap_bundle=None (default, "round 1"): solve with
    heuristic_dynamic.run_grasp_dynamic(bundle=None) -- distance+time, no ML,
    since no Option-C model exists yet.

    bootstrap_bundle=<loaded Option C model> ("round 2+"): solve with
    heuristic_c.run_grasp_c(bundle=bootstrap_bundle) instead -- training
    routes come from something closer to what will actually be deployed,
    reducing train/serve skew on arrival_hour (the model's own merge
    decisions shift real arrival times away from what a plain dynamic-only
    bootstrap produced).
    """
    params, nodes, dist_matrix, orders, locker_cap = read_full_instance(filepath)
    if not locker_cap:
        return []   # no capacity data -- nothing to label

    dep_h = params.departure
    if bootstrap_bundle is None:
        best_sol, _elapsed = run_grasp_dynamic(
            nodes, dist_matrix, params.capacity,
            n_iter=n_iter, departure_h=dep_h, bundle=None, verbose=False)
    else:
        best_sol, _elapsed = run_grasp_c(
            nodes, dist_matrix, params.capacity,
            bundle=bootstrap_bundle, locker_cap=locker_cap,
            n_iter=n_iter, alpha=bootstrap_alpha, beta=bootstrap_beta,
            departure_h=dep_h, verbose=False)
    if best_sol is None:
        return []

    arrivals = _compute_arrival_times(best_sol, dep_h, dist_matrix)
    _fb_km, _n_fb, overflow_locations = simulate_solution_with_overflow(
        best_sol, orders, locker_cap, dist_matrix)

    feats = extract_features_capacity(params, orders, locker_cap)

    rows = []
    for node_id, t_arrival_s in arrivals.items():
        if node_id not in feats:
            continue
        row = dict(feats[node_id])
        row['arrival_hour'] = (t_arrival_s / 3600.0) % 24.0
        row['overflowed']   = overflow_locations.get(node_id, 0)
        row['instance']     = params.name
        row['source_pool']  = source_pool
        rows.append(row)

    if verbose:
        n_over = sum(r['overflowed'] for r in rows)
        print(f"  {params.name:<30} [{source_pool:<4}]  nodes={len(rows):>4}  "
              f"overflowed={n_over:>3} ({100*n_over/len(rows) if rows else 0:.1f}%)")
    return rows


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description='Generate real, time-conditioned labels for Option C')
    p.add_argument('--max-train', type=int, default=None,
                   help='Cap on training instances used (default: all of the 80%% split -- '
                        'use this only for a quick smoke test)')
    p.add_argument('--n-iter', type=int, default=N_ITER_DEFAULT,
                   help='GRASP iterations per label-generation solve (reduced -- one '
                        'representative solution is enough, not a converged optimum)')
    p.add_argument('--data-low',  default=DATA_DIR_LOW)
    p.add_argument('--data-high', default=DATA_DIR_HIGH)
    p.add_argument('--test-split', type=float, default=TEST_SPLIT)
    p.add_argument('--seed', type=int, default=RANDOM_SEED)
    p.add_argument('--out', default=OUT_PATH)
    p.add_argument('--bootstrap-model', default=None,
                   help='Path to a trained Option C bundle. If given, training routes are '
                        'generated with heuristic_c.run_grasp_c(bundle=<this>) instead of '
                        'the plain dynamic heuristic -- a "round 2" bootstrap iteration '
                        'that reduces train/serve skew (see module docstring).')
    p.add_argument('--bootstrap-alpha', type=float, default=ALPHA_DEFAULT)
    p.add_argument('--bootstrap-beta',  type=float, default=BETA_DEFAULT)
    args = p.parse_args()

    train_items, test_items = split_all_instances(
        args.data_low, args.data_high, args.test_split, args.seed)
    if args.max_train:
        train_items = train_items[:args.max_train]

    bootstrap_bundle = load_model(args.bootstrap_model) if args.bootstrap_model else None

    print(f"\n{'='*70}")
    print(f"  GENERATE LABELS — Option C")
    print(f"  Train split: {len(train_items)} instances "
          f"({sum(1 for _, p in train_items if p=='low')} low + "
          f"{sum(1 for _, p in train_items if p=='high')} high)")
    print(f"  Test split (held out, NOT used here): {len(test_items)} instances")
    print(f"  n_iter (per solve) : {args.n_iter}")
    if bootstrap_bundle:
        print(f"  Bootstrap: heuristic_c (alpha={args.bootstrap_alpha}, "
              f"beta={args.bootstrap_beta}) using {args.bootstrap_model}")
    else:
        print(f"  Bootstrap: heuristic_dynamic (no ML) -- round 1")
    print(f"{'='*70}\n")

    all_rows: list[dict] = []
    t0 = time.perf_counter()
    for idx, (path, pool_name) in enumerate(train_items, 1):
        print(f"[{idx}/{len(train_items)}] ", end='')
        try:
            all_rows.extend(extract_labels(
                path, args.n_iter, pool_name, verbose=True,
                bootstrap_bundle=bootstrap_bundle,
                bootstrap_alpha=args.bootstrap_alpha,
                bootstrap_beta=args.bootstrap_beta))
        except Exception as exc:
            print(f"  ERROR on {path}: {exc}")
    elapsed = time.perf_counter() - t0

    if not all_rows:
        print("No rows generated -- check that data/instances/ and/or data/instances_B/ exist.")
        return

    n_over = sum(r['overflowed'] for r in all_rows)
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed:.1f}s: {len(all_rows)} rows, "
          f"{n_over} overflowed ({100*n_over/len(all_rows):.1f}%)")
    print(f"{'='*70}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = FEATURE_COLS_C + ['overflowed', 'instance', 'source_pool']
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Labels -> {args.out}")


if __name__ == '__main__':
    main()
