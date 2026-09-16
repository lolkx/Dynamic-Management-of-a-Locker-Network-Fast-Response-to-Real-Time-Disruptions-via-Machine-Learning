"""
sensitivity_release_distribution.py
-------------------------------------
Automates the release-distribution robustness check raised in code review
(design notes): Option D is trained and evaluated with a SINGLE
assumed background self-collection release distribution
(Normal(mean_hour=14, std_hour=3), see simulator.py/generate_labels_D.py) --
an explicit assumption, not calibrated against real data. This script
generates labels, trains a model, and compares AUC/feature importances
across several candidate distributions, all on the SAME held-out split
(generate_labels_C.split_all_instances), so the sensitivity of Option D's
conclusions to that one assumption can be judged directly instead of taken
on faith.

This does NOT change the default Normal(14, 3) used elsewhere (generate_
labels_D.py, train_model_D.py, run_experiments_D.py) -- it is a standalone
diagnostic that generates its own labelled datasets per distribution under
data/datasets_D/sensitivity_release/ and reports a comparison table.

Usage
-----
    # Quick smoke test (small sample, fast)
    python sensitivity_release_distribution.py --max-train 40 --n-replicas 10

    # Full comparison (slow -- same cost as generate_labels_D.py x N distributions)
    python sensitivity_release_distribution.py

    # Custom distributions (mean,std pairs)
    python sensitivity_release_distribution.py --distributions "10,2;14,3;18,2"
"""

from __future__ import annotations
import os
import sys
import csv
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from instance_reader import FEATURE_COLS_D
from generate_labels_D import (extract_labels, N_ITER_DEFAULT, N_REPLICAS_DEFAULT,
                               START_H_DEFAULT, END_H_DEFAULT)
from generate_labels_C import (split_all_instances, DATA_DIR_LOW, DATA_DIR_HIGH,
                               TEST_SPLIT, RANDOM_SEED)
from ml_common import MODEL_FACTORIES, _cv_scores

_HERE    = os.path.dirname(__file__)
OUT_DIR  = os.path.join(_HERE, '..', 'data', 'datasets_D', 'sensitivity_release')

# (mean_hour, std_hour, label) -- default trio spans "early", "current default",
# and "late" plausible pickup-time centres, per the sensitivity analysis the
# user described (varying the assumed mean pickup time; see design notes).
DEFAULT_DISTRIBUTIONS = [
    (10.0, 2.0, 'early_10h'),
    (14.0, 3.0, 'default_14h'),
    (18.0, 2.0, 'late_18h'),
]


def _generate_for_distribution(train_items, mean_hour: float, std_hour: float,
                               n_iter: int, n_replicas: int, seed: int,
                               verbose: bool = True) -> list[dict]:
    """Same loop as generate_labels_D.main(), parameterised by distribution,
    reused here instead of duplicated."""
    all_rows: list[dict] = []
    for idx, (path, pool_name) in enumerate(train_items, 1):
        if verbose:
            print(f"  [{idx}/{len(train_items)}] ", end='', flush=True)
        base_seed = seed * 1_000_003 + idx
        try:
            rows = extract_labels(
                path, n_iter, n_replicas, pool_name, mean_hour, std_hour,
                START_H_DEFAULT, END_H_DEFAULT, base_seed, verbose=verbose)
            all_rows.extend(rows)
        except Exception as exc:
            if verbose:
                print(f"  ERROR on {path}: {exc}")
    return all_rows


def _train_quick(rows: list[dict], model_type: str = 'rf') -> dict:
    """Train ONE model (default: RF, fastest of the three -- this is a
    sensitivity check across several datasets, not the final model
    selection) and return its CV metrics + feature importances."""
    import numpy as np
    X = np.array([[float(r[c]) for c in FEATURE_COLS_D] for r in rows], dtype=float)
    y = np.array([int(r['delivered_ok']) for r in rows], dtype=int)

    m = _cv_scores(X, y, model_type, cv=5)
    clf = MODEL_FACTORIES[model_type]()
    clf.fit(X, y)
    importances = dict(zip(FEATURE_COLS_D, clf.feature_importances_)) \
        if hasattr(clf, 'feature_importances_') else {}
    return {'auc': m.get('roc_auc_mean', 0.0), 'f1': m.get('f1_mean', 0.0),
           'n_rows': len(rows), 'pos_rate': float(y.mean()),
           'importances': importances}


def main() -> None:
    p = argparse.ArgumentParser(
        description='Compare Option D across several assumed release distributions')
    p.add_argument('--distributions', default=None,
                   help='Semicolon-separated "mean,std" pairs, e.g. "10,2;14,3;18,2" '
                        '(default: early_10h / default_14h / late_18h)')
    p.add_argument('--max-train', type=int, default=None,
                   help='Cap on training instances (default: full 80%% split -- '
                        'use a small value for a quick smoke test)')
    p.add_argument('--n-iter', type=int, default=N_ITER_DEFAULT)
    p.add_argument('--n-replicas', type=int, default=N_REPLICAS_DEFAULT)
    p.add_argument('--data-low',  default=DATA_DIR_LOW)
    p.add_argument('--data-high', default=DATA_DIR_HIGH)
    p.add_argument('--test-split', type=float, default=TEST_SPLIT)
    p.add_argument('--seed', type=int, default=RANDOM_SEED)
    p.add_argument('--model', default='rf', choices=list(MODEL_FACTORIES))
    p.add_argument('--out-dir', default=OUT_DIR)
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    if args.distributions:
        dists = []
        for i, pair in enumerate(args.distributions.split(';')):
            mean_s, std_s = pair.split(',')
            dists.append((float(mean_s), float(std_s), f'mean{mean_s}_std{std_s}'))
    else:
        dists = DEFAULT_DISTRIBUTIONS

    train_items, _test_items = split_all_instances(
        args.data_low, args.data_high, args.test_split, args.seed)
    if args.max_train:
        train_items = train_items[:args.max_train]

    print(f"\n{'='*70}")
    print(f"  RELEASE-DISTRIBUTION SENSITIVITY  ({len(train_items)} train instances, "
          f"model={args.model.upper()})")
    print(f"  Distributions: {[(m, s, lbl) for m, s, lbl in dists]}")
    print(f"{'='*70}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for mean_hour, std_hour, label in dists:
        print(f"\n--- {label}: Normal(mean={mean_hour}h, std={std_hour}h) ---")
        t0 = time.perf_counter()
        rows = _generate_for_distribution(
            train_items, mean_hour, std_hour, args.n_iter, args.n_replicas,
            args.seed, verbose=not args.quiet)
        elapsed = time.perf_counter() - t0
        if not rows:
            print(f"  No rows generated for {label} -- skipping")
            continue

        csv_path = os.path.join(args.out_dir, f'labels_D_{label}.csv')
        fieldnames = FEATURE_COLS_D + ['delivered_ok', 'instance', 'source_pool', 'replica']
        with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

        metrics = _train_quick(rows, args.model)
        metrics['elapsed_s'] = round(elapsed, 1)
        results[label] = metrics
        print(f"  {label}: {metrics['n_rows']} rows, pos_rate={metrics['pos_rate']:.3f}, "
              f"AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}  ({elapsed:.1f}s)")

    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"  {'Distribution':<16}{'rows':>10}{'pos_rate':>10}{'AUC':>8}{'F1':>8}")
    for label, m in results.items():
        print(f"  {label:<16}{m['n_rows']:>10}{m['pos_rate']:>10.3f}"
              f"{m['auc']:>8.3f}{m['f1']:>8.3f}")

    print(f"\n  Feature importance for 'arrival_hour' / 'initial_occupancy_ratio' "
          f"(the two features the whole hypothesis depends on):")
    for label, m in results.items():
        imp = m['importances']
        print(f"  {label:<16}  arrival_hour={imp.get('arrival_hour', 0):.4f}  "
              f"initial_occupancy_ratio={imp.get('initial_occupancy_ratio', 0):.4f}")
    print('='*70)


if __name__ == '__main__':
    main()
