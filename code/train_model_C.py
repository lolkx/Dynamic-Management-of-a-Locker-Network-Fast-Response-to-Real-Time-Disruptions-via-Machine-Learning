"""
train_model_C.py  — OPTION C
-----------------------------
Train the time-conditioned saturation model for heuristic_c.py's unified
learnheuristic: P(locker j overflows | arrival hour h).

Unlike Option D's release-based label (see train_model_D.py) -- which comes
from a stochastic pickup-release simulation -- this label comes from REAL
simulated routes: generate_labels_C.py solves instances with
heuristic_dynamic.run_grasp_dynamic (no ML), replays the real per-node
arrival hour, and records real per-locker overflow via
simulator.simulate_solution_with_overflow (route-order-dependent, cascading
compartment state). See generate_labels_C.py's module docstring for the
full rationale.

This script does NOT run any GRASP itself -- it just loads the CSV that
generate_labels_C.py already produced (label generation is comparatively
expensive; training should be fast and repeatable without re-solving
anything).

Usage
-----
    python generate_labels_C.py            # once, produces data/datasets_C/labels_C.csv
    python train_model_C.py --compare       # train all, pick best
    python train_model_C.py --model rf
"""

from __future__ import annotations
import os, sys, csv, pickle, argparse, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from instance_reader import FEATURE_COLS_C
from ml_common import MODEL_FACTORIES, _cv_scores

# =============================================================================
# PATHS
# =============================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets_C', 'labels_C.csv')
OUT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'models_C')
CV_FOLDS  = 5


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset_C(csv_path: str = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load labels_C.csv (produced by generate_labels_C.py) into (X, y)."""
    if not os.path.exists(csv_path):
        raise RuntimeError(
            f"No dataset at {csv_path}.\n"
            "Generate it first: python generate_labels_C.py"
        )
    X_rows, y_rows = [], []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            X_rows.append([float(row[c]) for c in FEATURE_COLS_C])
            y_rows.append(int(float(row['overflowed'])))
    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    if len(X) == 0:
        raise RuntimeError(f"Dataset at {csv_path} is empty.")
    if y.sum() == 0:
        raise RuntimeError(
            f"All {len(X)} samples have label 0 (no overflow) in {csv_path}.\n"
            "Increase --n-high / re-check data/instances_B/ occupancy when generating labels."
        )
    return X, y


# =============================================================================
# TRAIN + COMPARE  (CV/factory machinery reused from train_model_B.py)
# =============================================================================

def train_and_save_C(csv_path: str, out_path: str, model_type: str = 'rf',
                     cv_folds: int = CV_FOLDS, verbose: bool = True) -> dict:
    print(f"\n  [C/{model_type.upper()}] Loading data (hourly overflow label)...")
    X, y = load_dataset_C(csv_path)
    n_pos = int(y.sum())
    print(f"  Dataset: {len(X)} samples | {n_pos} overflowed ({100*n_pos/len(X):.1f}%)")

    print(f"  Running {cv_folds}-fold CV...")
    m = _cv_scores(X, y, model_type, cv_folds)
    print(f"    Acc {m.get('accuracy_mean',0):.3f}  "
          f"F1 {m.get('f1_mean',0):.3f}  "
          f"AUC {m.get('roc_auc_mean',0):.3f}")

    clf = MODEL_FACTORIES[model_type]()
    clf.fit(X, y)

    if model_type in ('rf', 'gbm'):
        imps = clf.feature_importances_
        print("  Feature importances:")
        for feat, imp in sorted(zip(FEATURE_COLS_C, imps), key=lambda x: -x[1]):
            print(f"    {feat:<28} {imp:.4f}  {'#'*int(imp*30)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bundle = dict(model=clf, model_type=model_type, features=FEATURE_COLS_C,
                  feature_set='hourly',
                  n_train=len(X), metrics=m, label_strategy='hourly_overflow')
    with open(out_path, 'wb') as fh:
        pickle.dump(bundle, fh)
    print(f"    Saved -> {out_path}")
    return m


def compare_models_C(csv_path: str = DATA_PATH, out_dir: str = OUT_DIR,
                     cv_folds: int = CV_FOLDS, verbose: bool = True) -> str | None:
    print(f"\n{'='*60}")
    print(f"  OPTION C — MODEL COMPARISON (hourly overflow label)")
    print(f"  Data: {csv_path}")
    print(f"{'='*60}")

    all_m, failed = {}, {}
    for mtype in MODEL_FACTORIES:
        out_path = os.path.join(out_dir, f'saturation_C_{mtype}.pkl')
        try:
            all_m[mtype] = train_and_save_C(csv_path, out_path, mtype, cv_folds, verbose)
        except Exception as exc:
            failed[mtype] = str(exc)
            print(f"  ERROR [{mtype}]: {exc}")

    print(f"\n{'='*60}")
    print(f"  {'Model':<8}  {'F1':>10}  {'ROC-AUC':>12}")
    print(f"  {'-'*34}")
    best_type, best_auc = None, -1.0
    for mtype in MODEL_FACTORIES:
        if mtype in failed:
            print(f"  {mtype.upper():<8}  ERROR: {failed[mtype]}")
            continue
        m = all_m[mtype]
        auc = m.get('roc_auc_mean', 0)
        if auc > best_auc:
            best_auc, best_type = auc, mtype
        print(f"  {mtype.upper():<8}  "
              f"{m.get('f1_mean',0):>6.3f}±{m.get('f1_std',0):.3f}  "
              f"{auc:>6.3f}±{m.get('roc_auc_std',0):.3f}")

    if best_type:
        src = os.path.join(out_dir, f'saturation_C_{best_type}.pkl')
        dst = os.path.join(out_dir, 'saturation_C_best.pkl')
        shutil.copy2(src, dst)
        print(f"\n  Best: {best_type.upper()} (AUC={best_auc:.3f}) -> saturation_C_best.pkl")
    print('='*60)
    return best_type


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Option C: train the hourly-overflow model from generate_labels_C.py\'s dataset'
    )
    p.add_argument('--model',   default=None, choices=list(MODEL_FACTORIES))
    p.add_argument('--data',    default=DATA_PATH)
    p.add_argument('--out',     default=None)
    p.add_argument('--cv',      type=int, default=CV_FOLDS)
    p.add_argument('--compare', action='store_true')
    args = p.parse_args()

    if args.compare or args.model is None:
        compare_models_C(args.data, OUT_DIR, args.cv)
    else:
        out = args.out or os.path.join(OUT_DIR, f'saturation_C_{args.model}.pkl')
        train_and_save_C(args.data, out, args.model, args.cv)
