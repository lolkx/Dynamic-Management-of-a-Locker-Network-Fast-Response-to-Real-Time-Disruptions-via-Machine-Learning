"""
compare_dynamic_priority.py
----------------------------
Standalone-vs-dynamic-priority comparison used by Results.tex's
subsec:comparison-dynamic (Tables tab:dynamic-result, tab:dynamic-grid).
Standard vs HSR-dynamic vs SRA-dynamic on a held-out sample (10 low + 10
high), all under common random numbers (same per-instance seed), plus a
full alpha x beta grid for HSR-dynamic on 5 high-saturation instances.

Reuses generate_labels_C.split_all_instances for the held-out test set, so
results are drawn from instances neither model was trained on.

Usage
-----
    python compare_dynamic_priority.py --n-iter 50
"""
from __future__ import annotations
import os, sys, csv, time, random, argparse
sys.path.insert(0, os.path.dirname(__file__))

from instance_reader import read_full_instance
from heuristic import run_grasp
from heuristic_c import run_grasp_c_dynamic, load_model as load_model_c
from heuristic_d import run_grasp_d_dynamic
from simulator import simulate_solution
from generate_labels_C import split_all_instances

_HERE    = os.path.dirname(__file__)
MODEL_C  = os.path.join(_HERE, '..', 'data', 'models_C', 'saturation_C_best.pkl')
MODEL_D  = os.path.join(_HERE, '..', 'data', 'models_D', 'saturation_D_best.pkl')
ALPHA    = 0.5
BETA     = 0.3


def _solve_all(path: str, n_iter: int, bundle_c, bundle_d, alpha: float, beta: float):
    params, nodes, dist_matrix, orders, locker_cap = read_full_instance(path)
    dep_h = params.departure
    seed = abs(hash(params.name)) % (2**31)

    sol_std, _ = run_grasp(nodes, dist_matrix, params.capacity, n_iter=n_iter,
                           verbose=False, rng=random.Random(seed))
    sol_hsr, _ = run_grasp_c_dynamic(nodes, dist_matrix, params.capacity,
                                     bundle=bundle_c, locker_cap=locker_cap,
                                     n_iter=n_iter, alpha=alpha, beta=beta,
                                     departure_h=dep_h, verbose=False, rng=random.Random(seed))
    sol_sra, _ = run_grasp_d_dynamic(nodes, dist_matrix, params.capacity,
                                     bundle=bundle_d, locker_cap=locker_cap,
                                     n_iter=n_iter, alpha=alpha, beta=beta,
                                     departure_h=dep_h, verbose=False, rng=random.Random(seed))

    out = {}
    for tag, sol in [('std', sol_std), ('hsr', sol_hsr), ('sra', sol_sra)]:
        fb_km, n_fb = simulate_solution(sol, orders, locker_cap, dist_matrix)
        out[tag] = dict(dist_km=sol.cost / 1000, fallback_km=fb_km,
                        effective_km=sol.cost / 1000 + fb_km, n_fallbacks=n_fb)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--n-iter', type=int, default=50)
    p.add_argument('--n-per-regime', type=int, default=10)
    p.add_argument('--n-grid-instances', type=int, default=5)
    args = p.parse_args()

    bundle_c = load_model_c(MODEL_C)
    bundle_d = load_model_c(MODEL_D)

    _train, test_items = split_all_instances()
    low = [f for f, pool in test_items if pool == 'low'][:args.n_per_regime]
    high = [f for f, pool in test_items if pool == 'high'][:args.n_per_regime]

    print(f"\n{'='*70}\n  STANDARD vs DYNAMIC-PRIORITY HSR/SRA -- {len(low)} low + {len(high)} high\n{'='*70}")

    results = {'low': [], 'high': []}
    for regime, files in [('low', low), ('high', high)]:
        for i, fname in enumerate(files, 1):
            print(f"[{regime} {i}/{len(files)}] {fname} ...", flush=True)
            r = _solve_all(fname, args.n_iter, bundle_c, bundle_d, ALPHA, BETA)
            results[regime].append(r)

    print(f"\n{'='*70}\n  TABLE: tab:dynamic-result\n{'='*70}")
    for regime in ('low', 'high'):
        rows = results[regime]
        avg_std = sum(r['std']['effective_km'] for r in rows) / len(rows)
        avg_hsr = sum(r['hsr']['effective_km'] for r in rows) / len(rows)
        avg_sra = sum(r['sra']['effective_km'] for r in rows) / len(rows)
        print(f"  {regime}: Std={avg_std:.3f}  HSR={avg_hsr:.3f} ({100*(avg_hsr/avg_std-1):+.1f}%)  "
              f"SRA={avg_sra:.3f} ({100*(avg_sra/avg_std-1):+.1f}%)")
        if regime == 'high':
            avg_fb_std = sum(r['std']['n_fallbacks'] for r in rows) / len(rows)
            avg_fb_hsr = sum(r['hsr']['n_fallbacks'] for r in rows) / len(rows)
            avg_fb_sra = sum(r['sra']['n_fallbacks'] for r in rows) / len(rows)
            print(f"    n_fallbacks: Std={avg_fb_std:.2f}  HSR={avg_fb_hsr:.2f} ({100*(avg_fb_hsr/avg_fb_std-1):+.1f}%)  "
                  f"SRA={avg_fb_sra:.2f} ({100*(avg_fb_sra/avg_fb_std-1):+.1f}%)")

    # --- alpha x beta grid on high-saturation instances ---
    grid_files = high[:args.n_grid_instances]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    betas = [0.0, 0.1, 0.3, 0.5]

    print(f"\n{'='*70}\n  TABLE: tab:dynamic-grid ({len(grid_files)} high-sat instances)\n{'='*70}")
    grid_out_rows = []
    std_cache = {}
    for fname in grid_files:
        params, nodes, dist_matrix, orders, locker_cap = read_full_instance(fname)
        seed = abs(hash(params.name)) % (2**31)
        sol_std, _ = run_grasp(nodes, dist_matrix, params.capacity, n_iter=args.n_iter,
                               verbose=False, rng=random.Random(seed))
        fb_std, nfb_std = simulate_solution(sol_std, orders, locker_cap, dist_matrix)
        std_cache[fname] = dict(effective_km=sol_std.cost / 1000 + fb_std, n_fallbacks=nfb_std)

    for beta in betas:
        deltas_km, deltas_fb = [], []
        for alpha in alphas:
            per_inst_km, per_inst_fb = [], []
            for fname in grid_files:
                params, nodes, dist_matrix, orders, locker_cap = read_full_instance(fname)
                seed = abs(hash(params.name)) % (2**31)
                sol_hsr, _ = run_grasp_c_dynamic(nodes, dist_matrix, params.capacity,
                                                 bundle=bundle_c, locker_cap=locker_cap,
                                                 n_iter=args.n_iter, alpha=alpha, beta=beta,
                                                 departure_h=params.departure, verbose=False,
                                                 rng=random.Random(seed))
                fb_hsr, nfb_hsr = simulate_solution(sol_hsr, orders, locker_cap, dist_matrix)
                eff_hsr = sol_hsr.cost / 1000 + fb_hsr
                delta_km = eff_hsr - std_cache[fname]['effective_km']
                delta_fb = nfb_hsr - std_cache[fname]['n_fallbacks']
                per_inst_km.append(delta_km)
                per_inst_fb.append(delta_fb)
                grid_out_rows.append(dict(beta=beta, alpha=alpha, instance=params.name,
                                          delta_km=delta_km, delta_fb=delta_fb))
            avg_km = sum(per_inst_km) / len(per_inst_km)
            avg_fb = sum(per_inst_fb) / len(per_inst_fb)
            deltas_km.append((alpha, avg_km))
            deltas_fb.append((alpha, avg_fb))
            print(f"  beta={beta} alpha={alpha}: avg delta-km={avg_km:+.3f}  avg delta-fb={avg_fb:+.2f}")
        best_alpha, best_km = min(deltas_km, key=lambda x: x[1])
        km_min = min(d[1] for d in deltas_km); km_max = max(d[1] for d in deltas_km)
        fb_min = min(d[1] for d in deltas_fb); fb_max = max(d[1] for d in deltas_fb)
        print(f"  -> beta={beta}: best alpha={best_alpha} (delta={best_km:+.3f}km), "
              f"km range [{km_min:+.3f}, {km_max:+.3f}], fb range [{fb_min:+.2f}, {fb_max:+.2f}]")

    out_csv = os.path.join(_HERE, '..', 'data', 'results_dynamic_priority_grid.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['beta', 'alpha', 'instance', 'delta_km', 'delta_fb'])
        w.writeheader()
        w.writerows(grid_out_rows)
    print(f"\nGrid CSV -> {out_csv}")


if __name__ == '__main__':
    main()
