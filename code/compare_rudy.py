"""
compare_rudy.py
----------------
Compares our dynamic (distance+time, no ML) heuristic and HSR (dynamic-
priority) against each other on the exact real Rudy (2025) benchmark
instances, using the same three quality measures the reference paper uses:
total distance (Sigma D), maximum delivery time (Tmax, deliveries only),
and the Hyper-Volume Indicator (HVI) between Pareto front approximations.

Reference/baseline choice: the reference front is built from
heuristic_dynamic.run_grasp_dynamic (distance+time-aware, congestion-adjusted,
no ML), NOT the plain Standard heuristic (pure static distance). Rudy's own
problem formulation treats travel time as a first-class criterion (Tmax is
one of its two objectives; travel speed varies by hour via its own
speed-by-hour profile) -- so a heuristic that is already time-aware is the
fair, comparable baseline for that paper's metric space. Comparing HSR
(distance+time+ML) directly against pure-distance Standard would conflate
two effects at once -- adding time-awareness AND adding the ML penalty --
into a single number; holding time-awareness constant (dynamic vs. HSR) and
only toggling the ML term isolates the ML's actual marginal contribution.

IMPORTANT caveat, and why this does NOT attempt to reproduce Rudy's own
published numbers directly: HVI is an explicitly RELATIVE measure -- the
paper itself states "absolute HVI values cannot be compared between
various instances, and relative HVI values cannot be compared between
various experiments" (Section 8.4). Rudy's own tables report Sigma D/Tmax
normalised per-instance against the worst value among THEIR OWN compared
representations, not absolute kilometres/hours -- so there is no valid way
to place our absolute numbers next to their published relative ones.

What IS valid, and what this script does: compute HVI, Sigma D and Tmax
using the EXACT SAME definitions and the SAME benchmark instances Rudy's
paper introduces, comparing our own methods against each other. This
mirrors Rudy's own greedy-vs-GA comparison methodology (single-solution
front vs multi-solution front) with our dynamic heuristic (a single
distance+time solution -- a single-point "front") against HSR's
distance/time-blend front, obtained by sweeping alpha in {0, 0.25, 0.5,
0.75, 1.0} at beta=0.3 under the dynamic-priority mechanism (heuristic_c.
run_grasp_c_dynamic) -- the only mechanism where alpha actually changes
the constructed route (see design notes).

HVI definition (matching the paper): given two point sets (fronts) in
(Sigma D, Tmax) space, take the nadir point Z = 1.2 * (worst Sigma D,
worst Tmax) among ALL points from both fronts. HVI(F) is the area
dominated by F's non-dominated subset, bounded by Z.

Usage
------
    python compare_rudy.py --per-bucket 2 --n-iter 30
"""

from __future__ import annotations
import os, sys, csv, time, random, argparse
sys.path.insert(0, os.path.dirname(__file__))

from instance_reader import read_full_instance
from heuristic import N_ITER, P_BIAS
from heuristic_dynamic import run_grasp_dynamic
from heuristic_c import run_grasp_c_dynamic, build_graph_c, br_CWS_c_dynamic, _group_out_edges
from heuristic_learn import load_model, _compute_arrival_times

_HERE      = os.path.dirname(__file__)
DATA_DIR   = os.path.join(_HERE, '..', 'data', 'instances')
MODEL_C    = os.path.join(_HERE, '..', 'data', 'models_C', 'saturation_C_best.pkl')
OUT_CSV    = os.path.join(_HERE, '..', 'data', 'results_rudy_comparison.csv')

BUCKETS  = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
ALPHAS   = [0.0, 0.25, 0.5, 0.75, 1.0]
BETA     = 0.3


def _makespan_h(sol, departure_h: float, dist_matrix) -> float:
    """Tmax: completion time (h) of the LAST delivery (deliveries only) --
    identical definition to run_experiments_C._makespan_h, duplicated here
    to keep this script standalone."""
    arrivals = _compute_arrival_times(sol, departure_h, dist_matrix)
    max_t = departure_h * 3600.0
    for route in sol.routes:
        for node in route.nodes:
            if node.n_deliveries > 0 and node.Id in arrivals:
                completion = arrivals[node.Id] + node.service_time
                if completion > max_t:
                    max_t = completion
    return max_t / 3600.0


def _pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated subset for 2D minimisation."""
    front = []
    for p in points:
        if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in points):
            front.append(p)
    # de-duplicate
    return sorted(set(front))


def _hypervolume_2d(front: list[tuple[float, float]], z: tuple[float, float]) -> float:
    """Standard 2D hypervolume (minimisation) of a non-dominated front,
    bounded by nadir point z (must weakly dominate every front point)."""
    if not front:
        return 0.0
    pts = sorted(front)  # ascending by first coordinate
    hv = 0.0
    prev_y = z[1]
    for x, y in pts:
        if y < prev_y:
            hv += (z[0] - x) * (prev_y - y)
            prev_y = y
    return hv


def _select_instances(per_bucket: int) -> list[tuple[str, tuple[int, int]]]:
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.txt'))
    bucketed: dict[tuple[int, int], list[tuple[str, int]]] = {b: [] for b in BUCKETS}
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        params, *_ = read_full_instance(path)
        for lo, hi in BUCKETS:
            if lo < params.n_orders <= hi:
                bucketed[(lo, hi)].append((fname, params.n_orders))
                break
    selected = []
    for b, items in bucketed.items():
        items.sort(key=lambda x: x[1])
        # spread across the bucket: smallest, and one further in (median-ish)
        picks = [items[0]] if items else []
        if len(items) > 1:
            picks.append(items[len(items) // 2])
        for fname, n in picks[:per_bucket]:
            selected.append((fname, b))
    return selected


def run_instance(fname: str, bucket: tuple[int, int], n_iter: int, bundle_c) -> list[dict]:
    path = os.path.join(DATA_DIR, fname)
    params, nodes, dist_matrix, orders, locker_cap = read_full_instance(path)
    dep_h = params.departure
    seed = abs(hash(params.name)) % (2**31)

    rows = []

    # --- Reference: dynamic (distance+time, no ML) -- single point.
    # NOT the pure-distance Standard heuristic: Rudy's own formulation treats
    # travel time as a first-class criterion, so the fair baseline here is
    # already time-aware (see module docstring).
    sol_std, t_std = run_grasp_dynamic(nodes, dist_matrix, params.capacity, n_iter=n_iter,
                               p_bias=P_BIAS, departure_h=dep_h, verbose=False,
                               rng=random.Random(seed))
    d_std = sol_std.cost / 1000
    t_std_h = _makespan_h(sol_std, dep_h, dist_matrix)
    rows.append(dict(instance=params.name, bucket=str(bucket), method='dynamic', alpha=None,
                     dist_km=round(d_std, 3), tmax_h=round(t_std_h, 3),
                     elapsed_s=round(t_std, 2)))

    # --- HSR dynamic-priority: build once, sweep alpha (loop-level param) ---
    active_nodes, savings_list, hourly_probs, d_fb = build_graph_c(
        nodes, dist_matrix, bundle_c, locker_cap)
    out_edges = _group_out_edges(savings_list)
    depot = nodes[0]

    for alpha in ALPHAS:
        t0 = time.perf_counter()
        rng = random.Random(seed)
        best_sol, best_cost = None, float('inf')
        for it in range(n_iter):
            sol = br_CWS_c_dynamic(active_nodes, savings_list, params.capacity, depot,
                                   hourly_probs, d_fb, alpha, BETA, dep_h, P_BIAS, rng,
                                   out_edges=out_edges)
            if sol.cost < best_cost:
                best_sol, best_cost = sol, sol.cost
        elapsed = time.perf_counter() - t0
        d_hsr = best_sol.cost / 1000
        t_hsr_h = _makespan_h(best_sol, dep_h, dist_matrix)
        rows.append(dict(instance=params.name, bucket=str(bucket), method='hsr_dynamic',
                         alpha=alpha, dist_km=round(d_hsr, 3), tmax_h=round(t_hsr_h, 3),
                         elapsed_s=round(elapsed, 2)))

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description='Compare the dynamic (dist+time) heuristic vs HSR-dynamic on real Rudy instances')
    p.add_argument('--per-bucket', type=int, default=2)
    p.add_argument('--n-iter', type=int, default=30)
    p.add_argument('--out', default=OUT_CSV)
    args = p.parse_args()

    bundle_c = load_model(MODEL_C)
    instances = _select_instances(args.per_bucket)
    print(f"\n{'='*70}")
    print(f"  RUDY (2025) COMPARISON — {len(instances)} instances, n_iter={args.n_iter}")
    print(f"{'='*70}")
    for fname, b in instances:
        print(f"  {b}: {fname}")
    print()

    all_rows: list[dict] = []
    t0 = time.perf_counter()
    for i, (fname, b) in enumerate(instances, 1):
        print(f"[{i}/{len(instances)}] {fname} ...", flush=True)
        try:
            rows = run_instance(fname, b, args.n_iter, bundle_c)
            all_rows.extend(rows)
            std_row = next(r for r in rows if r['method'] == 'dynamic')
            hsr_rows = [r for r in rows if r['method'] == 'hsr_dynamic']
            print(f"    std: dist={std_row['dist_km']}km tmax={std_row['tmax_h']}h")
            for r in hsr_rows:
                print(f"    hsr alpha={r['alpha']}: dist={r['dist_km']}km tmax={r['tmax_h']}h")
        except Exception as exc:
            print(f"    ERROR: {exc}")
    print(f"\nTotal: {time.perf_counter()-t0:.1f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['instance', 'bucket', 'method', 'alpha',
                                                'dist_km', 'tmax_h', 'elapsed_s'])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Results -> {args.out}")

    # --- HVI per instance + aggregate by bucket ---
    print(f"\n{'='*70}")
    print("  HVI PER INSTANCE (Dynamic dist+time = single-point front, HSR = alpha-swept front)")
    print(f"{'='*70}")
    by_instance: dict[str, list[dict]] = {}
    for r in all_rows:
        by_instance.setdefault(r['instance'], []).append(r)

    bucket_hvi: dict[str, list[tuple]] = {}
    for inst, rows in by_instance.items():
        std_pt = next((r['dist_km'], r['tmax_h']) for r in rows if r['method'] == 'dynamic')
        hsr_pts = [(r['dist_km'], r['tmax_h']) for r in rows if r['method'] == 'hsr_dynamic']
        all_pts = [std_pt] + hsr_pts
        z = (1.2 * max(p[0] for p in all_pts), 1.2 * max(p[1] for p in all_pts))
        hvi_std = _hypervolume_2d(_pareto_front([std_pt]), z)
        hvi_hsr = _hypervolume_2d(_pareto_front(hsr_pts), z)
        b = rows[0]['bucket']
        bucket_hvi.setdefault(b, []).append((inst, hvi_std, hvi_hsr,
                                             std_pt[0], std_pt[1]))
        print(f"  {inst:<20} std_dist={std_pt[0]:.1f}km std_tmax={std_pt[1]:.2f}h  "
              f"HVI(std)={hvi_std:.2f}  HVI(hsr-front)={hvi_hsr:.2f}  "
              f"HSR/std ratio={hvi_hsr/hvi_std if hvi_std>0 else float('nan'):.3f}")

    print(f"\n{'='*70}")
    print("  AGGREGATE BY BUCKET")
    print(f"{'='*70}")
    for b, items in bucket_hvi.items():
        avg_std = sum(i[1] for i in items) / len(items)
        avg_hsr = sum(i[2] for i in items) / len(items)
        print(f"  {b:<16}  n={len(items)}  avg HVI(std)={avg_std:.2f}  "
              f"avg HVI(hsr-front)={avg_hsr:.2f}  ratio={avg_hsr/avg_std if avg_std>0 else float('nan'):.3f}")


if __name__ == '__main__':
    main()
