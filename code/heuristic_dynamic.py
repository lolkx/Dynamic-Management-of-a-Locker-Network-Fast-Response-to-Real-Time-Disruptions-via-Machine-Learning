"""
heuristic_dynamic.py
---------------------
Third heuristic: BR-CWS + GRASP with a genuinely time-of-day-aware savings
recalculation, optionally combined with the same ML saturation penalty used
by heuristic_learn.py.

Why the real-time part needs its own module instead of a parameter on
heuristic_learn.py:
Both heuristic.build_graph and heuristic_learn.build_graph_learn compute one
static, pre-sorted savings list BEFORE any route exists, and br_CWS/
br_CWS_learn only sample from that frozen list. A saving that reflects real
traffic must be evaluated against each route's ACTUAL, currently-accumulated
arrival time at its tail node -- unknowable before merging starts, and
different every GRASP iteration since merge order is randomised. That
requires recomputing the saving at merge-decision time, not just re-weighting
a precomputed number.

Design
------
1. Candidate arcs come from a static, once-built ranking (build_graph_dynamic
   below): distance-only (heuristic.build_graph) if no ML bundle is given, or
   distance-minus-ML-penalty (heuristic_learn.build_graph_learn, unchanged
   formula/function -- no duplication) if one is. This ranking also drives
   the same biased-random sampling order used by heuristic.br_CWS, so the
   sampling cost is unchanged either way.

   Why the ML penalty belongs in the STATIC ranking rather than being
   recomputed dynamically: P(sat_j) and d_fallback(j) are properties of node
   j and the instance alone (demand, dist_matrix, locker_cap) -- independent
   of which route currently owns j or the order arcs were merged in. Route
   state (tail_time) only ever appears on the i-side of a candidate arc, via
   the dynamic gate below, never on the j-side the ML model scores. So
   recomputing the ML term per-merge would add cost with zero benefit.

2. Each Route tracks `tail_time` (seconds from midnight, Model.Route field)
   -- the real, accumulated arrival time at its current tail node.
3. When a candidate arc (i -> j) is popped from the sampling loop, its
   savings value is RECALCULATED by fusing distance + live time + the ML
   penalty (all three signals the user asked for, not just a side gate):

       s_fused(i,j) = edge.s_raw + congestion_delta_m(i,j)

   where edge.s_raw is the static term from build_graph_dynamic (s_dist_m,
   minus the ML penalty term if a bundle was given -- see point 1 above),
   and congestion_delta_m is the LIVE correction computed at merge-decision
   time using iRoute.tail_time (see _congestion_delta_m below). The merge is
   accepted only if s_fused > 0.

   IMPORTANT -- avoiding double-counting distance: a first version of this
   simply added the full live time-saving (converted to a metres-equivalent
   via speed_profile.AVG_SPEED_MS) to edge.s_raw. That double-counts: if the
   live speed at iRoute.tail_time equals the reference departure-hour speed,
   the time-saving term collapses to s_dist_m * (AVG_SPEED_MS / speed_dep) --
   a rescaled COPY of the same distance term already inside edge.s_raw, not
   independent information (confirmed by expanding the algebra). The fix is
   congestion_delta_m: it isolates ONLY the excess effect caused by the live
   speed differing from the reference departure-hour speed, by subtracting
   the "everything at the reference speed" baseline from the raw time-saving
   formula. The j.dnEdge.cost term cancels out in that subtraction, leaving:

       congestion_delta_s(i,j) = (i.ndEdge.cost - edge.cost)
                                  * (1/speed_ms(iRoute.tail_time)
                                     - 1/speed_ms(departure_h*3600))

   which is exactly 0 when the live speed matches the reference speed (no
   phantom addition) and captures only genuine congestion differential
   otherwise. This is NOT the same mistake as the old, removed lambda_t
   blend (heuristic_learn.py's module docstring) -- that one was circular,
   estimating arrival time before any route existed; this one is safe
   because iRoute.tail_time only exists once construction has genuinely
   started, but it still needed this normalisation fix to avoid silently
   double-weighting distance.

   Why the accept/reject gate is kept (rather than dropped for strict
   behavioural parity with heuristic.py/heuristic_learn.py, which never
   reject a structurally-valid merge based on savings sign): dropping the
   gate without giving the recalculated score any other role would make
   this module behave IDENTICALLY to whichever static heuristic underlies
   it, since re-sorting the whole candidate list after every merge to let
   the score influence exploration order too would need an O(n^2*K)
   worst case with the current flat-list sampling structure -- not worth it
   for this codebase. The gate is this heuristic's actual differentiator:
   it deliberately declines merges that real, current traffic makes
   unprofitable, exactly as heuristic_learn.py's ML penalty already makes
   it choose differently from heuristic.py without that being "unfair" --
   both are intentionally different decision rules being compared, not a
   handicap.
4. On acceptance, the route's tail_time is advanced through the merged
   chain using the same accumulated-time walk as
   heuristic_learn._compute_arrival_times, but applied incrementally during
   construction (see _merge_routes_dynamic).

Net result: distance-CWS (heuristic.py) vs. ML-learnheuristic
(heuristic_learn.py) vs. this module in two modes -- pure distance+real-time
(bundle=None) or the "most powerful" combination of all three signals
(bundle=<loaded Model A/B bundle>).
"""

from __future__ import annotations
import math
import random
import time as _time
import numpy as np

from model import Route, Solution
from heuristic import (
    INF, P_BIAS, N_ITER,
    build_graph, _check_merging, _merge_routes,
    print_solution,
)
from heuristic_learn import build_graph_learn, load_model, BETA_DEFAULT
from speed_profile import speed_ms, AVG_SPEED_MS


# =============================================================================
# 0. CANDIDATE GRAPH  (distance-only, or distance-minus-ML-penalty)
# =============================================================================

def build_graph_dynamic(nodes, dist_matrix: np.ndarray, bundle: dict | None = None,
                         beta: float = BETA_DEFAULT, locker_cap: dict | None = None,
                         departure_h: float = 8.0, working_hours: float = 8.0):
    """
    Build the static candidate ranking consumed by br_CWS_dynamic.

    bundle=None  -> heuristic.build_graph (distance-only, today's default).
    bundle=<...> -> heuristic_learn.build_graph_learn (distance - ML
                    saturation penalty), same formula/function heuristic_learn
                    uses for the learnheuristic -- no duplication. See module
                    docstring for why the ML term belongs in this static
                    ranking rather than being recomputed per merge.
    """
    if bundle is None:
        return build_graph(nodes, dist_matrix)
    return build_graph_learn(nodes, dist_matrix, bundle, beta, departure_h,
                              working_hours, locker_cap)


# =============================================================================
# 1. DUMMY SOLUTION  (one route per active node, with tail_time initialised)
# =============================================================================

def _build_dummy_solution_dynamic(active_nodes, depot, departure_h: float) -> Solution:
    """Like heuristic._build_dummy_solution, plus per-route tail_time init."""
    sol = Solution()
    dep_t = departure_h * 3600.0
    for node in active_nodes:
        node.inRoute = None

    for node in active_nodes:
        route                 = Route()
        route.edges           = [node.dnEdge, node.ndEdge]
        route.cost            = node.dnEdge.cost + node.ndEdge.cost
        route.delivery_weight = node.delivery_weight
        route.pickup_weight   = node.pickup_weight
        route.service_time    = node.service_time
        route.nodes           = [node]
        route.tail_time       = dep_t + node.dnEdge.cost / speed_ms(dep_t)
        node.inRoute          = route
        sol.routes.append(route)
        sol.cost             += route.cost
    return sol


# =============================================================================
# 2. CONGESTION DELTA  (evaluated at merge-decision time, fused into savings)
# =============================================================================

def _congestion_delta_m(inode, jnode, iRoute, edge, dep_speed_ms: float) -> float:
    """
    Metres-equivalent correction to the static saving, from ONLY the excess
    effect of the live speed at iRoute's real accumulated tail_time differing
    from the reference departure-hour speed (see module docstring point 3 for
    the derivation -- this is NOT the raw time-saving, which would double
    count the distance term already in edge.s_raw):

        congestion_delta_s = (i.ndEdge.cost - edge.cost)
                              * (1/speed_ms(iRoute.tail_time) - 1/dep_speed_ms)

    Exactly 0 when the live speed equals the reference speed (no phantom
    addition); non-zero only when real congestion differs from the baseline.
    """
    speed_i = speed_ms(iRoute.tail_time)
    delta_s = (inode.ndEdge.cost - edge.cost) * (1.0 / speed_i - 1.0 / dep_speed_ms)
    return delta_s * AVG_SPEED_MS


# =============================================================================
# 3. MERGE ROUTES  (+ advance tail_time through the merged chain)
# =============================================================================

def _merge_routes_dynamic(inode, jnode, iRoute, jRoute, edge, sol: Solution) -> None:
    """Merge jRoute into iRoute (via heuristic._merge_routes), then advance
    iRoute.tail_time through the newly-appended chain using time-dependent
    speeds -- the same accumulated-time walk as
    heuristic_learn._compute_arrival_times, applied incrementally here."""
    t_arr_i = iRoute.tail_time

    # jRoute.nodes / jRoute.edges (post head-edge-pop) stay intact after the
    # merge -- _merge_routes only copies references into iRoute, it doesn't
    # clear jRoute's own lists.
    _merge_routes(inode, jnode, iRoute, jRoute, edge, sol)

    nodes_chain = jRoute.nodes          # [j, x2, ..., xn]
    edges_chain = jRoute.edges          # [j->x2, x2->x3, ..., xn->depot]

    t = t_arr_i + inode.service_time
    t = t + edge.cost / speed_ms(t)     # arrival at nodes_chain[0] (=jnode)

    for idx in range(1, len(nodes_chain)):
        prev_node = nodes_chain[idx - 1]
        leg_edge  = edges_chain[idx - 1]
        t = t + prev_node.service_time
        t = t + leg_edge.cost / speed_ms(t)

    iRoute.tail_time = t


# =============================================================================
# 4. BR-CWS DYNAMIC  (one GRASP iteration)
# =============================================================================

def br_CWS_dynamic(active_nodes, savings_list, vehicle_cap: float, depot,
                    departure_h: float = 8.0, p: float = P_BIAS) -> Solution:
    """
    One BR-CWS pass with fused, recalculated acceptance: candidates are
    sampled in the same distance(-ML)-ranked, biased-random order as
    heuristic.br_CWS/heuristic_learn.br_CWS_learn, but accepted only if the
    RECALCULATED fused saving (static edge.s_raw + live congestion_delta_m,
    see module docstring point 3) is still positive using each route's real
    accumulated tail_time.
    """
    sol         = _build_dummy_solution_dynamic(active_nodes, depot, departure_h)
    local_sav   = list(savings_list)
    log_p       = math.log(p)
    dep_speed_ms = speed_ms(departure_h * 3600.0)

    while local_sav:
        u   = random.random()
        idx = min(int(math.floor(math.log(max(u, 1e-300)) / log_p)),
                  len(local_sav) - 1)

        edge = local_sav[idx]
        del local_sav[idx]

        inode  = edge.origin
        jnode  = edge.end
        iRoute = inode.inRoute
        jRoute = jnode.inRoute

        if iRoute is None or jRoute is None:
            continue
        if not _check_merging(inode, jnode, iRoute, jRoute, vehicle_cap):
            continue

        delta_m = _congestion_delta_m(inode, jnode, iRoute, edge, dep_speed_ms)
        s_fused = edge.s_raw + delta_m
        if s_fused <= 0.0:
            continue   # not worth merging now, at this time of day

        _merge_routes_dynamic(inode, jnode, iRoute, jRoute, edge, sol)

    return sol


# =============================================================================
# 5. GRASP DYNAMIC
# =============================================================================

def run_grasp_dynamic(nodes, dist_matrix: np.ndarray, vehicle_cap: float,
                       n_iter: int = N_ITER, p_bias: float = P_BIAS,
                       departure_h: float = 8.0,
                       bundle: dict | None = None, beta: float = BETA_DEFAULT,
                       locker_cap: dict | None = None, working_hours: float = 8.0,
                       verbose: bool = True
                       ) -> tuple[Solution, float]:
    """
    GRASP loop: build the candidate graph once (build_graph_dynamic -- plain
    distance if bundle=None, distance-minus-ML-penalty otherwise), then
    repeat br_CWS_dynamic n_iter times with the real-time accept/reject gate.

    Parameters
    ----------
    bundle     : optional model bundle (heuristic_learn.load_model()). None
                 keeps this a pure distance+real-time heuristic; passing one
                 combines it with the ML saturation penalty (see module
                 docstring).
    beta       : ML saturation penalty weight [0, 1] (only used if bundle set)
    locker_cap : required to correctly score 'capacity' feature-set (Model B)
                 bundles; see heuristic_learn._saturation_probs.
    """
    t0 = _time.perf_counter()

    active_nodes, savings_list = build_graph_dynamic(
        nodes, dist_matrix, bundle, beta, locker_cap, departure_h, working_hours)
    depot     = nodes[0]
    best_sol  = None
    best_cost = INF

    for it in range(n_iter):
        sol = br_CWS_dynamic(active_nodes, savings_list, vehicle_cap, depot,
                              departure_h, p_bias)
        if sol.cost < best_cost:
            best_sol  = sol
            best_cost = sol.cost
            if verbose:
                print(f"    iter {it+1:>4}: new best -> "
                      f"{best_cost/1000:.3f} km | {len(sol.routes)} routes")

    return best_sol, _time.perf_counter() - t0


# =============================================================================
# QUICK SELF-TEST
# =============================================================================

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from instance_reader import read_full_instance

    if len(sys.argv) < 2:
        print("Usage: python heuristic_dynamic.py <instance.txt> [model.pkl] [beta] [departure_h]")
        print("       (omit model.pkl for the pure distance+real-time mode, no ML)")
        sys.exit(1)

    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    beta_val   = float(sys.argv[3]) if len(sys.argv) > 3 else BETA_DEFAULT
    params, nodes, dist_matrix, *_rest = read_full_instance(sys.argv[1])
    dep_h = float(sys.argv[4]) if len(sys.argv) > 4 else params.departure

    bundle = load_model(model_path) if model_path else None
    locker_cap = _rest[1] if len(_rest) > 1 else None

    print(f"Instance: {params.name}  "
          f"({params.n_orders} orders, {params.n_nodes} nodes, cap={params.capacity})")

    label = f"dynamic dep={dep_h}h" + (f" +ML beta={beta_val}" if bundle else "")
    best, elapsed = run_grasp_dynamic(nodes, dist_matrix, params.capacity,
                                      n_iter=100, departure_h=dep_h,
                                      bundle=bundle, beta=beta_val,
                                      locker_cap=locker_cap, verbose=True)
    print_solution(best, f"{params.name} [{label}]")
    print(f"\nSolved in {elapsed:.2f} s")
