"""
heuristic_c.py
---------------
"Option C": a single unified time-aware learnheuristic, kept alongside
heuristic_learn.py (Model A/B) and heuristic_dynamic.py rather than
replacing them.

Savings formula, recalculated live at merge-decision time for every
candidate arc (i -> j):

    s(i,j) = alpha * s_dist(i,j) + (1-alpha) * s_time(i,j)
             - beta * P_j(j, arrival_hour) * d_fallback(j)

Where:
    s_dist(i,j)    -- classic Clarke-Wright distance saving (static, metres)
    s_time(i,j)    -- the SAME saving in time units, converted to a
                      metres-equivalent, computed LIVE using the route's
                      real accumulated arrival time (iRoute.tail_time) --
                      exactly the mechanism heuristic_dynamic.py already
                      uses, NOT the old, circular, pre-route lambda_t blend
                      (heuristic_learn.py's module docstring explains why
                      that one was removed). This is safe to blend with
                      s_dist because it uses genuinely live state.
    P_j(j, hour)   -- probability that locker j overflows GIVEN it is
                      visited at that hour of day, from a model trained on
                      REAL simulated (arrival_hour, overflowed) labels (see
                      generate_labels_C.py) -- not a time-independent
                      per-node probability like Model A/B.
    d_fallback(j)  -- distance to j's nearest alternative locker (reused
                      from heuristic_learn._fallback_distances, unchanged).

Blending s_dist and s_time via alpha (a convex combination, alpha+(1-alpha)
=1) is a standard multi-criteria scalarisation of two normalised, comparable
-unit signals -- it does NOT have the double-counting pathology found and
fixed in heuristic_dynamic.py's ADDITIVE fusion (that was about stacking two
correlated absolute quantities on top of each other; a weighted average of
two bounded scores doesn't compound the same way).

Why P_j needs a PRECOMPUTED per-hour lookup table, not live inference:
a fresh sklearn predict_proba() call per candidate arc (thousands per GRASP
iteration x up to 100 iterations) would be far too slow -- the same
performance concern that keeps heuristic_learn._saturation_probs to exactly
ONE batched call per graph build, not per merge. Fix: _hourly_saturation_
probs computes P(sat_j | hour) for ALL 24 hour buckets x all active nodes in
ONE batched predict_proba call at build time, then br_CWS_c does an O(1)
array lookup at merge-decision time using the estimated arrival hour
(iRoute.tail_time + travel time i->j, using the KNOWN speed at i -- the same
single-step estimate _congestion_delta_m/_merge_routes_dynamic already use
elsewhere, no fixed-point iteration needed since speed-at-i is already known
before j is reached).

Why the accept/reject gate (s_fused <= 0 -> reject) is kept: same
established rationale as heuristic_dynamic.py -- without it (or without an
expensive full re-sort of the candidate list after every merge), the
recalculated score wouldn't influence anything, and this heuristic would
collapse to the plain distance-based K-NN ranking it's built on top of.

bundle=None (no trained Option-C model yet) makes P_j identically 0
everywhere -- this lets alpha/beta ablation runs (distance-only, or
distance+time with no penalty) work for the sensitivity study in
run_sensitivity.py's sweep_alpha_c.

------------------------------------------------------------------------------
DYNAMIC-PRIORITY VARIANT (br_CWS_c_dynamic / run_grasp_c_dynamic)
------------------------------------------------------------------------------
Why it exists: the accept/reject gate in br_CWS_c (s_fused <= 0 -> reject) was
found to be empirically INERT at the thesis's evaluation settings
(alpha=0.5, beta=0.3). Candidates in br_CWS_c are drawn from a STATIC,
distance-only-sorted list via biased-random geometric sampling, and s_fused is
only consulted AFTER a candidate has already been picked (a decision made 100%
by distance) as a binary veto. Instrumentation confirmed that veto never fires
in practice: HSR/SRA produced bit-for-bit identical solutions to the plain
distance-only baseline across the full 50-instance test set. Time and ML never
influenced WHICH candidate was tried, and never even managed to veto one.

The dynamic variant fixes this by letting time + ML drive PRIORITISATION, not
just a post-hoc veto. The savings ranking becomes genuinely dynamic
(re-scored as construction proceeds), while reusing _fused_saving_c,
_check_merging and _merge_routes_dynamic UNCHANGED. It is ADDITIVE: br_CWS_c /
run_grasp_c are untouched so all already-reported thesis results stay
bit-for-bit reproducible.

Key insight making it cheap: an arc i -> j can only ever merge if i is the
CURRENT TAIL of its route and j is the CURRENT HEAD of a different route (the
asymmetric conditions in _check_merging). So at any instant only ONE outgoing
edge per active ROUTE (from its tail node) can be that route's next accepted
merge -- the "live" candidate pool has size R (active routes), not n*K.

Mechanism:
  * Each route caches best_candidate = (score, edge): the max-scoring outgoing
    edge from its CURRENT TAIL node among that node's precomputed K-NN edges
    (the same static K-NN graph build_graph_c already produces -- only the
    SELECTION order changes, not the candidate edges), scored by
    _fused_saving_c. Head-validity is NOT checked here (cheap, and the score
    doesn't depend on it); a per-route excluded_targets set skips targets
    already found infeasible for this tail, so the route falls through to its
    next-best K-NN candidate instead of retrying forever.
  * Per inner step: sample an index with the SAME geometric formula used
    everywhere else in this codebase, then select that-ranked route's candidate
    from the top of a max-heap of per-route (score, route) entries -- so
    biased-randomisation diversity across GRASP seeds is preserved (a pure
    max-pop would collapse GRASP to deterministic greedy). The heap is
    lazily invalidated: when a route's best_candidate is recomputed a per-route
    token is bumped and a fresh entry pushed, so older entries pop as stale and
    are discarded; a merged-away route's best_candidate is nulled so its
    entries pop stale too. A first version instead rebuilt the whole live pool
    and heapq.nlargest'd it every step; that is O(R) per step and the pool is
    dominated by RETRY steps (a popular target is the top K-NN candidate of
    many routes but only one can claim it as head -- ~9,500 retries vs ~850
    merges on a 949-node instance), which made it ~19x slower than br_CWS_c.
    The heap only touches the few entries the geometric index needs.
  * Feasibility is the single unchanged _check_merging call. Infeasible ->
    exclude that target for this route and recompute its best_candidate.
    Feasible -> _merge_routes_dynamic, then the merged route's NEW TAIL is
    jRoute's original tail, so recompute best_candidate for the new tail with
    a fresh excluded_targets set; jRoute leaves the pool. Stop when no route
    has a viable candidate left (same "construction can't extend" termination
    as br_CWS_c's empty-list stop).

Complexity: O(n*K*log(n*K)) time, dominated by heap maintenance. The initial
per-route best-candidate scan is O(n*K); each of the ~n merges and each
retry (bounded by K exclusions per route, so O(n*K) retries) triggers one
O(K) rescan plus O(log(n*K)) heap ops, and each step pops only the few
top-ranked entries the geometric index actually needs, not the whole pool.
"""

from __future__ import annotations
import heapq
import itertools
import math
import random
import time as _time
import numpy as np

from model import Solution
from heuristic import (
    INF, P_BIAS, N_ITER,
    build_graph, _check_merging, _merge_routes,
    print_solution,
)
from heuristic_learn import _fallback_distances, load_model, BETA_DEFAULT
from heuristic_dynamic import _build_dummy_solution_dynamic, _merge_routes_dynamic
from speed_profile import speed_ms, AVG_SPEED_MS

ALPHA_DEFAULT = 0.5


# =============================================================================
# 1. HOURLY SATURATION PROBABILITIES  (precomputed once per build, not per merge)
# =============================================================================

def _hourly_saturation_probs(nodes, bundle: dict | None,
                             locker_cap: dict | None = None) -> dict[int, np.ndarray]:
    """
    P(sat_j | hour) for hour=0..23, for every active node, via ONE batched
    predict_proba call. bundle=None -> all-zero tables (no ML penalty).

    Features mirror heuristic_learn._saturation_probs' 'capacity' branch
    (reconstructed from Node aggregates, since routing time has no raw
    orders list) plus a varying 14th 'arrival_hour' column -- the schema is
    instance_reader.FEATURE_COLS_C.
    """
    active = [n for n in nodes[1:] if n.is_active]
    if not active:
        return {}
    if bundle is None:
        return {n.Id: np.zeros(24) for n in active}

    total_del_w = sum(n.delivery_weight for n in active) or 1.0
    mean_del_w  = total_del_w / len(active)

    rows: list[list[float]] = []
    ids: list[int] = []
    for node in active:
        total_ord  = node.n_deliveries + node.n_pickups
        total_comp = (float(sum(locker_cap.get(node.Id, {}).values()))
                     if locker_cap is not None else 0.0)
        base = [
            float(node.n_deliveries),
            float(node.n_pickups),
            float(total_ord),
            node.delivery_weight,
            node.pickup_weight,
            node.n_deliveries / total_ord if total_ord > 0 else 0.0,
            node.delivery_weight / node.n_deliveries if node.n_deliveries > 0 else 0.0,
            node.delivery_weight / mean_del_w,
            node.delivery_weight / total_del_w,
            total_comp,
            node.n_deliveries / total_comp if total_comp > 0 else 0.0,
            (node.n_large_deliveries / node.n_deliveries
             if node.n_deliveries > 0 else 0.0),
            node.size_weighted_demand,
        ]
        for h in range(24):
            rows.append(base + [float(h)])
        ids.append(node.Id)

    X   = np.array(rows, dtype=float)
    clf = bundle['model']
    probs = (clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba')
             else clf.predict(X).astype(float))
    probs = probs.reshape(len(ids), 24)
    return {node_id: probs[i] for i, node_id in enumerate(ids)}


# =============================================================================
# 2. CANDIDATE GRAPH  (distance-only K-NN pruning, reused from heuristic.py)
# =============================================================================

def build_graph_c(nodes, dist_matrix: np.ndarray, bundle: dict | None = None,
                  locker_cap: dict | None = None):
    """
    Candidate arcs come from heuristic.build_graph, UNCHANGED -- selection is
    purely geometric (K-nearest by distance), unrelated to alpha/beta/time;
    see module docstring for why P_j is precomputed here rather than baked
    into a static ranking.
    """
    active_nodes, savings_list = build_graph(nodes, dist_matrix)
    hourly_probs = _hourly_saturation_probs(nodes, bundle, locker_cap)
    d_fb         = _fallback_distances(active_nodes, dist_matrix)
    return active_nodes, savings_list, hourly_probs, d_fb


# =============================================================================
# 3. FUSED SAVING  (recalculated live at merge-decision time)
# =============================================================================

def _fused_saving_c(inode, jnode, iRoute, edge,
                    hourly_probs: dict[int, np.ndarray], d_fb: dict[int, float],
                    alpha: float, beta: float, dep_speed_ms: float) -> float:
    """
    s(i,j) = alpha*s_dist(i,j) + (1-alpha)*s_time(i,j) - beta*P_j(j,hour)*d_fallback(j)

    s_dist is the static Clarke-Wright saving; s_time is the same saving in
    time units (converted to metres-equivalent), evaluated LIVE using
    iRoute's real accumulated tail_time -- see module docstring.
    """
    speed_i   = speed_ms(iRoute.tail_time)
    t_i_depot = inode.ndEdge.cost / speed_i
    t_depot_j = jnode.dnEdge.cost / dep_speed_ms
    t_i_j     = edge.cost / speed_i
    s_time_m  = (t_i_depot + t_depot_j - t_i_j) * AVG_SPEED_MS

    s_dist_m = inode.ndEdge.cost + jnode.dnEdge.cost - edge.cost

    # Estimated arrival hour at j: depart i (after service) at the speed
    # known at i's current tail_time -- single-step estimate, same pattern
    # as heuristic_dynamic._merge_routes_dynamic uses for the same purpose.
    t_est_j     = iRoute.tail_time + inode.service_time + edge.cost / speed_i
    hour_bucket = int(t_est_j / 3600.0) % 24
    p_j = hourly_probs.get(jnode.Id, np.zeros(24))[hour_bucket]

    return alpha * s_dist_m + (1.0 - alpha) * s_time_m - beta * p_j * d_fb.get(jnode.Id, 0.0)


# =============================================================================
# 4. BR-CWS OPTION C  (one GRASP iteration)
# =============================================================================

def br_CWS_c(active_nodes, savings_list, vehicle_cap: float, depot,
            hourly_probs: dict[int, np.ndarray], d_fb: dict[int, float],
            alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT,
            departure_h: float = 8.0, p: float = P_BIAS, rng=None) -> Solution:
    """
    Candidates sampled in the same distance-ranked, biased-random order as
    heuristic.br_CWS, but accepted only if the fused, recalculated saving
    (distance + live time + hourly ML penalty) is still positive.

    rng: optional random.Random instance (see heuristic.br_CWS's docstring
    on common random numbers across methods -- HSR/SRA both funnel through
    this same function). Defaults to the global `random` module.
    """
    rng = rng or random
    sol          = _build_dummy_solution_dynamic(active_nodes, depot, departure_h)
    local_sav    = list(savings_list)
    log_p        = math.log(p)
    dep_speed_ms = speed_ms(departure_h * 3600.0)

    while local_sav:
        u   = rng.random()
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

        s_fused = _fused_saving_c(inode, jnode, iRoute, edge, hourly_probs, d_fb,
                                  alpha, beta, dep_speed_ms)
        if s_fused <= 0.0:
            continue

        _merge_routes_dynamic(inode, jnode, iRoute, jRoute, edge, sol)

    return sol


# =============================================================================
# 4b. BR-CWS OPTION C -- DYNAMIC PRIORITY  (one GRASP iteration)
# =============================================================================

def _group_out_edges(savings_list) -> dict[int, list]:
    """Group the static K-NN candidate arcs by origin node id, so each active
    node's outgoing edges are retrievable in O(1). Built once per br_CWS_c_
    dynamic call; the grouping's origins never change during construction (only
    which edge is SELECTED does)."""
    out_edges: dict[int, list] = {}
    for edge in savings_list:
        out_edges.setdefault(edge.origin.Id, []).append(edge)
    return out_edges


def _recompute_best_candidate_c(route, out_edges: dict[int, list],
                                hourly_probs: dict[int, np.ndarray],
                                d_fb: dict[int, float],
                                alpha: float, beta: float,
                                dep_speed_ms: float) -> None:
    """Set route.best_candidate = (score, edge) to the max-scoring outgoing arc
    from route's CURRENT TAIL node, skipping targets in route.excluded_targets.
    None if the tail has no remaining viable candidate (so the route drops out
    of the live pool permanently). Score = _fused_saving_c, unchanged; head-
    validity is deliberately NOT checked here (it is the separate, cheap
    _check_merging step, and the score formula doesn't depend on it)."""
    tail     = route.edges[-1].origin
    excluded = route.excluded_targets
    best_edge  = None
    best_score = None
    for edge in out_edges.get(tail.Id, ()):
        if edge.end.Id in excluded:
            continue
        score = _fused_saving_c(tail, edge.end, route, edge, hourly_probs,
                                d_fb, alpha, beta, dep_speed_ms)
        if best_score is None or score > best_score:
            best_score = score
            best_edge  = edge
    route.best_candidate = (best_score, best_edge) if best_edge is not None else None


def br_CWS_c_dynamic(active_nodes, savings_list, vehicle_cap: float, depot,
                     hourly_probs: dict[int, np.ndarray], d_fb: dict[int, float],
                     alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT,
                     departure_h: float = 8.0, p: float = P_BIAS, rng=None,
                     top_k: int = 30, out_edges: dict[int, list] | None = None) -> Solution:
    """
    One BR-CWS pass where the fused (distance + live time + hourly ML) saving
    drives PRIORITISATION, not just a post-hoc veto (see module docstring for
    why the veto in br_CWS_c is inert). Each active route keeps its single
    best-scoring outgoing arc from its current tail node; each step samples --
    biased-random, geometric index -- from the top-`top_k` routes by score,
    checks feasibility with the unchanged _check_merging, and merges via
    _merge_routes_dynamic. Reuses _fused_saving_c exactly as br_CWS_c does.

    Selection uses a lazy-invalidation max-heap of per-route best candidates
    rather than rebuilding + nlargest-ing the whole live pool every step: on a
    100-node instance the two are indistinguishable, but the pool is dominated
    by RETRY steps (a target node is the top K-NN candidate of many routes, yet
    only one can claim it as its head), and an O(R) scan per retry made an
    early version ~19x slower than br_CWS_c on a 949-node instance. The heap
    pops only the few entries the geometric index actually needs and discards
    stale ones lazily (a route's entry is stale once its best_candidate has
    been recomputed -- tracked by a per-route token -- or the route was merged
    away), keeping each step O(idx * log(heap)) instead of O(R). This changes
    only tie-breaking on exactly-equal scores versus a full nlargest, never the
    rank that the geometric index selects.

    rng: optional random.Random instance (see heuristic.br_CWS's docstring on
    common random numbers across methods). Defaults to the global `random`
    module. savings_list is used ONLY to index each node's precomputed K-NN
    outgoing edges -- the candidate graph itself is not rebuilt.

    out_edges: precomputed _group_out_edges(savings_list) result, reused
    across every GRASP iteration by run_grasp_c_dynamic (the grouping is
    invariant across iterations of the same run -- only recomputing it once
    per run_grasp_c_dynamic call, not once per br_CWS_c_dynamic call, avoids
    O(n*K) repeated work n_iter times). Computed on the fly if omitted (e.g.
    when calling this function directly, outside run_grasp_c_dynamic).
    """
    rng = rng or random
    sol          = _build_dummy_solution_dynamic(active_nodes, depot, departure_h)
    log_p        = math.log(p)
    dep_speed_ms = speed_ms(departure_h * 3600.0)

    if out_edges is None:
        out_edges = _group_out_edges(savings_list)
    heap: list = []
    seq = itertools.count()

    def refresh(route) -> None:
        """Recompute route's best candidate and, if any, publish a fresh heap
        entry (bumping its token so older entries for this route pop as stale)."""
        _recompute_best_candidate_c(route, out_edges, hourly_probs, d_fb,
                                    alpha, beta, dep_speed_ms)
        if route.best_candidate is not None:
            route.cand_token += 1
            heapq.heappush(heap, (-route.best_candidate[0], next(seq),
                                  route, route.cand_token))

    for route in sol.routes:
        route.excluded_targets = set()
        route.cand_token = 0
        refresh(route)

    while heap:
        u       = rng.random()
        raw_idx = min(int(math.floor(math.log(max(u, 1e-300)) / log_p)), top_k - 1)

        # Pop the (raw_idx+1) highest-scoring still-valid entries (or fewer if
        # the heap runs out); stale entries are dropped permanently. The one to
        # act on is always the last valid entry popped -- see loop invariant.
        picked: list = []
        while heap and len(picked) <= raw_idx:
            entry = heapq.heappop(heap)
            route, token = entry[2], entry[3]
            if route.best_candidate is None or route.cand_token != token:
                continue
            picked.append(entry)
        if not picked:
            break
        sel = picked[-1]
        for entry in picked[:-1]:
            heapq.heappush(heap, entry)

        iRoute = sel[2]
        edge   = iRoute.best_candidate[1]
        inode  = edge.origin
        jnode  = edge.end
        jRoute = jnode.inRoute

        if not _check_merging(inode, jnode, iRoute, jRoute, vehicle_cap):
            iRoute.excluded_targets.add(jnode.Id)
            refresh(iRoute)
            continue

        _merge_routes_dynamic(inode, jnode, iRoute, jRoute, edge, sol)
        jRoute.best_candidate = None
        iRoute.excluded_targets = set()
        refresh(iRoute)

    return sol


# =============================================================================
# 5. GRASP OPTION C
# =============================================================================

def run_grasp_c(nodes, dist_matrix: np.ndarray, vehicle_cap: float,
                bundle: dict | None = None, locker_cap: dict | None = None,
                n_iter: int = N_ITER, p_bias: float = P_BIAS,
                alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT,
                departure_h: float = 8.0, verbose: bool = True, rng=None
                ) -> tuple[Solution, float]:
    """
    GRASP loop: build the candidate graph + hourly P_j table once
    (build_graph_c), then repeat br_CWS_c n_iter times.

    Parameters
    ----------
    bundle     : optional Option-C model bundle (heuristic_learn.load_model()).
                 None keeps P_j == 0 everywhere (pure distance+time ablation).
    locker_cap : required for the ML penalty's capacity features; see
                 _hourly_saturation_probs.
    alpha      : distance/time blend weight [0,1]; 1=pure distance, 0=pure time.
    beta       : ML saturation penalty weight [0,1] (only used if bundle set).
    rng        : optional random.Random instance, threaded to every br_CWS_c
                 call (see heuristic.br_CWS's docstring on common random numbers).
    """
    t0 = _time.perf_counter()

    active_nodes, savings_list, hourly_probs, d_fb = build_graph_c(
        nodes, dist_matrix, bundle, locker_cap)
    depot     = nodes[0]
    best_sol  = None
    best_cost = INF

    for it in range(n_iter):
        sol = br_CWS_c(active_nodes, savings_list, vehicle_cap, depot,
                       hourly_probs, d_fb, alpha, beta, departure_h, p_bias, rng)
        if sol.cost < best_cost:
            best_sol  = sol
            best_cost = sol.cost
            if verbose:
                print(f"    iter {it+1:>4}: new best -> "
                      f"{best_cost/1000:.3f} km | {len(sol.routes)} routes")

    return best_sol, _time.perf_counter() - t0


def run_grasp_c_dynamic(nodes, dist_matrix: np.ndarray, vehicle_cap: float,
                        bundle: dict | None = None, locker_cap: dict | None = None,
                        n_iter: int = N_ITER, p_bias: float = P_BIAS,
                        alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT,
                        departure_h: float = 8.0, verbose: bool = True, rng=None
                        ) -> tuple[Solution, float]:
    """
    GRASP loop for the DYNAMIC-PRIORITY variant: same candidate graph +
    hourly P_j table as run_grasp_c (one build_graph_c call), but repeats
    br_CWS_c_dynamic -- so the fused distance+time+ML saving reorders which
    merges are attempted, instead of only vetoing them after the fact (see
    module docstring). Additive: run_grasp_c is left untouched.

    Parameters mirror run_grasp_c exactly; see that function's docstring.
    """
    t0 = _time.perf_counter()

    active_nodes, savings_list, hourly_probs, d_fb = build_graph_c(
        nodes, dist_matrix, bundle, locker_cap)
    depot     = nodes[0]
    best_sol  = None
    best_cost = INF

    # Computed ONCE per run_grasp_c_dynamic call and reused across every GRASP
    # iteration -- the K-NN grouping by origin node is invariant across
    # iterations, so recomputing it inside br_CWS_c_dynamic every time (as the
    # first version did) is wasted O(n*K) work repeated n_iter times.
    out_edges = _group_out_edges(savings_list)

    for it in range(n_iter):
        sol = br_CWS_c_dynamic(active_nodes, savings_list, vehicle_cap, depot,
                               hourly_probs, d_fb, alpha, beta, departure_h,
                               p_bias, rng, out_edges=out_edges)
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
        print("Usage: python heuristic_c.py <instance.txt> [model.pkl] [alpha] [beta] [departure_h]")
        print("       (omit model.pkl for the pure distance+time mode, no ML)")
        sys.exit(1)

    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    alpha_val  = float(sys.argv[3]) if len(sys.argv) > 3 else ALPHA_DEFAULT
    beta_val   = float(sys.argv[4]) if len(sys.argv) > 4 else BETA_DEFAULT
    params, nodes, dist_matrix, orders, locker_cap = read_full_instance(sys.argv[1])
    dep_h = float(sys.argv[5]) if len(sys.argv) > 5 else params.departure

    bundle = load_model(model_path) if model_path else None

    print(f"Instance: {params.name}  "
          f"({params.n_orders} orders, {params.n_nodes} nodes, cap={params.capacity})")

    label = f"C dep={dep_h}h alpha={alpha_val}" + (f" +ML beta={beta_val}" if bundle else "")
    best, elapsed = run_grasp_c(nodes, dist_matrix, params.capacity,
                               bundle=bundle, locker_cap=locker_cap,
                               n_iter=100, alpha=alpha_val, beta=beta_val,
                               departure_h=dep_h, verbose=True)
    print_solution(best, f"{params.name} [{label}]")
    print(f"\nSolved in {elapsed:.2f} s")
