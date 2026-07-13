"""
simulator.py
------------
Simulate a BR-CWS/GRASP solution against actual locker capacity constraints.

When a vehicle arrives at a locker with insufficient compartments for a
delivery, it redirects to the nearest locker with a free compartment of
adequate size — exactly as specified by Rudy (2025).

A shared compartment state is maintained across all routes, so fallback
deliveries at an alternative locker consume capacity there before that
locker's own route vehicle arrives. This interaction is key: route structure
affects whether cascading capacity shortfalls occur.

Usage
-----
    from simulator import simulate_solution
    fallback_km, n_fallbacks = simulate_solution(sol, orders, locker_cap, dist_matrix)

    from simulator import simulate_solution_with_overflow
    fallback_km, n_fallbacks, overflow_locations = simulate_solution_with_overflow(
        sol, orders, locker_cap, dist_matrix)

    from simulator import simulate_solution_stochastic
    visit_log = simulate_solution_stochastic(
        sol, orders, locker_cap, dist_matrix, departure_h=8.0,
        mean_hour=14.0, std_hour=3.0, start_h=6, end_h=22, rng=random.Random(0))

STOCHASTIC RELEASE MODEL (Option D)
------------------------------------
_simulate_core (above) only frees a compartment when a route-scheduled
PICKUP ORDER is visited by the vehicle -- it has no notion of a locker's
already-occupied compartments (from locker_cap, i.e. parcels sitting there
BEFORE this instance's routes even start) ever becoming free on their own.

In reality, a locker's existing occupants collect their own packages
throughout the day, independent of any vehicle route -- so a locker that
looks "full" in the static instance data may well have emptied out by the
time a vehicle *reaches* it later in the day. simulate_solution_stochastic
models this explicitly: each currently-occupied compartment (computed from
MAX_COMPARTMENTS - locker_cap's free count) is assigned an independent,
randomly-sampled release hour (see _sample_release_hours), and when the
vehicle arrives at hour h, the locker's *effective* free count already
reflects every background release with a sampled hour <= h -- on top of
(not instead of) the existing route-triggered pickup-order mechanic, which
is unrelated (a courier physically collecting an outbound parcel is a
different event from a customer self-collecting a delivered one).

This is a CENSORED model, not a truncated one: the per-hour cumulative
distribution _pickup_cdf is the raw Normal(mean_hour, std_hour) CDF, so
CDF(end_h) < 1 in general -- a compartment's occupant may simply never
show up within the simulated day (float('inf') release hour), rather than
being forced to release by end_h via renormalization (which would make
"a package can just sit there all day" impossible by construction).
"""

from __future__ import annotations
import bisect
import math
import numpy as np
from model import Solution


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _build_compartment_state(locker_cap: dict[int, dict[int, int]]
                              ) -> dict[int, dict[int, int]]:
    """Deep-copy locker_cap into a mutable state dict."""
    return {k: dict(v) for k, v in locker_cap.items()}


def _use_compartment(state: dict[int, dict[int, int]],
                     k: int, order_size: int) -> bool:
    """
    Try to use the smallest available compartment >= order_size at locker k.

    Updates state in-place.

    Returns True if a compartment was found and allocated, False otherwise.
    """
    comp = state.get(k)
    if comp is None:
        return False
    for s in range(order_size, 4):   # try size order_size, then larger
        if comp.get(s, 0) > 0:
            comp[s] -= 1
            return True
    return False


def _free_compartment(state: dict[int, dict[int, int]],
                      k: int, order_size: int) -> None:
    """
    Return a compartment of size order_size to locker k (pickup event).
    Creates the entry if missing (defensive).
    """
    if k not in state:
        state[k] = {}
    comp = state[k]
    comp[order_size] = comp.get(order_size, 0) + 1


def _find_fallback(original_k: int,
                   active_ids: list[int],
                   dist_matrix: np.ndarray,
                   state: dict[int, dict[int, int]],
                   order_size: int) -> tuple[int | None, float]:
    """
    Find the nearest locker (distance from original_k) that still has at
    least one compartment of size >= order_size available.

    Returns (alt_k, distance_to_alt) or (None, inf) if no alternative found.
    The Rudy rule: nearest candidate by distance from the *original* locker,
    not by distance from the vehicle's current position.
    """
    best_dist = float('inf')
    best_k    = None
    for j in active_ids:
        if j == original_k:
            continue
        comp = state.get(j, {})
        has_space = any(comp.get(s, 0) > 0 for s in range(order_size, 4))
        if has_space:
            d = float(dist_matrix[original_k, j])
            if d < best_dist:
                best_dist = d
                best_k    = j
    return best_k, best_dist


# =============================================================================
# CORE SIMULATION  (shared by simulate_solution / simulate_solution_with_overflow)
# =============================================================================

def _simulate_core(sol: Solution,
                   orders: list[dict],
                   locker_cap: dict[int, dict[int, int]],
                   dist_matrix: np.ndarray
                   ) -> tuple[float, int, dict[int, int]]:
    """
    Simulate solution execution under locker capacity constraints.

    For each delivery order at its intended locker k:
      - If a compartment of size >= order.size is available → use it.
      - Else → fallback: find nearest locker j with a free compartment,
        deliver there, add dist(k, j) as extra distance.

    For each pickup order:
      - Free a compartment of the order's size at the visited locker.

    Compartment state is SHARED across all routes, so a fallback delivery
    to locker j consumes capacity before j's own scheduled route vehicle
    arrives. This models the real-world cascade where one locker's overflow
    can affect another locker's available space.

    Routes are processed in the order they appear in sol.routes.
    Within each route, nodes are visited in route order.

    Parameters
    ----------
    sol        : Solution from run_grasp / run_grasp_learn
    orders     : raw order list {type, location, weight, size}
    locker_cap : {location_id: {size: compartment_count}}
    dist_matrix: distance matrix in metres, shape (m+1, m+1)

    Returns
    -------
    fallback_km        : total extra distance from fallback detours (km)
    n_fallbacks        : number of individual orders that triggered a fallback
    overflow_locations : {location_id: 1} for every locker where at least one
                         delivery triggered a fallback (real, route-order and
                         cascading-state-dependent overflow -- used as the
                         Option C training label, see generate_labels_C.py)
    """
    if not locker_cap:
        return 0.0, 0, {}      # no capacity data → can't simulate

    # --- Group orders by location ---
    del_orders: dict[int, list[dict]] = {}
    pck_orders: dict[int, list[dict]] = {}
    for o in orders:
        loc = o['location']
        if o['type'] == 'delivery':
            del_orders.setdefault(loc, []).append(o)
        else:
            pck_orders.setdefault(loc, []).append(o)

    # --- Mutable locker state (shared across all routes) ---
    state = _build_compartment_state(locker_cap)

    # --- Active locker IDs (all nodes that appear in any route) ---
    active_ids: list[int] = list({
        node.Id
        for route in sol.routes
        for node in route.nodes
        if node.Id != 0
    })

    fallback_m         = 0.0
    n_fallbacks        = 0
    overflow_locations: dict[int, int] = {}

    # --- Process routes sequentially ---
    for route in sol.routes:
        for node in route.nodes:
            k = node.Id
            if k == 0:          # depot — skip
                continue

            # 1. Deliveries: try to place each order; fallback if full
            for order in del_orders.get(k, []):
                s = int(order.get('size', 1))
                if _use_compartment(state, k, s):
                    pass        # success — compartment consumed at k
                else:
                    # Fallback: find nearest locker with free compartment >= s
                    alt_k, d_extra = _find_fallback(k, active_ids,
                                                    dist_matrix, state, s)
                    overflow_locations[k] = 1
                    if alt_k is not None:
                        fallback_m  += d_extra
                        n_fallbacks += 1
                        _use_compartment(state, alt_k, s)
                    # If no alternative: package undelivered (count as fallback
                    # with zero extra distance — edge case, should be rare)
                    else:
                        n_fallbacks += 1

            # 2. Pickups: free a compartment of the order's size
            for order in pck_orders.get(k, []):
                s = int(order.get('size', 1))
                _free_compartment(state, k, s)

    return fallback_m / 1000.0, n_fallbacks, overflow_locations


# =============================================================================
# PUBLIC API
# =============================================================================

def simulate_solution(sol: Solution,
                      orders: list[dict],
                      locker_cap: dict[int, dict[int, int]],
                      dist_matrix: np.ndarray
                      ) -> tuple[float, int]:
    """Same as simulate_solution_with_overflow, without the per-locker
    overflow dict. See _simulate_core for the shared implementation."""
    fallback_km, n_fallbacks, _ = _simulate_core(sol, orders, locker_cap, dist_matrix)
    return fallback_km, n_fallbacks


def simulate_solution_with_overflow(sol: Solution,
                                    orders: list[dict],
                                    locker_cap: dict[int, dict[int, int]],
                                    dist_matrix: np.ndarray
                                    ) -> tuple[float, int, dict[int, int]]:
    """Same as simulate_solution, plus a per-locker real-overflow dict
    ({location_id: 1} for lockers where >=1 delivery triggered a fallback).
    See _simulate_core for the shared implementation."""
    return _simulate_core(sol, orders, locker_cap, dist_matrix)


# =============================================================================
# STOCHASTIC RELEASE MODEL  (Option D — see module docstring)
# =============================================================================

def _pickup_cdf(hour: float, mean_hour: float, std_hour: float) -> float:
    """Raw (non-renormalized) Normal(mean_hour, std_hour) CDF at `hour`."""
    z = (hour - mean_hour) / (std_hour * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _sample_release_hours(rng, n: int, mean_hour: float, std_hour: float,
                          start_h: int, end_h: int) -> list[float]:
    """
    Sample n independent release hours from the CENSORED (not truncated)
    Normal(mean_hour, std_hour) pickup-time model, resolved against the
    integer hour grid [start_h, end_h].

    For each draw: u ~ Uniform(0,1); if u > CDF(end_h) -> float('inf') (this
    compartment's occupant never self-collects within the simulated day —
    CENSORED, since CDF(end_h) < 1 by construction and that missing mass is
    NOT renormalized back into the window). Otherwise, the smallest integer
    hour h in [start_h, end_h] with CDF(h) >= u is returned — this also
    naturally handles occupants whose "true" pickup time predates start_h:
    they're simply already available the moment the window opens, with no
    special-casing needed.
    """
    hours = list(range(start_h, end_h + 1))
    cdfs  = [_pickup_cdf(h, mean_hour, std_hour) for h in hours]
    cdf_end = cdfs[-1]

    out: list[float] = []
    for _ in range(n):
        u = rng.random()
        if u > cdf_end:
            out.append(float('inf'))
            continue
        for h, c in zip(hours, cdfs):
            if c >= u:
                out.append(float(h))
                break
    return out


def _stochastic_free_count(state: dict[int, dict[int, int]],
                           release_schedule: dict[int, dict[int, list[float]]],
                           k: int, s: int, hour: float) -> int:
    """
    Effective free-compartment count of size s at locker k at time `hour`:
    the route-driven ledger (state[k][s], mutated by deliveries/pickups only
    -- never by background releases) PLUS however many of that locker's
    originally-occupied size-s compartments have a sampled release hour
    <= hour so far (computed fresh via bisect each call, not accumulated —
    stateless in hour, so no double-counting risk).
    """
    base = state.get(k, {}).get(s, 0)
    sched = release_schedule.get(k, {}).get(s)
    if not sched:
        return base
    return base + bisect.bisect_right(sched, hour)


def _use_compartment_stochastic(state: dict[int, dict[int, int]],
                                release_schedule: dict[int, dict[int, list[float]]],
                                k: int, order_size: int, hour: float) -> bool:
    """Same role as _use_compartment, but checks the background-release-aware
    effective free count (_stochastic_free_count) instead of raw state."""
    for s in range(order_size, 4):
        if _stochastic_free_count(state, release_schedule, k, s, hour) > 0:
            comp = state.setdefault(k, {})
            comp[s] = comp.get(s, 0) - 1
            return True
    return False


def simulate_solution_stochastic(sol: Solution,
                                 orders: list[dict],
                                 locker_cap: dict[int, dict[int, int]],
                                 dist_matrix: np.ndarray,
                                 departure_h: float,
                                 mean_hour: float,
                                 std_hour: float,
                                 start_h: int,
                                 end_h: int,
                                 rng) -> list[dict]:
    """
    ONE stochastic replica of solution execution under the background
    self-collection release model (see module docstring).

    For each locker, every originally-occupied compartment (MAX_COMPARTMENTS
    minus locker_cap's free count, per size) is given an independent sampled
    release hour. When the vehicle's real arrival hour at a locker (replayed
    via heuristic_learn._compute_arrival_times, exact for a finished
    Solution) is reached, its effective free count already reflects every
    background release with a sampled hour <= arrival hour — on top of the
    existing route-triggered pickup-order mechanic (deterministic, kept
    unchanged: a courier collecting an outbound parcel is a separate event).

    A failed delivery still tries the same nearest-available-locker fallback
    as _simulate_core (for cascading-state realism against OTHER lockers'
    background-release-aware capacity), but the outcome of that fallback is
    NOT what determines delivered_ok — matching Option C's overflow_locations
    convention, this label is about whether the ORIGINAL locker had capacity,
    not whether a redirect happened to succeed elsewhere.

    Returns a visit_log: one row per locker visit with >=1 delivery order,
    {location, initial_occupancy_ratio, arrival_hour, delivered_ok}.
    delivered_ok=1 iff EVERY delivery order at that visit found a
    compartment directly at its own locker (locker-level granularity).
    """
    if not locker_cap:
        return []

    from heuristic_learn import _compute_arrival_times
    from instance_reader import MAX_COMPARTMENTS

    arrivals = _compute_arrival_times(sol, departure_h, dist_matrix)

    # --- Group orders by location ---
    del_orders: dict[int, list[dict]] = {}
    pck_orders: dict[int, list[dict]] = {}
    for o in orders:
        loc = o['location']
        if o['type'] == 'delivery':
            del_orders.setdefault(loc, []).append(o)
        else:
            pck_orders.setdefault(loc, []).append(o)

    # --- Mutable route-driven ledger (background releases NOT mutated in) ---
    state = _build_compartment_state(locker_cap)

    # --- Static initial_occupancy_ratio (independent of the replica) ---
    total_max = sum(MAX_COMPARTMENTS.values())
    init_occ_ratio: dict[int, float] = {}
    for k, cap in locker_cap.items():
        free = sum(cap.get(s, 0) for s in (1, 2, 3))
        init_occ_ratio[k] = 1.0 - free / total_max

    # --- Per-locker, per-size background release schedule (this replica) ---
    release_schedule: dict[int, dict[int, list[float]]] = {}
    for k, cap in locker_cap.items():
        by_size: dict[int, list[float]] = {}
        for s in (1, 2, 3):
            n_occupied = MAX_COMPARTMENTS[s] - cap.get(s, 0)
            if n_occupied > 0:
                by_size[s] = sorted(_sample_release_hours(
                    rng, n_occupied, mean_hour, std_hour, start_h, end_h))
        release_schedule[k] = by_size

    active_ids: list[int] = list({
        node.Id for route in sol.routes for node in route.nodes if node.Id != 0
    })

    visit_log: list[dict] = []

    # --- Process routes sequentially (same traversal as _simulate_core) ---
    for route in sol.routes:
        for node in route.nodes:
            k = node.Id
            if k == 0:
                continue
            hour = (arrivals.get(k, departure_h * 3600.0) / 3600.0) % 24.0

            dels = del_orders.get(k, [])
            if dels:
                delivered_ok = True
                for order in dels:
                    s = int(order.get('size', 1))
                    if _use_compartment_stochastic(state, release_schedule, k, s, hour):
                        continue
                    delivered_ok = False
                    # Fallback for cascading-state realism (see docstring) --
                    # outcome doesn't affect this locker's delivered_ok.
                    best_dist, alt_k = float('inf'), None
                    for j in active_ids:
                        if j == k:
                            continue
                        if any(_stochastic_free_count(state, release_schedule, j, s2, hour) > 0
                               for s2 in range(s, 4)):
                            d = float(dist_matrix[k, j])
                            if d < best_dist:
                                best_dist, alt_k = d, j
                    if alt_k is not None:
                        _use_compartment_stochastic(state, release_schedule, alt_k, s, hour)
                visit_log.append({
                    'location': k,
                    'initial_occupancy_ratio': init_occ_ratio.get(k, 0.0),
                    'arrival_hour': hour,
                    'delivered_ok': 1 if delivered_ok else 0,
                })

            for order in pck_orders.get(k, []):
                s = int(order.get('size', 1))
                _free_compartment(state, k, s)

    return visit_log
