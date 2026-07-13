"""
instance_reader.py
------------------
Parser for Rudy (2025) PLBDP benchmark instances.

File format (one .txt per instance):
    Line 1      :  n  m  z          (n orders, m locker nodes, z size categories)
    Line 2      :  P  S  C  B       (parking_s, service_s, vehicle_capacity, departure_h)
    Lines 3..n+2:  type  weight  location  size   (one order per line)
                   type: 0=delivery, 1/2=pickup
    Lines n+3.. :  distance matrix, (m+1) x (m+1), depot=row/col 0, metres
    After matrix:  locker capacity section, m rows:
                   k  count_size1  count_size2  count_size3

Reference: J. Rudy, "Multi-criteria parcel locker-based vehicle routing with
           pickup and delivery", Archives of Control Sciences, 35(3), 2025.
"""

from __future__ import annotations
import os
from collections import defaultdict
import numpy as np
from model import Node
from speed_profile import SPEED_PROFILE_KMH as _SPEED_PROFILE_KMH, V_MAX_MS as _V_MAX_MS


def _speed_kmh(hour: float) -> float:
    """Speed in m/s for the given hour of day (float, e.g. 8.5 = 8:30 AM)."""
    return _SPEED_PROFILE_KMH[int(hour) % 24] / 3.6


# =============================================================================
# FEATURE COLUMN ORDER  (must match between training and inference)
# =============================================================================
# NOTE: models trained with 9 features skip the last 4 (temporal/spatial).
#       The first 9 columns are kept first and in the same order for backward
#       compatibility — old 9-feature models still work when callers slice
#       bundle['features'][:, :9].
FEATURE_COLS = [
    'n_deliveries',           # raw count
    'n_pickups',              # raw count
    'total_orders',           # n_deliveries + n_pickups
    'delivery_weight',        # absolute (kg)
    'pickup_weight',          # absolute (kg)
    'delivery_ratio',         # n_deliveries / total_orders
    'weight_per_delivery',    # delivery_weight / n_deliveries  (0 if no deliveries)
    'relative_demand',        # delivery_weight / mean_del_weight across active nodes
    'demand_concentration',   # delivery_weight / total_instance_delivery_weight
    # Temporal / spatial features (require dist_matrix)
    'dist_to_depot_km',       # straight-line road dist depot→locker (km)
    'estimated_arrival_h',    # h_dep + dist/(v_dep*3600): when vehicle likely arrives
    'congestion_factor',      # (v_max - v_arrival) / v_max  ∈ [0,1]
    'time_pressure',          # n_deliveries × congestion_factor
]

# Feature columns for Model B (binary overflow prediction, train_model_B.py).
# Model B's label is a pure capacity bin-packing check (see
# train_model_B.saturation_label_overflow) with no time/distance dependence,
# so the 4 temporal/spatial columns above are replaced with capacity-aware
# ones computed from locker_cap -- see extract_features_capacity.
FEATURE_COLS_B = FEATURE_COLS[:9] + [
    'total_compartments',      # sum(locker_cap[k].values()) -- raw capacity
    'utilization_ratio',       # n_deliveries / total_compartments
    'large_size_share',        # fraction of deliveries requesting a size-3 compartment
    'size_weighted_demand',    # sum of order sizes (proxy for compartment pressure)
]

# Feature columns for Option C (heuristic_c.py's unified time-aware learnheuristic).
# Same capacity-aware base as FEATURE_COLS_B (overflow is fundamentally about
# capacity) plus an explicit arrival_hour column -- the label this model is
# trained on is REAL simulated overflow conditioned on when the locker was
# actually visited (see generate_labels_C.py), not a time-independent risk.
FEATURE_COLS_C = FEATURE_COLS_B + ['arrival_hour']

# Feature columns for Option D (heuristic_d.py / train_model_D.py): same 13
# capacity-aware base columns as Model B, plus the locker's OWN initial
# occupancy ratio (computable identically at train and inference time, no
# simulation needed) and arrival_hour -- the model predicts
# P(delivered_ok | initial_occupancy_ratio, arrival_hour) directly (see
# generate_labels_D.py / simulator.simulate_solution_stochastic).
FEATURE_COLS_D = FEATURE_COLS_B + ['initial_occupancy_ratio', 'arrival_hour']

# Type alias for locker capacity:  {location_id: {size(1/2/3): compartment_count}}
LockerCap = dict[int, dict[int, int]]

# Max compartments per size, per locker (instance.h's maxA/maxB/maxC,
# 1-indexed here to match this codebase's size convention). Shared by
# generate_instances_B.py, simulator.py's stochastic release model, and
# heuristic_d.py's initial_occupancy_ratio computation.
MAX_COMPARTMENTS = {1: 32, 2: 29, 3: 18}


# =============================================================================
# INSTANCE PARAMETERS
# =============================================================================
class InstanceParams:
    """Scalar parameters extracted from the header of a Rudy instance file."""
    __slots__ = ('n_orders', 'n_nodes', 'n_sizes',
                 'parking', 'service', 'capacity', 'departure', 'name')

    def __init__(self, n, m, z, P, S, C, B, name=''):
        self.n_orders  = n
        self.n_nodes   = m
        self.n_sizes   = z
        self.parking   = P    # seconds per stop
        self.service   = S    # seconds per order
        self.capacity  = C    # vehicle weight capacity
        self.departure = B    # departure hour (float)
        self.name      = name

    def __repr__(self):
        return (f"InstanceParams(name={self.name!r}, "
                f"orders={self.n_orders}, nodes={self.n_nodes}, "
                f"cap={self.capacity})")


# =============================================================================
# CORE READER
# =============================================================================

def _read_lines(filepath: str) -> list[str]:
    """Read non-empty stripped lines from a file."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        return [l.strip() for l in fh if l.strip()]


def _parse_locker_cap(lines: list[str],
                      after_idx: int,
                      n_matrix_values: int) -> 'LockerCap':
    """
    Parse the locker capacity section that follows the distance matrix.

    Skips exactly n_matrix_values whitespace-separated values starting at
    lines[after_idx], then reads m rows of the form:

        k  count_size1  count_size2  count_size3

    Returns
    -------
    locker_cap : {location_id: {1: n_small, 2: n_medium, 3: n_large}}
                 Empty dict if the section is missing or malformed.
    """
    # Advance past the distance matrix values
    count = 0
    cur   = after_idx
    while cur < len(lines) and count < n_matrix_values:
        count += len(lines[cur].split())
        cur   += 1

    # Parse locker capacity rows
    locker_cap: LockerCap = {}
    while cur < len(lines):
        cols = lines[cur].split()
        if len(cols) >= 4:
            try:
                k = int(cols[0])
                locker_cap[k] = {1: int(cols[1]), 2: int(cols[2]), 3: int(cols[3])}
            except ValueError:
                pass
        cur += 1
    return locker_cap


def read_orders_only(filepath: str
                     ) -> tuple[InstanceParams, list[dict], 'LockerCap']:
    """
    Fast read: parse header, orders, AND locker capacity section.
    Skips the large distance matrix (not needed for ML training).

    Order line format (Rudy 2025): type  weight  location  size
      type: 0=delivery, 1/2=pickup

    Returns
    -------
    params     : InstanceParams
    orders     : list of dicts  {type, location, weight, size}
    locker_cap : {location_id: {1: n_small, 2: n_medium, 3: n_large}}
                 Empty dict if the capacity section is absent.
    """
    lines = _read_lines(filepath)
    name  = os.path.splitext(os.path.basename(filepath))[0]

    n, m, z = map(int,   lines[0].split())
    P, S, C, B = map(float, lines[1].split())
    params = InstanceParams(n, m, z, P, S, C, B, name)

    orders = []
    for i in range(2, 2 + n):
        if i >= len(lines):
            break
        cols = lines[i].split()
        if len(cols) < 3:
            continue
        orders.append({
            'type':     'delivery' if int(cols[0]) == 0 else 'pickup',
            'weight':   float(cols[1]),
            'location': int(cols[2]),
            'size':     int(cols[3]) if len(cols) > 3 else 1,
        })

    # Locker capacity section lives after (m+1)² distance matrix values
    locker_cap = _parse_locker_cap(lines, 2 + n, (m + 1) ** 2)
    return params, orders, locker_cap


def read_full_instance(filepath: str,
                       ) -> tuple[InstanceParams, list[Node], np.ndarray,
                                  list[dict], 'LockerCap']:
    """
    Full read: parse header, orders, distance matrix, AND locker capacity.
    Used during routing experiments.

    Returns
    -------
    params      : InstanceParams
    nodes       : list of Node objects, index 0 = depot
                  (demands aggregated from orders)
    dist_matrix : np.ndarray, shape (m+1, m+1), metres
    orders      : list of raw order dicts {type, location, weight, size}
                  (needed by the simulator for per-order fallback logic)
    locker_cap  : {location_id: {1: n_small, 2: n_medium, 3: n_large}}
                  (empty dict if the capacity section is absent)
    """
    lines = _read_lines(filepath)
    name  = os.path.splitext(os.path.basename(filepath))[0]

    # --- Header ---
    n, m, z = map(int,   lines[0].split())
    P, S, C, B = map(float, lines[1].split())
    params = InstanceParams(n, m, z, P, S, C, B, name)

    # --- Orders ---
    # Column order (Rudy 2025): type  weight  location  size
    orders = []
    for i in range(2, 2 + n):
        if i >= len(lines):
            break
        cols = lines[i].split()
        if len(cols) < 3:
            continue
        orders.append({
            'type':     'delivery' if int(cols[0]) == 0 else 'pickup',
            'weight':   float(cols[1]),
            'location': int(cols[2]),
            'size':     int(cols[3]) if len(cols) > 3 else 1,
        })

    # --- Build Node objects ---
    nodes = [Node(i) for i in range(m + 1)]
    for o in orders:
        loc = o['location']
        if loc < 1 or loc > m:
            continue
        node = nodes[loc]
        w    = o['weight']
        if o['type'] == 'delivery':
            node.delivery_weight += w
            node.n_deliveries    += 1
            s = int(o.get('size', 1))
            node.size_weighted_demand += s
            if s >= 3:
                node.n_large_deliveries += 1
        else:
            node.pickup_weight += w
            node.n_pickups     += 1
        node.service_time += S
    for node in nodes[1:]:
        if node.is_active:
            node.service_time += P

    # --- Distance matrix  (m+1) x (m+1) ---
    needed    = (m + 1) ** 2
    dist_flat: list[float] = []
    mat_start = 2 + n
    for row in lines[mat_start:]:
        dist_flat.extend(map(float, row.split()))
        if len(dist_flat) >= needed:
            break
    dist_matrix = np.array(dist_flat[:needed], dtype=float).reshape(m + 1, m + 1)

    # --- Locker capacity section (after the distance matrix) ---
    locker_cap = _parse_locker_cap(lines, mat_start, needed)

    return params, nodes, dist_matrix, orders, locker_cap


# =============================================================================
# FEATURE EXTRACTION  (for ML training and inference)
# =============================================================================

def _aggregate_demand(orders: list[dict]
                      ) -> tuple[dict, dict, dict, dict, list[int], float, float]:
    """
    Shared demand aggregation used by both extract_features (Model A) and
    extract_features_capacity (Model B). Returns
    (n_del, n_pck, w_del, w_pck, all_locs, mean_del_w, total_del_w).
    """
    n_del: dict[int, float] = defaultdict(float)
    n_pck: dict[int, float] = defaultdict(float)
    w_del: dict[int, float] = defaultdict(float)
    w_pck: dict[int, float] = defaultdict(float)

    for o in orders:
        k = o['location']
        if o['type'] == 'delivery':
            n_del[k] += 1
            w_del[k] += o['weight']
        else:
            n_pck[k] += 1
            w_pck[k] += o['weight']

    all_locs    = sorted(set(n_del) | set(n_pck))
    total_del_w = sum(w_del.values()) or 1.0
    mean_del_w  = (sum(w_del[k] for k in all_locs) / len(all_locs)
                   if all_locs else 1.0)
    return n_del, n_pck, w_del, w_pck, all_locs, mean_del_w, total_del_w


def _base_demand_features(k: int, n_del: dict, n_pck: dict, w_del: dict,
                          w_pck: dict, mean_del_w: float,
                          total_del_w: float) -> dict:
    """The 9 demand-aggregation features shared by both feature sets."""
    total_ord = n_del[k] + n_pck[k]
    return {
        'n_deliveries':        n_del[k],
        'n_pickups':           n_pck[k],
        'total_orders':        total_ord,
        'delivery_weight':     w_del[k],
        'pickup_weight':       w_pck[k],
        'delivery_ratio':      n_del[k] / total_ord if total_ord > 0 else 0.0,
        'weight_per_delivery': w_del[k] / n_del[k] if n_del[k] > 0 else 0.0,
        'relative_demand':     w_del[k] / mean_del_w,
        'demand_concentration': w_del[k] / total_del_w,
    }


def extract_features(params: InstanceParams,
                     orders: list[dict],
                     dist_matrix=None,
                     departure_h: float = 8.0) -> dict[int, dict]:
    """
    Compute one feature vector per non-depot node (Model A feature set).

    Uses defaultdict so arbitrary location IDs are safe regardless of
    whether they exceed the m value stated in the header.

    Relative features (delivery_ratio, relative_demand, demand_concentration)
    are normalised within the instance so they transfer across instances
    of different scales (number of orders, weights, etc.).

    Parameters
    ----------
    params      : InstanceParams
    orders      : list of raw order dicts {type, location, weight, size}
    dist_matrix : optional np.ndarray shape (m+1, m+1), metres.
                  When provided, the four temporal/spatial features are
                  computed from actual depot distances; otherwise they are
                  set to 0 / departure_h so the array shape matches FEATURE_COLS.
    departure_h : vehicle departure hour of day (float, default 8.0 = 08:00).

    Returns
    -------
    features : {node_id: {feature_name: value}}
    """
    n_del, n_pck, w_del, w_pck, all_locs, mean_del_w, total_del_w = \
        _aggregate_demand(orders)

    features: dict[int, dict] = {}
    for k in all_locs:
        features[k] = _base_demand_features(
            k, n_del, n_pck, w_del, w_pck, mean_del_w, total_del_w)
        # --- Temporal / spatial features ---
        if dist_matrix is not None:
            d0k_m  = float(dist_matrix[0, k]) if k < dist_matrix.shape[0] else 0.0
            v_dep  = _speed_kmh(departure_h)                   # m/s at departure
            t_arr  = departure_h + d0k_m / (v_dep * 3600.0)   # estimated arrival hour
            v_arr  = _speed_kmh(t_arr)                         # m/s at arrival
            cong   = (_V_MAX_MS - v_arr) / _V_MAX_MS           # congestion factor ∈ [0,1]
            features[k]['dist_to_depot_km']    = d0k_m / 1000.0
            features[k]['estimated_arrival_h'] = t_arr
            features[k]['congestion_factor']   = cong
            features[k]['time_pressure']       = n_del[k] * cong
        else:
            features[k]['dist_to_depot_km']    = 0.0
            features[k]['estimated_arrival_h'] = departure_h
            features[k]['congestion_factor']   = 0.0
            features[k]['time_pressure']       = 0.0
    return features


def extract_features_capacity(params: InstanceParams,
                              orders: list[dict],
                              locker_cap: 'LockerCap') -> dict[int, dict]:
    """
    Compute one feature vector per non-depot node (Model B feature set).

    Model B's label (train_model_B.saturation_label_overflow) is a pure
    greedy bin-packing check of order sizes against locker_cap compartments
    -- it has NO dependence on distance or time of day. The 4 distance/time
    features used by extract_features (dist_to_depot_km, estimated_arrival_h,
    congestion_factor, time_pressure) therefore carry no causal signal for
    this label. This function replaces them with capacity-aware features
    computed directly from locker_cap -- the same data the label itself is
    derived from, but never previously exposed as a feature.

    Parameters
    ----------
    params      : InstanceParams
    orders      : list of raw order dicts {type, location, weight, size}
    locker_cap  : {location_id: {size(1/2/3): compartment_count}}

    Returns
    -------
    features : {node_id: {feature_name: value}}  (columns = FEATURE_COLS_B)
    """
    n_del, n_pck, w_del, w_pck, all_locs, mean_del_w, total_del_w = \
        _aggregate_demand(orders)

    size_sum: dict[int, float] = defaultdict(float)
    n_large:  dict[int, float] = defaultdict(float)
    for o in orders:
        if o['type'] == 'delivery':
            k = o['location']
            s = int(o.get('size', 1))
            size_sum[k] += s
            if s >= 3:
                n_large[k] += 1

    features: dict[int, dict] = {}
    for k in all_locs:
        base = _base_demand_features(
            k, n_del, n_pck, w_del, w_pck, mean_del_w, total_del_w)
        total_comp = float(sum(locker_cap.get(k, {}).values()))
        base.update({
            'total_compartments':   total_comp,
            'utilization_ratio':    n_del[k] / total_comp if total_comp > 0 else 0.0,
            'large_size_share':     n_large[k] / n_del[k] if n_del[k] > 0 else 0.0,
            'size_weighted_demand': size_sum[k],
        })
        features[k] = base
    return features


def saturation_label(features: dict[int, dict],
                     threshold_pct: float = 75.0) -> dict[int, int]:
    """
    Binary saturation label: 1 if node delivery weight >= threshold_pct
    percentile across all nodes in this instance.
    (Legacy / fallback — used when locker capacity is unavailable.)
    """
    weights   = [v['delivery_weight'] for v in features.values()]
    threshold = float(np.percentile(weights, threshold_pct)) if weights else 0.0
    return {k: int(v['delivery_weight'] >= threshold) for k, v in features.items()}


def saturation_label_capacity(orders: list[dict],
                               locker_cap: 'LockerCap',
                               percentile: float = 75.0) -> dict[int, int]:
    """
    Capacity-aware saturation label based on utilization percentile.

    Motivation: Rudy instances have ~79 compartments per locker (32+29+18
    default) but only ~1-2 deliveries per locker on average, so hard binary
    overflow never occurs. Instead, we use the locker's *utilization ratio*
    (deliveries / total compartments) as a continuous risk signal, and label
    as high-risk the top (100-percentile)% of lockers by utilization.

    Utilization is computed as:
        util(k) = n_deliveries_at_k / total_compartments_at_k

    This is capacity-aware (normalises by each locker's actual compartment
    count from the parsed instance file) while guaranteeing ~25% positive
    examples (top quartile by default).

    Lockers not present in locker_cap → utilization assumed 0 (low risk).

    Returns
    -------
    {location_id: 0|1}  keyed only for locations with at least one delivery.
    """
    del_count: dict[int, int] = {}
    for o in orders:
        if o['type'] == 'delivery':
            k = o['location']
            del_count[k] = del_count.get(k, 0) + 1

    if not del_count:
        return {}

    # Compute utilization ratio per locker
    utilizations: dict[int, float] = {}
    for k, n_del in del_count.items():
        cap = locker_cap.get(k)
        if cap:
            total_comp = sum(cap.values())   # small + medium + large
            utilizations[k] = n_del / total_comp if total_comp > 0 else 0.0
        else:
            utilizations[k] = 0.0   # unknown capacity → treat as low risk

    # Label 1 if utilization ≥ percentile threshold (top 25% by default)
    util_vals = list(utilizations.values())
    threshold = float(np.percentile(util_vals, percentile)) if util_vals else 0.0
    return {k: int(u >= threshold) for k, u in utilizations.items()}


def features_to_array(features: dict[int, dict],
                      columns: list[str] = FEATURE_COLS
                      ) -> tuple[list[int], np.ndarray]:
    """Convert feature dicts to (node_id_list, numpy_array) for sklearn.

    Pass columns=FEATURE_COLS_B when features was built with
    extract_features_capacity (Model B)."""
    node_ids = sorted(features.keys())
    X = np.array([[features[k][c] for c in columns] for k in node_ids],
                 dtype=float)
    return node_ids, X


def features_to_hourly_matrix(features: dict[int, dict]
                              ) -> tuple[list[int], np.ndarray]:
    """
    Expand capacity-aware features (FEATURE_COLS_B) into a (node_ids x 24,
    FEATURE_COLS_C) matrix: each node's 13 base columns repeated once per
    hour 0-23, with 'arrival_hour' varying as the 14th column.

    Used by heuristic_c.py to score P(saturation | hour) for every hour in
    ONE batched sklearn call at graph-build time, rather than one call per
    candidate merge (which would be far too slow -- see heuristic_c.py).

    Returns
    -------
    node_ids : list of node ids, length n (same order as rows repeat)
    X        : np.ndarray shape (n*24, 14); row block [k*24:(k+1)*24] is
               node_ids[k] at hours 0..23 in order.
    """
    node_ids, base = features_to_array(features, columns=FEATURE_COLS_B)
    n = len(node_ids)
    hours = np.arange(24, dtype=float)
    X_base = np.repeat(base, 24, axis=0)              # (n*24, 13)
    X_hour = np.tile(hours, n).reshape(-1, 1)          # (n*24, 1)
    X = np.hstack([X_base, X_hour])                    # (n*24, 14)
    return node_ids, X


# =============================================================================
# DATASET BUILDERS
# =============================================================================

def _load_from_file(path: str,
                    threshold_pct: float,
                    verbose: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one instance file -> (X_rows, y_rows). Returns None on error."""
    fname = os.path.basename(path)
    try:
        params, orders, locker_cap = read_orders_only(path)
        feats = extract_features(params, orders)

        # Use capacity-based label when locker_cap is available (preferred),
        # otherwise fall back to percentile-based label.
        if locker_cap:
            labels = saturation_label_capacity(orders, locker_cap)
        else:
            labels = saturation_label(feats, threshold_pct)

        _, Xi = features_to_array(feats)
        yi    = np.array([labels.get(k, 0) for k in sorted(feats.keys())], dtype=int)

        if verbose:
            n_sat     = int(yi.sum())
            label_src = 'cap' if locker_cap else 'pct'
            print(f"  {fname}: {params.n_nodes} nodes | "
                  f"{n_sat} saturated ({100*n_sat/len(yi):.1f}%, {label_src}) | "
                  f"cap={params.capacity:.0f}  orders={params.n_orders}")
        return Xi, yi
    except Exception as exc:
        if verbose:
            print(f"  SKIP {fname}: {exc}")
        return None


def load_dataset(instance_dir: str,
                 threshold_pct: float = 75.0,
                 verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load all .txt files in a directory."""
    files = sorted(
        os.path.join(instance_dir, f)
        for f in os.listdir(instance_dir)
        if f.endswith('.txt')
    )
    return load_dataset_from_files(files, threshold_pct, verbose)


def load_dataset_from_files(file_paths: list[str],
                             threshold_pct: float = 75.0,
                             verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a specific list of instance files -> (X, y).
    Used to train only on the training split of instances.
    """
    X_rows, y_rows = [], []
    for path in file_paths:
        result = _load_from_file(path, threshold_pct, verbose)
        if result is not None:
            X_rows.append(result[0])
            y_rows.append(result[1])
    if not X_rows:
        raise RuntimeError("No valid instances loaded.")
    return np.vstack(X_rows), np.concatenate(y_rows)
