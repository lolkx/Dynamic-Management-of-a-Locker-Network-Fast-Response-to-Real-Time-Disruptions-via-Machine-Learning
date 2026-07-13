"""
generate_instances_B.py
------------------------
Generates high-saturation synthetic instances used as a training/test data
pool for Option C/D (data/instances_B/), complementing the low-occupancy
real instances in data/instances/.

Ports Instance::createRandom's actual random-generation ALGORITHM from
code/instance.h (weight-bucket distribution, per-compartment occupancy
Bernoulli trials, delivery/pickup order generation with pickup compartment
consumption) faithfully into Python. instance.h itself cannot be compiled in
this repo: it #includes multicrit.h (absent anywhere in the checkout), reads
distance matrices from matrix/{city}.csv (absent), and there's no main.cpp
entry point -- so "reusing exactly the code" means porting the algorithm
that doesn't depend on that missing external data, not fabricating those
missing pieces from scratch.

Two adaptations from instance.h, both required to interoperate with this
codebase (see instance_reader.py):
  - Compartment/order sizes are RE-INDEXED from instance.h's 0,1,2 to this
    codebase's 1,2,3 convention (matching simulator.py's _use_compartment/
    _find_fallback and saturation_label_overflow's [3,2,1] check order).
  - Distance matrices are SYNTHESISED (random 2D points, Euclidean distance)
    instead of read from matrix/{city}.csv, since instance_reader.py's
    read_orders_only path never touches the distance matrix at all --
    donor-reuse or exact geographic fidelity buys nothing for label training,
    and a synthetic matrix is still usable for routing experiments
    (run_experiments_C.py / run_experiments_D.py) since those DO need it.

Tuning note (why occupancy_probability alone isn't enough): saturation_label_
overflow checks size-3 demand against size-3 supply first (scarcest bucket,
maxC=18 per locker) -- the binding constraint. Order DENSITY (orders per
locker) matters more than occupancy_probability for actually triggering
overflow. CLI defaults (occupancy_probability 0.70-0.80, orders_per_locker
5-9) were empirically tuned via --report: an initial guess of 0.80-0.90 /
12-15 produced 73-88% overflow (too saturated, label nearly degenerate the
other way); this range gives a healthy, diverse 0-47% spread (mean ~15%)
across the combo grid, with the natural quantisation noise at small m
(overflow_rate granularity is 1/m) adding useful difficulty variety across
instances, similar in spirit to Model A training on real instances of
varying difficulty.

Usage
-----
    python generate_instances_B.py
    python generate_instances_B.py --m-values 15,30,50 --seeds 1 --report
"""

from __future__ import annotations
import os
import sys
import argparse
import random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ml_common import saturation_label_overflow
from instance_reader import MAX_COMPARTMENTS

# =============================================================================
# CONFIGURATION
# =============================================================================
_HERE   = os.path.dirname(__file__)
OUT_DIR = os.path.join(_HERE, '..', 'data', 'instances_B')


# =============================================================================
# WEIGHT DISTRIBUTION  (direct port of instance.h:54-64, 152-162)
# =============================================================================

def _weight_thresholds() -> list[float]:
    thresholds: list[float] = []
    threshold = 0.0
    factor = 1.1
    current = 1.0
    for _ in range(25):
        threshold += current
        thresholds.append(threshold)
        current /= factor
    return thresholds


def _draw_weight(rng: random.Random, thresholds: list[float]) -> int:
    draw = rng.uniform(0.0, thresholds[-1])
    for j, t in enumerate(thresholds):
        if draw < t:
            return j + 1
    return len(thresholds)


# =============================================================================
# SYNTHETIC DISTANCE MATRIX  (stand-in for instance.h's matrix/{city}.csv read)
# =============================================================================

def _random_distance_matrix(rng: random.Random, m: int, area_km: float = 20.0) -> np.ndarray:
    """m+1 random 2D points (depot = point 0) in an area_km x area_km box;
    pairwise Euclidean distance in metres, rounded to int."""
    pts = [(rng.uniform(0.0, area_km), rng.uniform(0.0, area_km)) for _ in range(m + 1)]
    mat = np.zeros((m + 1, m + 1), dtype=float)
    for i in range(m + 1):
        xi, yi = pts[i]
        for j in range(m + 1):
            if i == j:
                continue
            xj, yj = pts[j]
            mat[i, j] = round(((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5 * 1000.0)
    return mat


# =============================================================================
# ORDERS + LOCKER CAPACITY  (port of instance.h:126-191, re-indexed to 1,2,3)
# =============================================================================

def generate_orders_and_lockers(rng: random.Random, m: int, n_orders: int,
                                pickup_to_delivery_factor: float,
                                occupancy_probability: float,
                                max_pickup_attempts: int = 1000
                                ) -> tuple[list[dict], dict[int, dict[int, int]], int]:
    """
    Port of Instance::createRandom's locker-occupancy + order-generation loop
    (instance.h:126-191). Returns (orders, locker_cap, n_dropped_pickups).

    Locker compartments: each of MAX_COMPARTMENTS[size] slots per locker is
    free with probability (1 - occupancy_probability) -- instance.h:136-144.

    Orders: delivery orders generated first (random location, size just
    recorded), then pickup orders (instance.h:167-185) -- a pickup order
    immediately consumes a compartment at its drawn location, upgrading to a
    larger size if the drawn size isn't free there (for s in range(size, 4),
    matching simulator.py's _use_compartment), retried at a new random
    location up to max_pickup_attempts times before being dropped (added
    safety guard -- instance.h's original while loop has no bound and can
    spin forever once occupancy is high enough that some draws never find a
    free compartment).
    """
    thresholds = _weight_thresholds()

    locker_cap: dict[int, dict[int, int]] = {}
    for k in range(1, m + 1):
        counts = {1: 0, 2: 0, 3: 0}
        for size, max_n in MAX_COMPARTMENTS.items():
            for _ in range(max_n):
                if rng.random() >= occupancy_probability:
                    counts[size] += 1
        locker_cap[k] = counts

    n_pickup   = round(n_orders * pickup_to_delivery_factor)
    n_delivery = n_orders - n_pickup

    orders: list[dict] = []
    n_dropped = 0

    for i in range(n_orders):
        size   = rng.randint(1, 3)
        weight = _draw_weight(rng, thresholds)

        if i < n_delivery:
            location = rng.randint(1, m)
            orders.append({'type': 'delivery', 'weight': float(weight),
                          'location': location, 'size': size})
        else:
            placed = False
            for _attempt in range(max_pickup_attempts):
                location = rng.randint(1, m)
                for s in range(size, 4):
                    if locker_cap[location][s] > 0:
                        locker_cap[location][s] -= 1
                        size = s
                        placed = True
                        break
                if placed:
                    break
            if placed:
                orders.append({'type': 'pickup', 'weight': float(weight),
                              'location': location, 'size': size})
            else:
                n_dropped += 1

    return orders, locker_cap, n_dropped


# =============================================================================
# WRITER  (instance_reader.py's exact expected format)
# =============================================================================

def write_instance_b(path: str, n_orders: int, m: int, P: float, S: float,
                     C: float, B: float, orders: list[dict],
                     dist_matrix: np.ndarray, locker_cap: dict[int, dict[int, int]]
                     ) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(f"{n_orders} {m} 3\n")
        fh.write(f"{P:.0f} {S:.0f} {C:.0f} {B:.2f}\n")
        for o in orders:
            type_int = 0 if o['type'] == 'delivery' else 1
            fh.write(f"{type_int} {o['weight']:.0f} {o['location']} {o['size']}\n")
        for row in dist_matrix:
            fh.write(' '.join(f"{v:.0f}" for v in row) + '\n')
        for k in range(1, m + 1):
            cap = locker_cap.get(k, {1: 0, 2: 0, 3: 0})
            fh.write(f"{k} {cap[1]} {cap[2]} {cap[3]}\n")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description='Generate high-saturation synthetic instances for Model B'
    )
    p.add_argument('--m-values', default='15,30,50,100,200,400',
                   help='Comma-separated node counts to generate')
    p.add_argument('--pickup-factors', default='0.0,0.2,0.4',
                   help='Comma-separated pickupToDeliveryFactor values')
    p.add_argument('--occupancy-probs', default='0.70,0.75,0.80',
                   help='Comma-separated occupancyProbability values (high = saturated)')
    p.add_argument('--orders-per-locker', default='5,7,9',
                   help='Comma-separated n_orders/m ratios (density -- see module docstring)')
    p.add_argument('--seeds', type=int, default=2,
                   help='Number of random seeds per (m, pickup, occupancy, density) combo')
    p.add_argument('--P', type=float, default=30.0, help='Parking/depart time (s)')
    p.add_argument('--S', type=float, default=60.0, help='Service time per order (s)')
    p.add_argument('--C', type=float, default=700.0, help='Vehicle capacity (kg)')
    p.add_argument('--B', type=float, default=9.0, help='Departure hour')
    p.add_argument('--area-km', type=float, default=20.0, help='Synthetic city bounding box side (km)')
    p.add_argument('--out', default=OUT_DIR)
    p.add_argument('--report', action='store_true',
                   help='Print per-file overflow rate (saturation_label_overflow)')
    args = p.parse_args()

    m_values        = [int(x) for x in args.m_values.split(',')]
    pickup_factors  = [float(x) for x in args.pickup_factors.split(',')]
    occupancy_probs = [float(x) for x in args.occupancy_probs.split(',')]
    densities       = [float(x) for x in args.orders_per_locker.split(',')]

    os.makedirs(args.out, exist_ok=True)

    total_written  = 0
    total_dropped  = 0
    overflow_rates: list[float] = []

    combos = [(m, pf, op, dens)
             for m in m_values for pf in pickup_factors
             for op in occupancy_probs for dens in densities]

    print(f"Generating {len(combos) * args.seeds} instances -> {args.out}")

    for m, pickup_factor, occ_prob, density in combos:
        for seed_idx in range(args.seeds):
            seed = hash((m, pickup_factor, occ_prob, density, seed_idx)) & 0xFFFFFFFF
            rng  = random.Random(seed)

            n_orders = max(1, round(m * density))
            orders, locker_cap, n_dropped = generate_orders_and_lockers(
                rng, m, n_orders, pickup_factor, occ_prob)
            dist_matrix = _random_distance_matrix(rng, m, args.area_km)

            fname = (f"B_m{m}_pf{pickup_factor}_occ{occ_prob}_"
                    f"d{density}_s{seed_idx}.txt")
            path = os.path.join(args.out, fname)
            write_instance_b(path, len(orders), m, args.P, args.S, args.C, args.B,
                            orders, dist_matrix, locker_cap)

            total_written += 1
            total_dropped  += n_dropped

            if args.report:
                labels = saturation_label_overflow(orders, locker_cap)
                rate = (sum(labels.values()) / len(labels)) if labels else 0.0
                overflow_rates.append(rate)
                print(f"  {fname:<45} orders={len(orders):>5}  "
                      f"overflow_rate={rate:.1%}  dropped_pickups={n_dropped}")

    print(f"\nDone: {total_written} instances written, {total_dropped} pickup "
          f"orders dropped (max-attempts guard) across all files.")
    if overflow_rates:
        print(f"Mean overflow rate: {sum(overflow_rates)/len(overflow_rates):.1%}  "
              f"(min={min(overflow_rates):.1%}, max={max(overflow_rates):.1%})")


if __name__ == '__main__':
    main()
