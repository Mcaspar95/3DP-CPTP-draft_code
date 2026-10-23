# Simulated Annealing for the 3DP-CPTP — random single-neighbour variant
# =========================================================================
# Each SA step is a textbook SA step:
#
#     1. pick ONE move operator uniformly at random (relocate/swap/insert/remove/2-opt)
#     2. sample ONE random neighbour from that operator — partners are drawn from ALL
#        customers, NOT restricted to the p nearest ones
#     3. accept or reject that single neighbour by the Metropolis criterion
#     4. cool down once per iteration, regardless of the outcome
#
#   So there is no neighbour enumeration and no p_nearest candidate restriction: the
#   whole neighbourhood is reachable, which trades intensification for exploration.
#   Each iteration costs at most one packing check, so iterations are much cheaper.
#
#   - Improving moves are always accepted
#   - Worsening moves are accepted with probability exp(-Δ / T)
#   - Temperature T decreases geometrically: T *= alpha each iteration
#   - Reheating + Diversification is applied after prolonged stagnation (see parameter REHEAT_THRESHOLD)
#   - Function check_packing_feasible() has a time_limit, but this can enhance exploration as feasible solutions are still rejected
#
#
#
# As before, 5 neighborhood move types (same as tabu), each sampled uniformly at random:
#   Relocate — move a customer between routes
#   Swap     — exchange two customers between routes
#   Insert   — serve an unserved customer
#   Remove   — unserve an unprofitable customer
#   2-opt    — reverse a segment within a route
#
# 3D packing feasibility checked via Gurobi sub-model with caching

import math
import os
import random
import time
import copy
from itertools import permutations

from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from instances import cvrp01, cvrp02, cvrp03, cvrp04, cvrp05, cvrp06, cvrp07, cvrp08, cvrp09, cvrp10, cvrp11, cvrp12, cvrp13, cvrp14, cvrp15, cvrp16, cvrp17, cvrp18, cvrp19, cvrp20, cvrp21, cvrp22, cvrp23, cvrp24, cvrp25, cvrp26, cvrp27
from instances_cutted import ccvrp01, ccvrp02, ccvrp03, ccvrp04, ccvrp05
# -------------------------
# Parameters
# -------------------------
T_INIT = 100.0             # initial temperature
ALPHA = 0.9999             # geometric cooling factor (T *= ALPHA each iteration)
T_MIN = 0.01               # minimum temperature before stopping / reheating
MAX_ITERATIONS = 50000000    # maximum SA iterations
TIME_LIMIT = 1000          # total wall-clock seconds
PACKING_TIME_LIMIT = 5     # time limit per packing feasibility check (seconds)
GAMMA = 1.0                # travel cost factor
P_NEAREST = 1              # kept only so main_slurm.py can still set it; the random
                           # neighbourhood below deliberately IGNORES it (see MOVE_WEIGHTS)
# Temperature restored on reheating, as a fraction of T_INIT. A reheat fires only after
# prolonged stagnation, by which point T has annealed close to T_MIN — so the reheat has
# to RESET the temperature, not scale it: doubling 0.01 gives 0.02, which is still far
# too cold to accept anything. Resetting to T_0/4 makes worsening moves acceptable again
# and the schedule then anneals down as usual, giving repeated explore->intensify cycles.
REHEAT_FRACTION = 0.25
T_REHEAT = REHEAT_FRACTION * T_INIT

# Iterations for one full geometric sweep T_INIT -> T_MIN at the current ALPHA.
# ~92k for ALPHA=0.9999; ~900 for ALPHA=0.99.
COOLING_SWEEP_ITERS = math.log(T_MIN / T_INIT) / math.log(ALPHA)

# Reheat only after the schedule has had a real chance to cool, i.e. after a
# meaningful FRACTION of a full sweep without a new global best. Deriving it from
# ALPHA instead of hard-coding a number is what keeps annealing intact: an absolute
# threshold far below COOLING_SWEEP_ITERS resets T to T_INIT before it can ever cool,
# which pins acceptance near 100% and degrades SA into a random walk (measured: with
# REHEAT_THRESHOLD=5000 vs a 92k sweep the acceptance rate was 82% and every reheat
# printed T=100.0). Override REHEAT_THRESHOLD directly if you want to experiment.
REHEAT_THRESHOLD = max(500, int(0.75 * COOLING_SWEEP_ITERS))

# Relative probability of drawing each move operator per iteration. Set an entry to 0
# to disable that operator. These are weights, not probabilities — they are normalised.
MOVE_WEIGHTS = {
    "relocate": 1.0,
    "swap":     1.0,
    "insert":   1.0,
    "remove":   1.0,
    "2opt":     1.0,
}

# NOTE on ALPHA / REHEAT_THRESHOLD being different from 3dp-cptp-SA.py:
# One iteration here is a single sampled neighbour instead of a full enumerate-and-scan
# pass, so this variant performs orders of magnitude more iterations per second. With the
# original ALPHA=0.99 the temperature would collapse to T_MIN within ~900 iterations, i.e.
# almost immediately, and the run would degenerate into a pure hill-climber. ALPHA=0.9999
# stretches one cooling sweep over ~92k iterations. REHEAT_THRESHOLD is raised for the same
# reason. Both are worth tuning per instance size.

#rule of thumb (ALPHA = 0.990): T_INIT = 100 - 90% acceptance, T_INIT = 50 - 75% acceptance

# Why a random, non-p_nearest neighbourhood here:
# The p=1 restriction is a strong intensification device: it only ever proposes moves between
# a customer and its single nearest neighbour's route, so large parts of the search space are
# unreachable in one step. That is what makes the original variant efficient (few wasted
# packing checks) but also what can trap it — the caveats noted in 3dp-cptp-SA.py. Drawing the
# partner uniformly from all customers makes every neighbour reachable, so the chain can in
# principle traverse the whole space; the price is that many proposals are poor or
# packing-infeasible. Whether that trade pays off is exactly what this file is for.

# -------------------------
# Instance generation (shared with other solvers)
# -------------------------
def generate_instance(
    instance,
    gamma=1.0,
    scenario="P1",
):
    if hasattr(instance, "__name__"):
        name = instance.__name__.split(".")[-1]
    else:
        name = str(instance)
    m = instance.m
    coords = instance.coords
    n = instance.n
    C = instance.C
    Q = instance.Q
    items = instance.items
    weights = instance.weights

    if scenario == "P1":
        revenues = instance.revenues
    elif scenario == "P2":
        revenues = {i: 1 + ((7141 * i + 73) % 100) for i in range(1, n + 1)}
    elif scenario == "P3":
        max_dist = max(math.dist(coords[0], coords[j]) for j in range(1, n + 1))
        revenues = {i: 1 + math.ceil(99 * math.dist(coords[0], coords[i]) / max_dist) for i in range(1, n + 1)}
    elif scenario == "P4":
        revenues = {i: round(math.dist(coords[0], coords[i])) for i in range(1, n + 1)}
    else:
        raise ValueError(f"Unknown scenario {scenario}")

    vehicles = {k: C for k in range(1, m + 1)}

    c = {}
    N0 = list(range(0, n + 1))
    for i in N0:
        for j in N0:
            if i == j:
                continue
            c[(i, j)] = gamma * math.dist(coords[i], coords[j])

    return {
        "n": n,
        "m": m,
        "coords": coords,
        "vehicles": vehicles,
        "Q": Q,
        "items": items,
        "revenues": revenues,
        "weights": weights,
        "costs": c,
        "tightness": None,
        "name": name,
        "scenario": scenario,
        "p_nearest": _precompute_p_nearest(n, c, P_NEAREST),
    }

# -------------------------
# Helper: precompute p-nearest neighbors
# -------------------------
def _precompute_p_nearest(n, costs, p):
    N = list(range(1, n + 1))
    p_nearest = {}
    for i in N:
        others = [(costs.get((i, j), float('inf')), j) for j in N if j != i]
        others.sort()
        p_nearest[i] = set(j for _, j in others[:p])
    return p_nearest

# -------------------------
# Helper: unique orientations
# -------------------------
def unique_orientations(l, w, h):
    return list(dict.fromkeys(permutations([l, w, h], 3)))

# -------------------------
# 3D Packing feasibility check via Gurobi
# -------------------------
def check_packing_feasible(item_ids, instance, time_limit=None, deadline=None):
    # resolve at call time so main_slurm can override the module global
    if time_limit is None:
        time_limit = PACKING_TIME_LIMIT
    if not item_ids:
        return True
    start_time = time.time()
    if deadline is not None:
        remaining = deadline - time.time()
        if remaining <= 0:
            print(f"check_packing_feasible: items={len(item_ids)}, time_limit={time_limit:.3f}s, elapsed=0.000s, status=deadline_expired, feasible=False")
            return False
        time_limit = min(time_limit, remaining)

    items = instance["items"]
    C = list(instance["vehicles"].values())[0]
    L, W, H = C
    N = list(item_ids) #items to be tested
    R_i = {i: unique_orientations(*items[i]) for i in N}

    model = gp.Model("packing_check")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    p_x = model.addVars(N, lb=0.0, name="px")
    p_y = model.addVars(N, lb=0.0, name="py")
    p_z = model.addVars(N, lb=0.0, name="pz")

    o = {}
    for i in N:
        for ridx in range(len(R_i[i])):
            o[(i, ridx)] = model.addVar(vtype=GRB.BINARY, name=f"o[{i},{ridx}]")
    o = gp.tupledict(o)

    pairs_ordered = [(i, j) for i in N for j in N if i != j]
    a_x = model.addVars(pairs_ordered, vtype=GRB.BINARY, name="ax")
    a_y = model.addVars(pairs_ordered, vtype=GRB.BINARY, name="ay")
    a_z = model.addVars(pairs_ordered, vtype=GRB.BINARY, name="az")

    def size_expr(i, axis):
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        return gp.quicksum(R_i[i][ridx][idx] * o[(i, ridx)] for ridx in range(len(R_i[i])))

    model.setObjective(0, GRB.MINIMIZE)

    for i in N:
        model.addConstr(gp.quicksum(o[(i, ridx)] for ridx in range(len(R_i[i]))) == 1)

    for i in N:
        model.addConstr(p_x[i] + size_expr(i, "x") <= L)
        model.addConstr(p_y[i] + size_expr(i, "y") <= W)
        model.addConstr(p_z[i] + size_expr(i, "z") <= H)

    pairs = [(i, j) for i in N for j in N if i < j]
    for i, j in pairs:
        model.addConstr(
            a_x[(i, j)] + a_x[(j, i)]
            + a_y[(i, j)] + a_y[(j, i)]
            + a_z[(i, j)] + a_z[(j, i)]
            >= 1
        )

    for i, j in pairs_ordered:
        model.addConstr(p_x[i] + size_expr(i, "x") <= p_x[j] + L * (1 - a_x[(i, j)]))
        model.addConstr(p_y[i] + size_expr(i, "y") <= p_y[j] + W * (1 - a_y[(i, j)]))
        model.addConstr(p_z[i] + size_expr(i, "z") <= p_z[j] + H * (1 - a_z[(i, j)]))

    model.optimize()
    elapsed = time.time() - start_time
    result = model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)
    print(f"check_packing_feasible: items={len(item_ids)}, time_limit={time_limit:.3f}s, elapsed={elapsed:.3f}s, status={model.Status}, feasible={result}")

    if result:
        return True
    return False

# -------------------------
# 3D Packing solver — returns item positions & oriented dimensions
# -------------------------
def solve_packing(item_ids, instance, time_limit=None, deadline=None):
    # resolve at call time so main_slurm can override the module global
    if time_limit is None:
        time_limit = PACKING_TIME_LIMIT
    if not item_ids:
        return []
    if deadline is not None:
        remaining = deadline - time.time()
        if remaining <= 0:
            return []
        time_limit = min(time_limit, remaining)

    items = instance["items"]
    C = list(instance["vehicles"].values())[0]
    L, W, H = C
    N = list(item_ids)
    R_i = {i: unique_orientations(*items[i]) for i in N}

    model = gp.Model("packing_solve")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    p_x = model.addVars(N, lb=0.0, name="px")
    p_y = model.addVars(N, lb=0.0, name="py")
    p_z = model.addVars(N, lb=0.0, name="pz")

    o = {}
    for i in N:
        for ridx in range(len(R_i[i])):
            o[(i, ridx)] = model.addVar(vtype=GRB.BINARY, name=f"o[{i},{ridx}]")
    o = gp.tupledict(o)

    pairs_ordered = [(i, j) for i in N for j in N if i != j]
    a_x = model.addVars(pairs_ordered, vtype=GRB.BINARY, name="ax")
    a_y = model.addVars(pairs_ordered, vtype=GRB.BINARY, name="ay")
    a_z = model.addVars(pairs_ordered, vtype=GRB.BINARY, name="az")

    def size_expr(i, axis):
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        return gp.quicksum(R_i[i][ridx][idx] * o[(i, ridx)] for ridx in range(len(R_i[i])))

    model.setObjective(0, GRB.MINIMIZE)

    for i in N:
        model.addConstr(gp.quicksum(o[(i, ridx)] for ridx in range(len(R_i[i]))) == 1)

    for i in N:
        model.addConstr(p_x[i] + size_expr(i, "x") <= L)
        model.addConstr(p_y[i] + size_expr(i, "y") <= W)
        model.addConstr(p_z[i] + size_expr(i, "z") <= H)

    pairs = [(i, j) for i in N for j in N if i < j]
    for i, j in pairs:
        model.addConstr(
            a_x[(i, j)] + a_x[(j, i)]
            + a_y[(i, j)] + a_y[(j, i)]
            + a_z[(i, j)] + a_z[(j, i)]
            >= 1
        )

    for i, j in pairs_ordered:
        model.addConstr(p_x[i] + size_expr(i, "x") <= p_x[j] + L * (1 - a_x[(i, j)]))
        model.addConstr(p_y[i] + size_expr(i, "y") <= p_y[j] + W * (1 - a_y[(i, j)]))
        model.addConstr(p_z[i] + size_expr(i, "z") <= p_z[j] + H * (1 - a_z[(i, j)]))

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return []

    packed = []
    for i in N:
        chosen = None
        for ridx in range(len(R_i[i])):
            if o[(i, ridx)].X > 0.5:
                chosen = R_i[i][ridx]
                break
        if chosen is None:
            chosen = items[i]
        packed.append({
            "i": i,
            "pos": (p_x[i].X, p_y[i].X, p_z[i].X),
            "dims": chosen,
            "container": (L, W, H),
        })
    return packed

# Packing feasibility cache
_packing_cache = {}

def check_packing_cached(item_ids, instance, deadline=None):
    key = frozenset(item_ids)
    if key not in _packing_cache:
        _packing_cache[key] = check_packing_feasible(item_ids, instance, deadline=deadline)
    return _packing_cache[key]

# -------------------------
# Solution representation
# -------------------------
def route_cost(route, costs):
    total = 0.0
    for idx in range(len(route) - 1):
        total += costs.get((route[idx], route[idx + 1]), 0.0)
    return total

def route_weight(route, weights):
    return sum(weights.get(i, 0) for i in route if i != 0)

def route_items(route):
    return [i for i in route if i != 0]

def solution_objective(sol, instance):
    revenues = instance["revenues"]
    costs = instance["costs"]
    total_rev = sum(revenues.get(i, 0) for i in sol["served"])
    total_cost = 0.0
    for k, route in sol["routes"].items():
        if len(route) > 2:
            total_cost += route_cost(route, costs)
    return total_rev - total_cost

def copy_solution(sol):
    return {
        "routes": {k: list(route) for k, route in sol["routes"].items()},
        "served": set(sol["served"]),
        "unserved": set(sol["unserved"]),
    }

# -------------------------
# Initial solution: greedy savings heuristic
# -------------------------
def build_initial_solution(instance, deadline=None):
    n = instance["n"]
    m = instance["m"]
    Q = instance["Q"]
    costs = instance["costs"]
    revenues = instance["revenues"]
    weights = instance["weights"]
    K = list(range(1, m + 1))
    N = list(range(1, n + 1))

    attractiveness = []
    for i in N:
        dist_depot = costs.get((0, i), 1.0) + costs.get((i, 0), 1.0)
        ratio = revenues[i] / max(dist_depot, 0.01)
        attractiveness.append((ratio, i))
    attractiveness.sort(reverse=True)

    routes = {k: [0, 0] for k in K}
    served = set()
    unserved = set(N)

    for _, cust in attractiveness:
        best_k = None
        best_pos = None
        best_increase = float("inf")

        for k in K:
            curr_weight = route_weight(routes[k], weights)
            if curr_weight + weights[cust] > Q:
                continue
            r = routes[k]
            for pos in range(1, len(r)):
                c_insert = (costs.get((r[pos - 1], cust), 0)
                            + costs.get((cust, r[pos]), 0)
                            - costs.get((r[pos - 1], r[pos]), 0))
                if c_insert < best_increase:
                    trial_items = route_items(r) + [cust]
                    if check_packing_cached(trial_items, instance, deadline=deadline):
                        best_increase = c_insert
                        best_k = k
                        best_pos = pos

        if deadline is not None and time.time() >= deadline:
            break

        if best_k is not None:
            net = revenues[cust] - best_increase
            if net > 0:
                routes[best_k].insert(best_pos, cust)
                served.add(cust)
                unserved.discard(cust)

    return {
        "routes": routes,
        "served": served,
        "unserved": unserved,
    }

# -------------------------
# Cheapest insertion position
# -------------------------
def cheapest_insertion(cust, route, costs):
    best_pos = None
    best_delta = float("inf")
    for pos in range(1, len(route)):
        delta = (costs.get((route[pos - 1], cust), 0)
                 + costs.get((cust, route[pos]), 0)
                 - costs.get((route[pos - 1], route[pos]), 0))
        if delta < best_delta:
            best_delta = delta
            best_pos = pos
    return best_pos, best_delta

# -------------------------
# Remove customer from its route
# -------------------------
def remove_customer(cust, routes, costs):
    for k, route in routes.items():
        if cust in route:
            idx = route.index(cust)
            prev_node = route[idx - 1]
            next_node = route[idx + 1]
            savings = (costs.get((prev_node, cust), 0)
                       + costs.get((cust, next_node), 0)
                       - costs.get((prev_node, next_node), 0))
            route.remove(cust)
            return k, idx, savings
    return None, None, 0.0

# -------------------------
# Random single-move sampling (same 5 move types as Tabu Search)
# -------------------------
# Each sampler returns ONE randomly drawn move, or None when no valid move of that
# type exists in the current solution. Partners/targets are drawn uniformly from all
# customers/routes — the p_nearest restriction of 3dp-cptp-SA.py is deliberately NOT
# applied here. Every sampler still respects the weight capacity Q; packing
# feasibility is checked once afterwards, only for the move that SA accepts.
#
# Each sampler retries a few times because a random draw can hit an invalid
# combination (same route, capacity violation, ...). Retries are capped so a single
# iteration can never spin for long.

_SAMPLE_TRIES = 12


def _customer_to_route(sol):
    cust_to_route = {}
    for k, route in sol["routes"].items():
        for c in route_items(route):
            cust_to_route[c] = k
    return cust_to_route


def sample_relocate_move(sol, instance):
    """Move one random served customer into one random other route."""
    Q = instance["Q"]
    costs = instance["costs"]
    weights = instance["weights"]

    cust_to_route = _customer_to_route(sol)
    served = list(cust_to_route.keys())
    if not served:
        return None
    route_keys = list(sol["routes"].keys())
    if len(route_keys) < 2:
        return None

    # Try random customers, but for each one scan its possible target routes in random
    # order rather than guessing a single one. On tight instances (most routes near Q)
    # blind guessing would report "no move" even though a feasible relocate exists.
    random.shuffle(served)
    for cust in served[:_SAMPLE_TRIES]:
        k_from = cust_to_route[cust]
        targets = [k for k in route_keys if k != k_from]
        random.shuffle(targets)
        for k_to in targets:
            route_to = sol["routes"][k_to]
            if route_weight(route_to, weights) + weights[cust] > Q:
                continue
            pos, insert_cost = cheapest_insertion(cust, route_to, costs)
            if pos is None:
                continue

            route_from = sol["routes"][k_from]
            idx = route_from.index(cust)
            prev_n = route_from[idx - 1]
            next_n = route_from[idx + 1]
            remove_saving = (costs.get((prev_n, cust), 0)
                             + costs.get((cust, next_n), 0)
                             - costs.get((prev_n, next_n), 0))
            delta = insert_cost - remove_saving
            return {
                "type": "relocate",
                "cust": cust,
                "from_k": k_from,
                "to_k": k_to,
                "to_pos": pos,
                "delta_cost": delta,
                "delta_obj": -delta,
            }
    return None


def sample_swap_move(sol, instance):
    """Exchange two random served customers that sit in different routes."""
    Q = instance["Q"]
    costs = instance["costs"]
    weights = instance["weights"]

    cust_to_route = _customer_to_route(sol)
    served = list(cust_to_route.keys())
    if len(served) < 2:
        return None

    # Pick a random first customer, then scan candidate partners in random order. Drawing
    # both endpoints blindly would often keep hitting same-route pairs (which are invalid
    # for a cross-route swap) and give up while valid partners still existed.
    random.shuffle(served)
    for c1 in served[:_SAMPLE_TRIES]:
        k1 = cust_to_route[c1]
        partners = [c for c in served if cust_to_route[c] != k1]
        random.shuffle(partners)
        for c2 in partners[:_SAMPLE_TRIES]:
            k2 = cust_to_route[c2]

            r1 = sol["routes"][k1]
            r2 = sol["routes"][k2]
            w1_new = route_weight(r1, weights) - weights[c1] + weights[c2]
            w2_new = route_weight(r2, weights) - weights[c2] + weights[c1]
            if w1_new > Q or w2_new > Q:
                continue

            idx1 = r1.index(c1)
            idx2 = r2.index(c2)
            rem1 = (costs.get((r1[idx1 - 1], c1), 0) + costs.get((c1, r1[idx1 + 1]), 0)
                    - costs.get((r1[idx1 - 1], r1[idx1 + 1]), 0))
            rem2 = (costs.get((r2[idx2 - 1], c2), 0) + costs.get((c2, r2[idx2 + 1]), 0)
                    - costs.get((r2[idx2 - 1], r2[idx2 + 1]), 0))

            # idx1/idx2 are already known and each customer occurs exactly once,
            # so copy-and-delete-by-index is faster than filtering the whole route.
            r1_temp = list(r1)
            del r1_temp[idx1]
            r2_temp = list(r2)
            del r2_temp[idx2]
            pos2_in_r1, ins_c2_r1 = cheapest_insertion(c2, r1_temp, costs)
            pos1_in_r2, ins_c1_r2 = cheapest_insertion(c1, r2_temp, costs)
            if pos2_in_r1 is None or pos1_in_r2 is None:
                continue

            delta = ins_c2_r1 + ins_c1_r2 - rem1 - rem2
            return {
                "type": "swap",
                "cust1": c1,
                "cust2": c2,
                "k1": k1,
                "k2": k2,
                "delta_cost": delta,
                "delta_obj": -delta,
            }
    return None


def sample_insert_move(sol, instance):
    """Serve one random currently unserved customer in one random route."""
    Q = instance["Q"]
    costs = instance["costs"]
    weights = instance["weights"]
    revenues = instance["revenues"]

    unserved = list(sol["unserved"])
    if not unserved:
        return None
    route_keys = list(sol["routes"].keys())

    # Same rationale as sample_relocate_move: scan the target routes of a randomly
    # chosen customer in random order instead of guessing one, so an existing feasible
    # insertion is not missed on capacity-tight instances.
    random.shuffle(unserved)
    for cust in unserved[:_SAMPLE_TRIES]:
        targets = list(route_keys)
        random.shuffle(targets)
        for k in targets:
            route = sol["routes"][k]
            if route_weight(route, weights) + weights[cust] > Q:
                continue
            pos, insert_cost = cheapest_insertion(cust, route, costs)
            if pos is None:
                continue
            return {
                "type": "insert",
                "cust": cust,
                "to_k": k,
                "to_pos": pos,
                "delta_cost": insert_cost,
                "delta_obj": revenues[cust] - insert_cost,
            }
    return None


def sample_remove_move(sol, instance):
    """Unserve one random currently served customer."""
    costs = instance["costs"]
    revenues = instance["revenues"]

    cust_to_route = _customer_to_route(sol)
    served = list(cust_to_route.keys())
    if not served:
        return None

    cust = random.choice(served)
    k = cust_to_route[cust]
    route = sol["routes"][k]
    idx = route.index(cust)
    prev_n = route[idx - 1]
    next_n = route[idx + 1]
    remove_saving = (costs.get((prev_n, cust), 0)
                     + costs.get((cust, next_n), 0)
                     - costs.get((prev_n, next_n), 0))
    return {
        "type": "remove",
        "cust": cust,
        "from_k": k,
        "delta_cost": -remove_saving,
        "delta_obj": -revenues[cust] + remove_saving,
    }


def sample_2opt_move(sol, instance):
    """Reverse one random segment inside one random route.

    Unlike generate_2opt_moves() in 3dp-cptp-SA.py this does NOT filter for
    improving reversals — a worsening 2-opt may be accepted by the Metropolis
    criterion, which is the point of sampling rather than enumerating.
    """
    costs = instance["costs"]

    candidates = [k for k, route in sol["routes"].items() if len(route) >= 4]
    if not candidates:
        return None

    for _ in range(_SAMPLE_TRIES):
        k = random.choice(candidates)
        route = sol["routes"][k]
        i = random.randint(1, len(route) - 3)
        j = random.randint(i + 1, len(route) - 2)
        old_cost = (costs.get((route[i - 1], route[i]), 0)
                    + costs.get((route[j], route[j + 1]), 0))
        new_cost = (costs.get((route[i - 1], route[j]), 0)
                    + costs.get((route[i], route[j + 1]), 0))
        delta = new_cost - old_cost
        return {
            "type": "2opt",
            "k": k,
            "i": i,
            "j": j,
            "delta_cost": delta,
            "delta_obj": -delta,
        }
    return None


_SAMPLERS = {
    "relocate": sample_relocate_move,
    "swap":     sample_swap_move,
    "insert":   sample_insert_move,
    "remove":   sample_remove_move,
    "2opt":     sample_2opt_move,
}


def sample_random_move(sol, instance, move_weights=None):
    """Draw ONE random neighbour: pick a random operator, then a random move from it.

    Operators are drawn without replacement (by weight) so that if the chosen one
    cannot produce a move in the current solution, the next one is tried instead of
    wasting the whole iteration. Returns None only if no operator yields a move.
    """
    if move_weights is None:
        move_weights = MOVE_WEIGHTS

    pool = [(name, w) for name, w in move_weights.items() if w > 0 and name in _SAMPLERS]
    while pool:
        total = sum(w for _, w in pool)
        r = random.uniform(0.0, total)
        upto = 0.0
        pick = len(pool) - 1
        for idx, (_, w) in enumerate(pool):
            upto += w
            if r <= upto:
                pick = idx
                break
        name, _ = pool.pop(pick)
        move = _SAMPLERS[name](sol, instance)
        if move is not None:
            return move
    return None

# -------------------------
# Apply a move to a solution (in-place)
# -------------------------
def apply_move(sol, move, instance):
    costs = instance["costs"]

    if move["type"] == "relocate":
        cust = move["cust"]
        sol["routes"][move["from_k"]].remove(cust)
        pos, _ = cheapest_insertion(cust, sol["routes"][move["to_k"]], costs)
        sol["routes"][move["to_k"]].insert(pos, cust)

    elif move["type"] == "swap":
        c1, c2 = move["cust1"], move["cust2"]
        k1, k2 = move["k1"], move["k2"]
        sol["routes"][k1].remove(c1)
        sol["routes"][k2].remove(c2)
        pos2, _ = cheapest_insertion(c2, sol["routes"][k1], costs)
        sol["routes"][k1].insert(pos2, c2)
        pos1, _ = cheapest_insertion(c1, sol["routes"][k2], costs)
        sol["routes"][k2].insert(pos1, c1)

    elif move["type"] == "insert":
        cust = move["cust"]
        pos, _ = cheapest_insertion(cust, sol["routes"][move["to_k"]], costs)
        sol["routes"][move["to_k"]].insert(pos, cust)
        sol["served"].add(cust)
        sol["unserved"].discard(cust)

    elif move["type"] == "remove":
        cust = move["cust"]
        sol["routes"][move["from_k"]].remove(cust)
        sol["served"].discard(cust)
        sol["unserved"].add(cust)

    elif move["type"] == "2opt":
        k = move["k"]
        i, j = move["i"], move["j"]
        route = sol["routes"][k]
        route[i:j + 1] = route[i:j + 1][::-1]

# -------------------------
# Affected routes for packing check
# -------------------------
def _affected_routes(move):
    if move["type"] == "relocate":
        return {move["from_k"], move["to_k"]}
    elif move["type"] == "swap":
        return {move["k1"], move["k2"]}
    elif move["type"] == "insert":
        return {move["to_k"]}
    elif move["type"] == "remove":
        return {move["from_k"]}
    elif move["type"] == "2opt":
        return {move["k"]}
    return set()

# -------------------------
# Packing checks
# -------------------------
def check_route_packing(route, instance, deadline=None):
    items_on_route = route_items(route)
    if not items_on_route:
        return True
    return check_packing_cached(items_on_route, instance, deadline=deadline)

# -------------------------
# Diversification (same as tabu)
# -------------------------
def _diversify(sol, instance, deadline=None):
    costs = instance["costs"]
    weights = instance["weights"]
    revenues = instance["revenues"]
    Q = instance["Q"]
    K = list(sol["routes"].keys())

    served_list = list(sol["served"])
    if not served_list:
        return

    n_remove = max(1, len(served_list) // 3)
    to_remove = random.sample(served_list, min(n_remove, len(served_list)))

    for cust in to_remove:
        for k in K:
            if cust in sol["routes"][k]:
                sol["routes"][k].remove(cust)
                break
        sol["served"].discard(cust)
        sol["unserved"].add(cust)

    unserved_list = list(sol["unserved"])
    random.shuffle(unserved_list)
    for cust in unserved_list[:n_remove * 2]:
        if deadline is not None and time.time() >= deadline:
            break
        best_k = None
        best_pos = None
        best_delta = float("inf")
        for k in K:
            route = sol["routes"][k]
            if route_weight(route, weights) + weights[cust] > Q:
                continue
            pos, delta = cheapest_insertion(cust, route, costs)
            if pos is not None and delta < best_delta:
                trial_items = route_items(route) + [cust]
                if check_packing_cached(trial_items, instance, deadline=deadline):
                    best_delta = delta
                    best_k = k
                    best_pos = pos

        if best_k is not None and revenues[cust] - best_delta > 0:
            sol["routes"][best_k].insert(best_pos, cust)
            sol["served"].add(cust)
            sol["unserved"].discard(cust)

# -------------------------
# Simulated Annealing main loop
# -------------------------
def simulated_annealing(instance, max_iterations=MAX_ITERATIONS, time_limit=TIME_LIMIT,
                        t_init=T_INIT, alpha=ALPHA, t_min=T_MIN,
                        reheat_threshold=REHEAT_THRESHOLD, t_reheat=T_REHEAT,
                        verbose=True):
    """
    Simulated Annealing for the 3DP-CPTP — classic single-neighbour variant.

    Per iteration exactly ONE random neighbour is drawn (random operator + random
    move, partners taken from all customers rather than the p nearest) and then
    accepted or rejected. No candidate list is enumerated.

    Acceptance criterion:
      - Improving moves (delta_obj > 0): always accepted
      - Worsening moves: accepted with probability exp(delta_obj / T)
    Temperature schedule:
      - Geometric cooling: T *= alpha every iteration (also on rejection)
      - Reheating: T is reset to t_reheat after prolonged stagnation
    """
    t_start = time.time()
    deadline = t_start + time_limit

    # ---- Build initial solution ----
    if verbose:
        print("Building initial solution...")
    current = build_initial_solution(instance, deadline=deadline)
    current_obj = solution_objective(current, instance)

    best = copy_solution(current)
    best_obj = current_obj

    if verbose:
        n_served = len(current["served"])
        print(f"Initial solution: served={n_served}, obj={current_obj:.2f}")

    T = t_init
    no_improve_count = 0
    accepted_count = 0
    rejected_count = 0
    infeasible_count = 0   # drawn + accepted by Metropolis, but packing-infeasible
    no_move_count = 0      # no operator could produce a move at all
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        if time.time() - t_start > time_limit:
            if verbose:
                print(f"Time limit reached at iteration {iteration}.")
            break

        # ---- Draw ONE random neighbour (random operator + random move) ----
        move = sample_random_move(current, instance)

        if move is None:
            # Nothing can be done to this solution at all — shake it and retry.
            no_move_count += 1
            no_improve_count += 1
            if no_improve_count >= reheat_threshold:
                T = t_reheat
                _diversify(current, instance, deadline=deadline)
                current_obj = solution_objective(current, instance)
                no_improve_count = 0
                if verbose:
                    print(f"[iter {iteration}] Reheat + diversify. T={T:.4f}, obj={current_obj:.2f}")
            T = max(T * alpha, t_min)
            continue

        # ---- Metropolis acceptance on that single neighbour ----
        delta_obj = move["delta_obj"]
        if delta_obj > 0:
            accept = True  # improving move — always accept
        elif T > 1e-12:
            accept = random.random() < math.exp(delta_obj / T)
        else:
            accept = False

        if accept:
            # Only now pay for a packing check, and only for the affected routes.
            trial = copy_solution(current)
            apply_move(trial, move, instance)

            feasible = True
            for k in _affected_routes(move):
                if k in trial["routes"]:
                    if not check_route_packing(trial["routes"][k], instance, deadline=deadline):
                        feasible = False
                        break

            if feasible:
                current = trial
                current_obj += delta_obj
                accepted_count += 1

                if current_obj > best_obj + 1e-6:
                    best = copy_solution(current)
                    best_obj = current_obj
                    no_improve_count = 0
                    if verbose:
                        print(f"[iter {iteration}] *** NEW BEST: obj={best_obj:.2f}, "
                              f"served={len(best['served'])}, T={T:.4f}, "
                              f"elapsed={time.time()-t_start:.1f}s")
                else:
                    no_improve_count += 1
            else:
                infeasible_count += 1
                no_improve_count += 1
        else:
            rejected_count += 1
            no_improve_count += 1

        # ---- Cool down once per iteration, whatever happened ----
        T = max(T * alpha, t_min)

        # ---- Reheat + diversify on prolonged stagnation ----
        if no_improve_count >= reheat_threshold:
            T = t_reheat
            _diversify(current, instance, deadline=deadline)
            current_obj = solution_objective(current, instance)
            no_improve_count = 0
            if verbose:
                print(f"[iter {iteration}] Reheat + diversify. T={T:.4f}, obj={current_obj:.2f}")

        if time.time() >= deadline:
            if verbose:
                print(f"Time limit reached at iteration {iteration}.")
            break

    elapsed = time.time() - t_start
    if verbose:
        print(f"\nSimulated Annealing finished in {elapsed:.2f}s after "
              f"{iteration} iterations.")
        print(f"Best objective: {best_obj:.2f}")
        print(f"Served customers: {sorted(best['served'])}")
        print(f"Un-Served customers: {sorted(best['unserved'])}")
        print(f"Accepted moves: {accepted_count}, Rejected moves: {rejected_count}, "
              f"Packing-infeasible: {infeasible_count}, No-move draws: {no_move_count}")
        for k, route in best["routes"].items():
            if len(route) > 2:
                print(f"  Route {k}: {route}")

    return best, best_obj

# -------------------------
# Plotting utilities
# -------------------------
def _box_faces(x, y, z, dx, dy, dz):
    p000 = (x,     y,     z)
    p100 = (x+dx,  y,     z)
    p010 = (x,     y+dy,  z)
    p110 = (x+dx,  y+dy,  z)
    p001 = (x,     y,     z+dz)
    p101 = (x+dx,  y,     z+dz)
    p011 = (x,     y+dy,  z+dz)
    p111 = (x+dx,  y+dy,  z+dz)
    return [
        [p000, p100, p110, p010],
        [p001, p101, p111, p011],
        [p000, p100, p101, p001],
        [p010, p110, p111, p011],
        [p000, p010, p011, p001],
        [p100, p110, p111, p101],
    ]

def _item_color(i, cmap_name="tab20"):
    cmap = plt.get_cmap(cmap_name)
    return cmap((i - 1) % cmap.N)

def plot_all_tours_from_solution(instance, sol, out_path=None):
    coords = instance["coords"]
    plt.figure()
    xs = [coords[i][0] for i in coords]
    ys = [coords[i][1] for i in coords]
    plt.scatter(xs, ys, c="gray", s=20, zorder=1)
    for i in sol["served"]:
        plt.scatter(coords[i][0], coords[i][1], c="green", s=50, zorder=2)
    plt.scatter(coords[0][0], coords[0][1], c="red", s=100, marker="s", zorder=3)
    for i, (cx, cy) in coords.items():
        plt.text(cx, cy, str(i), fontsize=7)
    for k, route in sol["routes"].items():
        if len(route) > 2:
            rx = [coords[i][0] for i in route]
            ry = [coords[i][1] for i in route]
            plt.plot(rx, ry, label=f"truck {k}")
    plt.legend()
    plt.title("Simulated Annealing — Selected tours")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()

def plot_packing_for_vehicle(k, packed_items, container_dims, out_path=None):
    L, W, H = container_dims
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    edges = [
        ((0,0,0),(L,0,0)), ((L,0,0),(L,W,0)), ((L,W,0),(0,W,0)), ((0,W,0),(0,0,0)),
        ((0,0,H),(L,0,H)), ((L,0,H),(L,W,H)), ((L,W,H),(0,W,H)), ((0,W,H),(0,0,H)),
        ((0,0,0),(0,0,H)), ((L,0,0),(L,0,H)), ((L,W,0),(L,W,H)), ((0,W,0),(0,W,H))
    ]
    for (x1,y1,z1),(x2,y2,z2) in edges:
        ax.plot([x1,x2],[y1,y2],[z1,z2])
    for item in packed_items:
        i = item["i"]
        (px, py, pz) = item["pos"]
        (dx, dy, dz) = item["dims"]
        faces = _box_faces(px, py, pz, dx, dy, dz)
        color = _item_color(i)
        poly = Poly3DCollection(
            faces, facecolors=[color],
            edgecolors=[mcolors.to_rgba("black", 0.25)],
            linewidths=0.5, alpha=0.35
        )
        ax.add_collection3d(poly)
        cx, cy, cz = px + dx/2, py + dy/2, pz + dz/2
        ax.text(cx, cy, cz, str(i), size=9)
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    ax.set_box_aspect((L, W, H))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"3D packing - truck {k}")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()

# -------------------------
# main
# -------------------------
def main(instance_module=None, scenario="P3", timelimit=None):
    if timelimit is None:
        timelimit = TIME_LIMIT
    if instance_module is None:
        instance_module = cvrp05
    t_start = time.time()

    instance = generate_instance(
        instance=instance_module,
        gamma=GAMMA,
        scenario=scenario,
    )

    print("=" * 60)
    print("3DP-CPTP Simulated Annealing (random single-neighbour variant)")
    print(f"Instance: n={instance['n']}, m={instance['m']}, Q={instance['Q']}")
    print(f"Container: {list(instance['vehicles'].values())[0]}")
    print(f"SA params: T0={T_INIT}, alpha={ALPHA}, T_min={T_MIN}, "
          f"reheat_threshold={REHEAT_THRESHOLD}, t_reheat={T_REHEAT}")
    print(f"Move weights: {MOVE_WEIGHTS} (partners drawn from ALL customers, p_nearest unused)")
    print("=" * 60)

    best_sol, best_obj = simulated_annealing(
        instance,
        max_iterations=MAX_ITERATIONS,
        time_limit=timelimit,
        t_init=T_INIT,
        alpha=ALPHA,
        t_min=T_MIN,
        reheat_threshold=REHEAT_THRESHOLD,
        t_reheat=T_REHEAT,
        verbose=True,
    )

    elapsed = time.time() - t_start

    print(f"\nTotal running time: {elapsed:.2f} seconds")
    print(f"Final objective: {best_obj:.2f}")

    used_trucks = [k for k, route in best_sol["routes"].items() if len(route) > 2]
    print(f"Used trucks: {used_trucks}")
    print(f"Served customers ({len(best_sol['served'])}): {sorted(best_sol['served'])}")

    # Solve packing for each used truck
    packed_by_k = {}
    for k in used_trucks:
        items_on_route = route_items(best_sol["routes"][k])
        packed_by_k[k] = solve_packing(items_on_route, instance)

    for k in used_trucks:
        route = best_sol["routes"][k]
        print(f"Route truck {k}: {route}")
        total_weight = sum(instance["weights"][i] for i in route_items(route))
        print(f"Truck {k} weight utilization: {total_weight} / {instance['Q']}")

        packed = packed_by_k.get(k, [])
        print(f"\nPacked items for vehicle {k} (i, pos, dims):")
        for it in packed:
            print(it["i"], it["pos"], it["dims"])
        if packed:
            Lk, Wk, Hk = instance["vehicles"][k]
            cap = Lk * Wk * Hk
            vol = sum(it["dims"][0] * it["dims"][1] * it["dims"][2] for it in packed)
            print(f"Oriented-volume utilization: {round(vol / cap, 3)}")

    results_dir = Path(__file__).parent / "results"
    os.chdir(results_dir)

    # include the alpha cooling parameter in the filename so results_table.py
    # can discover SA parameter variants (e.g. a0.995 -> SA_Obj(a0.995))
    alpha_tag = f"a{ALPHA}"
    out_fname = f"SA-{instance['name']}-{instance['scenario']}-{timelimit}s-{alpha_tag}-v{REHEAT_THRESHOLD}-r{REHEAT_FRACTION}-t{PACKING_TIME_LIMIT}.lsg"
    with open(out_fname, 'w') as f:
        f.write(f"Runtime: {elapsed}\n")
        f.write(f"Objective: {best_obj}\n")
        f.write(f"Used vehicles: {used_trucks}\n")
        f.write(f"Served customers ({len(best_sol['served'])}): {sorted(best_sol['served'])}\n")

        Qk = instance["Q"]
        for k in used_trucks:
            route = best_sol["routes"][k]
            f.write(f"Route vehicle {k}: {route}\n")
            total_weight = sum(instance["weights"][i] for i in route_items(route))
            f.write(f"Vehicle {k} weight utilization: {total_weight} / {Qk}\n")
            packed = packed_by_k.get(k, [])
            f.write(f"Packed items for vehicle {k} (i, pos, dims):\n")
            for it in packed:
                f.write(f"{it['i']} {it['pos']} {it['dims']}\n")
            if packed:
                Lk, Wk, Hk = instance["vehicles"][k]
                cap = Lk * Wk * Hk
                vol = sum(it["dims"][0] * it["dims"][1] * it["dims"][2] for it in packed)
                f.write(f"Oriented-volume utilization: {round(vol / cap, 3)}\n")

if __name__ == "__main__":
    main()
