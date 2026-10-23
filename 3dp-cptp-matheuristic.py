# Matheuristic for the 3DP-CPTP, inspired by Caspar 2025 (A matheuristic for the split delivery capacitated profitable tour problem with incomplete service times, European Journal of Operational Research 2025)
# showed promising results for CPTP variant.

    # -------------------------
    # combines local search with exact MILP optimization. First, a feasible solution is built using fast heuristics
    # (greedy assignment and tightening rules). Then, at each iteration, a neighborhood is defined by fixing most
    # binary variables and leaving only a selected subset free. The restricted subproblem—much smaller than the full
    # MILP—is solved exactly with a MIP solver, guaranteeing optimality within that neighborhood. If an improvement is found,
    # the solution is updated and a new neighborhood is explored. This hybrid approach exploits the speed of heuristics and
    # the precision of MILP optimization, producing high-quality solutions within limited time.
    # -------------------------

import math
import os
import random
import time
from itertools import combinations, permutations

from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from instances import cvrp01, cvrp02, cvrp03, cvrp04, cvrp05, cvrp06, cvrp07, cvrp08, cvrp09, cvrp10, cvrp11, cvrp12, cvrp13, cvrp14, cvrp15, cvrp16, cvrp17, cvrp18, cvrp19, cvrp20, cvrp21, cvrp22, cvrp23, cvrp24, cvrp25, cvrp26, cvrp27
timelimit = 1800 #seconds for final run of MILP
iter_number = 25  #number of matheuristic iterations i
iter_timelimit = 60 #seconds per matheuristic iteration t
max_stagnation = 4  # consecutive non-improving iterations before increasing destroy fraction

# -------------------------
# Based on instance: 3l_cvrp01.txt from Gendreau et al. 2006
# -------------------------
def generate_instance(
    instance,
    gamma=1.0, # travel cost factor
    scenario="P1",
):
    # derive a short name for the instance (module basename if a module was passed)
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

    # Vehicles (homogeneous fleet)
    vehicles = {k: C for k in range(1, m + 1)}

    # Travel costs
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
    }

# -------------------------
# DFJ subtour elimination callback + subtour finder
# -------------------------
def subtour(edges, reallist, include_depot=False):
    """
    Given a gp.tuplelist of directed edges and a list of nodes 'reallist',
    find the shortest cycle among connected components. If include_depot=False then
    ignore cycles containing depot (node 0) when searching for the smallest subtour.
    Returns a list of nodes in the detected cycle (route). If no subtour found returns [].
    """
    unvisited = list(reallist)
    cycle = list(range(len(reallist) + 1))
    while unvisited:
        thiscycle = []
        neighbors = list(unvisited)
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            if current in unvisited:
                unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
        if include_depot:
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        else:
            if len(cycle) > len(thiscycle) and (0 not in thiscycle):
                cycle = thiscycle
    if len(cycle) == len(reallist) + 1:
        return []
    return cycle

def subtourelim(model, where):
    """
    Callback that adds DFJ lazy cuts when an integer solution is found.
    Expects model._x_keys, model._x_vars, model._x_dict, model._K to be attached.
    """
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._x_vars)
        valdict = {}
        for idx, key in enumerate(model._x_keys):
            valdict[key] = vals[idx]

        K = list(model._K)
        selected = {k: gp.tuplelist() for k in K}
        for (i, j, k) in model._x_keys:
            if valdict.get((i, j, k), 0.0) > 0.5:
                selected[k].append((i, j))

        for k in K:
            used_nodes = list({v for e in selected[k] for v in e})
            if len(used_nodes) == 0:
                continue
            tour = subtour(selected[k], used_nodes, include_depot=False)
            if tour and (len(tour) < len(used_nodes) or 0 not in tour):
                expr = gp.quicksum(
                    model._x_dict[(i, j, k)]
                    for i in tour for j in tour if i != j
                )
                model.cbLazy(expr <= len(tour) - 1)

# -------------------------
# helper: unique orientations
# -------------------------
def unique_orientations(l, w, h):
    """Unique permutations of (l,w,h)."""
    return list(dict.fromkeys(permutations([l, w, h], 3)))

# -------------------------
# Multi-truck model builder (WITHOUT MTZ)
# -------------------------
def build_multi_truck_model(instance, verbose=True, symmetry_breaking=True):
    n = instance["n"]
    m = instance["m"]
    coords = instance["coords"]
    vehicles = instance["vehicles"]
    items = instance["items"]
    r = instance["revenues"]
    weights = instance["weights"]
    c = instance["costs"]

    N = list(range(1, n + 1)) #items/customers
    N0 = [0] + N #including depot
    K = list(range(1, m + 1)) #vehicles
    Q = instance["Q"] #weight capacity per vehicle

    R_i = {i: unique_orientations(*items[i]) for i in N}

    Mx = max(vehicles[k][0] for k in K)
    My = max(vehicles[k][1] for k in K)
    Mz = max(vehicles[k][2] for k in K)

    model = gp.Model("PVRP_3D_multi_truck")
    if not verbose:
        model.Params.OutputFlag = 0

    # Variables
    x = model.addVars(
        [(i, j, k) for i in N0 for j in N0 if i != j for k in K],
        vtype=GRB.BINARY, name="x")
    y = model.addVars([(i, k) for i in N for k in K], vtype=GRB.BINARY, name="y")
    z = model.addVars(N, vtype=GRB.BINARY, name="z")
    v = model.addVars(K, vtype=GRB.BINARY, name="v")

    p_x = model.addVars([(i, k) for i in N for k in K], lb=0.0, name="p_x")
    p_y = model.addVars([(i, k) for i in N for k in K], lb=0.0, name="p_y")
    p_z = model.addVars([(i, k) for i in N for k in K], lb=0.0, name="p_z")

    o = {}
    for i in N:
        for k in K:
            for ridx in range(len(R_i[i])):
                o[(i, k, ridx)] = model.addVar(vtype=GRB.BINARY, name=f"o[{i},{k},{ridx}]")
    o = gp.tupledict(o)

    a_x = model.addVars([(i, j, k) for i in N for j in N if i != j for k in K],
                        vtype=GRB.BINARY, name="a_x")
    a_y = model.addVars([(i, j, k) for i in N for j in N if i != j for k in K],
                        vtype=GRB.BINARY, name="a_y")
    a_z = model.addVars([(i, j, k) for i in N for j in N if i != j for k in K],
                        vtype=GRB.BINARY, name="a_z")

    def size_expr(i, k, axis):
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        return gp.quicksum(R_i[i][ridx][idx] * o[(i, k, ridx)] for ridx in range(len(R_i[i])))

    model.setObjective(
        gp.quicksum(r[i] * z[i] for i in N)
        - gp.quicksum(c[(i, j)] * x[(i, j, k)] for i in N0 for j in N0 if i != j for k in K),
        GRB.MAXIMIZE
    )

    # Selection
    for i in N:
        model.addConstr(gp.quicksum(y[(i, k)] for k in K) == z[i], name=f"select[{i}]")

    # Routing
    for i in N:
        for k in K:
            model.addConstr(
                gp.quicksum(x[(i, j, k)] for j in N0 if j != i) == y[(i, k)],
                name=f"outflow[{i},{k}]"
            )
            model.addConstr(
                gp.quicksum(x[(j, i, k)] for j in N0 if j != i) == y[(i, k)],
                name=f"inflow[{i},{k}]"
            )
            model.addConstr(y[(i, k)] <= v[k], name=f"assign_active[{i},{k}]")

    for k in K:
        model.addConstr(gp.quicksum(x[(0, j, k)] for j in N) == v[k], name=f"depart[{k}]")
        model.addConstr(gp.quicksum(x[(i, 0, k)] for i in N) == v[k], name=f"return[{k}]")

    for k in K:
        for i in N0:
            for j in N0:
                if i == j:
                    continue
                if i != 0:
                    model.addConstr(x[(i, j, k)] <= y[(i, k)], name=f"link_i[{i},{j},{k}]")
                if j != 0:
                    model.addConstr(x[(i, j, k)] <= y[(j, k)], name=f"link_j[{i},{j},{k}]")

    # Capacity
    for k in K:
        model.addConstr(
            gp.quicksum(weights[i] * y[(i, k)] for i in N) <= Q,
            name=f"weight_cap[{k}]"
        )

    # Orientation
    for i in N:
        for k in K:
            model.addConstr(
                gp.quicksum(o[(i, k, ridx)] for ridx in range(len(R_i[i]))) == y[(i, k)],
                name=f"orient[{i},{k}]"
            )

    # Container bounds
    for i in N:
        for k in K:
            Lk, Wk, Hk = vehicles[k]
            model.addConstr(p_x[(i, k)] + size_expr(i, k, "x") <= Lk, name=f"bound_x[{i},{k}]")
            model.addConstr(p_y[(i, k)] + size_expr(i, k, "y") <= Wk, name=f"bound_y[{i},{k}]")
            model.addConstr(p_z[(i, k)] + size_expr(i, k, "z") <= Hk, name=f"bound_z[{i},{k}]")

    # Tight pos ub
    for i in N:
        for k in K:
            Lk, Wk, Hk = vehicles[k]
            model.addConstr(p_x[(i, k)] <= Lk - size_expr(i, k, "x"), name=f"pos_ub_x[{i},{k}]")
            model.addConstr(p_y[(i, k)] <= Wk - size_expr(i, k, "y"), name=f"pos_ub_y[{i},{k}]")
            model.addConstr(p_z[(i, k)] <= Hk - size_expr(i, k, "z"), name=f"pos_ub_z[{i},{k}]")

    # Non-overlap activation
    for k in K:
        for i in N:
            for j in N:
                if i == j:
                    continue
                model.addConstr(a_x[(i, j, k)] <= y[(i, k)], name=f"act_ax_i[{i},{j},{k}]")
                model.addConstr(a_x[(i, j, k)] <= y[(j, k)], name=f"act_ax_j[{i},{j},{k}]")
                model.addConstr(a_y[(i, j, k)] <= y[(i, k)], name=f"act_ay_i[{i},{j},{k}]")
                model.addConstr(a_y[(i, j, k)] <= y[(j, k)], name=f"act_ay_j[{i},{j},{k}]")
                model.addConstr(a_z[(i, j, k)] <= y[(i, k)], name=f"act_az_i[{i},{j},{k}]")
                model.addConstr(a_z[(i, j, k)] <= y[(j, k)], name=f"act_az_j[{i},{j},{k}]")

    pairs = [(i, j) for i in N for j in N if i < j]
    for k in K:
        for i, j in pairs:
            model.addConstr(a_x[(i, j, k)] + a_x[(j, i, k)] <= 1, name=f"excl_x[{i},{j},{k}]")
            model.addConstr(a_y[(i, j, k)] + a_y[(j, i, k)] <= 1, name=f"excl_y[{i},{j},{k}]")
            model.addConstr(a_z[(i, j, k)] + a_z[(j, i, k)] <= 1, name=f"excl_z[{i},{j},{k}]")
            model.addConstr(
                a_x[(i, j, k)] + a_x[(j, i, k)]
                + a_y[(i, j, k)] + a_y[(j, i, k)]
                + a_z[(i, j, k)] + a_z[(j, i, k)]
                >= y[(i, k)] + y[(j, k)] - 1,
                name=f"sep_axis[{i},{j},{k}]"
            )

    for k in K:
        for i in N:
            for j in N:
                if i == j:
                    continue
                model.addConstr(
                    p_x[(i, k)] + size_expr(i, k, "x")
                    <= p_x[(j, k)] + Mx * (1 - a_x[(i, j, k)]),
                    name=f"nox[{i},{j},{k}]"
                )
                model.addConstr(
                    p_y[(i, k)] + size_expr(i, k, "y")
                    <= p_y[(j, k)] + My * (1 - a_y[(i, j, k)]),
                    name=f"noy[{i},{j},{k}]"
                )
                model.addConstr(
                    p_z[(i, k)] + size_expr(i, k, "z")
                    <= p_z[(j, k)] + Mz * (1 - a_z[(i, j, k)]),
                    name=f"noz[{i},{j},{k}]"
                )

    if symmetry_breaking and m > 1:
        for k_idx in range(1, m):
            model.addConstr(v[k_idx] >= v[k_idx + 1], name=f"sym[{k_idx}]")

    model.update()

    model._x_keys = list(x.keys())
    model._x_vars = [x[key] for key in model._x_keys]
    model._x_dict = x
    model._K = K

    vars_dict = {
        "x": x, "y": y, "z": z, "v": v,
        "p_x": p_x, "p_y": p_y, "p_z": p_z,
        "o": o, "R_i": R_i
    }
    return model, vars_dict

# -------------------------
# Matheuristic (ruin & recreate / LNS) — optimized
# -------------------------
def matheuristic(instance, model, vars_dict,
                 iterations=40, destroy_fraction=0.35,
                 initial_time=8, sub_time=6, seed=0,
                 max_stagnation=max_stagnation,
                 verbose=True):
    """
    Optimised LNS matheuristic.  Key improvements over the naïve version:

      1. **Fix ALL variable classes** for non-destroyed customers (x, p, o, a)
         so the sub-MIP is genuinely small instead of only fixing z/y.
      2. **Multiple destroy strategies** cycled in round-robin:
         - random
         - worst-profit  (lowest revenue − travel cost per customer)
         - spatial cluster (pick a seed customer, destroy its geographic neighbours)
         - route-based   (free an entire vehicle's customers at once)
      3. **Also free nearby non-served customers** so the solver can add them.
      4. **Adaptive destroy size** — grows when stagnating, shrinks after success.
      5. **Reload best incumbent** after each non-improving iteration to avoid
         drifting away from the best solution.
    """
    random.seed(seed)

    n = instance["n"]
    m = instance["m"]
    N = list(range(1, n + 1))
    K = list(range(1, m + 1))
    coords = instance["coords"]

    # ---- helpers ----------------------------------------------------------
    model.Params.LazyConstraints = 1
    if not verbose:
        model.Params.OutputFlag = 0

    # precompute pairwise customer distances (for spatial destroy)
    _dist_cache = {}
    def cdist(i, j):
        if (i, j) not in _dist_cache:
            d = math.dist(coords[i], coords[j])
            _dist_cache[(i, j)] = d
            _dist_cache[(j, i)] = d
        return _dist_cache[(i, j)]

    # ---- initial solve ----------------------------------------------------
    model.Params.TimeLimit = initial_time
    model.optimize(subtourelim)

    best_obj = None
    if model.SolCount > 0:
        best_obj = model.ObjVal
        if verbose:
            print(f"[init] obj = {best_obj:.4f}")
    else:
        if verbose:
            print("[init] No feasible solution found in initial solve.")

    # Save original bounds once
    orig_bounds = {}
    for var in model.getVars():
        orig_bounds[var] = (var.LB, var.UB)

    # ---- value extraction helpers ----------------------------------------
    def get_incumbent_vals():
        zvals, yvals = {}, {}
        xvals, vvals = {}, {}
        for i in N:
            try:    zvals[i] = int(round(vars_dict["z"][i].X))
            except: zvals[i] = 0
            for k in K:
                try:    yvals[(i, k)] = int(round(vars_dict["y"][(i, k)].X))
                except: yvals[(i, k)] = 0
        for k in K:
            try:    vvals[k] = int(round(vars_dict["v"][k].X))
            except: vvals[k] = 0
        x = vars_dict["x"]
        for key in x:
            try:    xvals[key] = int(round(x[key].X))
            except: xvals[key] = 0
        return zvals, yvals, xvals, vvals

    def collect_full_solution_values():
        sol = {}
        for var in model.getVars():
            try:    sol[var] = float(var.X)
            except: sol[var] = 0.0
        return sol

    def set_mip_start(sol_vals):
        for var, val in sol_vals.items():
            var.start = float(val)

    best_solution_map = None
    if model.SolCount > 0:
        best_solution_map = collect_full_solution_values()

    # ---- destroy-strategy helpers ----------------------------------------
    def _destroy_random(served, non_served, sz):
        """Pure random destroy among served customers."""
        pool = served if served else N
        return set(random.sample(pool, min(sz, len(pool))))

    def _destroy_worst_profit(served, non_served, sz, zvals, yvals, xvals):
        """Remove customers with lowest effective profit (revenue − travel share)."""
        costs = instance["costs"]
        profits = []
        pool = served if served else N
        for i in pool:
            rev_i = instance["revenues"].get(i, 0)
            # approximate travel cost attributed to i: sum of arcs touching i
            tc = 0.0
            for k in K:
                if yvals.get((i, k), 0) < 1:
                    continue
                for j in range(0, n + 1):
                    if j == i:
                        continue
                    if xvals.get((i, j, k), 0) > 0.5:
                        tc += costs.get((i, j), 0)
                    if xvals.get((j, i, k), 0) > 0.5:
                        tc += costs.get((j, i), 0)
            profits.append((rev_i - tc, i))
        profits.sort()  # ascending: worst first
        return set(c for _, c in profits[:min(sz, len(profits))])

    def _destroy_spatial(served, non_served, sz):
        """Pick a random seed customer and destroy its closest neighbours."""
        pool = served if served else N
        seed_cust = random.choice(pool)
        dists = [(cdist(seed_cust, j), j) for j in pool if j != seed_cust]
        dists.sort()
        result = {seed_cust}
        for _, j in dists[:sz - 1]:
            result.add(j)
        return result

    def _destroy_route(served, non_served, sz, yvals):
        """Free all customers on one (or two) randomly chosen vehicle(s)."""
        # find vehicles with assigned customers
        veh_custs = {k: [] for k in K}
        for i in served:
            for k in K:
                if yvals.get((i, k), 0) > 0.5:
                    veh_custs[k].append(i)
        active = [k for k in K if veh_custs[k]]
        if not active:
            return _destroy_random(served, non_served, sz)
        random.shuffle(active)
        result = set()
        for k in active:
            result.update(veh_custs[k])
            if len(result) >= sz:
                break
        return result

    strategies = ["random", "worst_profit", "spatial", "route"]

    # ---- adaptive fraction ------------------------------------------------
    frac = destroy_fraction
    stagnation = 0

    # ---- main loop --------------------------------------------------------
    for it in range(iterations):
        zvals, yvals, xvals, vvals = get_incumbent_vals()
        served   = [i for i in N if zvals.get(i, 0) == 1]
        non_served = [i for i in N if zvals.get(i, 0) == 0]

        # adaptive destroy size
        base_pool = max(len(served), 1)
        destroy_size = max(2, int(base_pool * frac))

        # pick strategy (round-robin with some randomness)
        strat = strategies[it % len(strategies)]

        if strat == "random":
            destroy_set = _destroy_random(served, non_served, destroy_size)
        elif strat == "worst_profit":
            destroy_set = _destroy_worst_profit(served, non_served, destroy_size, zvals, yvals, xvals)
        elif strat == "spatial":
            destroy_set = _destroy_spatial(served, non_served, destroy_size)
        elif strat == "route":
            destroy_set = _destroy_route(served, non_served, destroy_size, yvals)
        else:
            destroy_set = _destroy_random(served, non_served, destroy_size)

        # also free nearby non-served customers (give solver chance to add them)
        extra_free = set()
        for i in destroy_set:
            for j in non_served:
                if j not in destroy_set and cdist(i, j) < 25:  # threshold
                    extra_free.add(j)
        # limit extras to avoid blowing up subproblem
        if len(extra_free) > destroy_size:
            extra_free = set(random.sample(list(extra_free), destroy_size))
        destroy_set = destroy_set | extra_free

        if verbose:
            print(f"[iter {it+1}/{iterations}] strategy={strat}, destroying {len(destroy_set)} "
                  f"customers (frac={frac:.2f})...")

        # ---- fix / free variables -----------------------------------------
        # For non-destroyed customers: fix z, y, x (routing arcs), and packing
        # For destroyed customers: free everything
        x_var = vars_dict["x"]
        o_var = vars_dict["o"]
        R_i   = vars_dict["R_i"]

        for i in N:
            if i in destroy_set:
                # free z, y
                vars_dict["z"][i].LB, vars_dict["z"][i].UB = 0.0, 1.0
                for k in K:
                    vars_dict["y"][(i, k)].LB, vars_dict["y"][(i, k)].UB = 0.0, 1.0
                    # free packing positions (restore original bounds)
                    for pvar_name in ("p_x", "p_y", "p_z"):
                        pv = vars_dict[pvar_name][(i, k)]
                        pv.LB, pv.UB = orig_bounds[pv]
                    # free orientation
                    for ridx in range(len(R_i[i])):
                        ov = o_var[(i, k, ridx)]
                        ov.LB, ov.UB = 0.0, 1.0
            else:
                # fix z
                zval = zvals.get(i, 0)
                vars_dict["z"][i].LB = float(zval)
                vars_dict["z"][i].UB = float(zval)
                for k in K:
                    # fix y
                    yv = yvals.get((i, k), 0)
                    vars_dict["y"][(i, k)].LB = float(yv)
                    vars_dict["y"][(i, k)].UB = float(yv)
                    # fix orientation
                    for ridx in range(len(R_i[i])):
                        ov = o_var[(i, k, ridx)]
                        try:    oval = int(round(ov.X))
                        except: oval = 0
                        ov.LB, ov.UB = float(oval), float(oval)
                    # fix packing positions (only for assigned items)
                    if yv > 0.5:
                        for pvar_name in ("p_x", "p_y", "p_z"):
                            pv = vars_dict[pvar_name][(i, k)]
                            try:    pval = pv.X
                            except: pval = 0.0
                            pv.LB, pv.UB = pval, pval
                    else:
                        for pvar_name in ("p_x", "p_y", "p_z"):
                            pv = vars_dict[pvar_name][(i, k)]
                            pv.LB, pv.UB = 0.0, 0.0

        # fix x arcs that don't touch any destroyed customer
        N0 = list(range(0, n + 1))
        for key in x_var:
            i_arc, j_arc, k_arc = key
            i_free = (i_arc in destroy_set) or (j_arc in destroy_set) or (i_arc == 0) or (j_arc == 0)
            if not i_free:
                xv = xvals.get(key, 0)
                x_var[key].LB, x_var[key].UB = float(xv), float(xv)
            else:
                x_var[key].LB, x_var[key].UB = 0.0, 1.0

        # fix v (vehicle used) — only free vehicles that serve a destroyed customer
        vehicles_with_destroyed = set()
        for i in destroy_set:
            for k in K:
                if yvals.get((i, k), 0) > 0.5:
                    vehicles_with_destroyed.add(k)
        # also free any vehicle not currently used (might be needed)
        for k in K:
            if k in vehicles_with_destroyed or vvals.get(k, 0) == 0:
                vars_dict["v"][k].LB, vars_dict["v"][k].UB = 0.0, 1.0
            else:
                vars_dict["v"][k].LB = float(vvals[k])
                vars_dict["v"][k].UB = float(vvals[k])

        # provide MIP start from best known
        if best_solution_map is not None:
            set_mip_start(best_solution_map)

        # ---- solve sub-problem --------------------------------------------
        model.Params.TimeLimit = sub_time
        model.optimize(subtourelim)

        # ---- check improvement --------------------------------------------
        improved = False
        if model.SolCount > 0:
            cur_obj = model.ObjVal
            if best_obj is None or cur_obj > best_obj + 1e-6:
                improved = True
                best_obj = cur_obj
                best_solution_map = collect_full_solution_values()
                stagnation = 0
                # shrink fraction back toward base on success
                frac = max(destroy_fraction, frac * 0.85)
                if verbose:
                    print(f"  -> IMPROVED: obj = {best_obj:.4f}")
            else:
                stagnation += 1
                if verbose:
                    print(f"  -> No improvement (obj = {cur_obj:.4f}, stagnation={stagnation}).")
        else:
            stagnation += 1
            if verbose:
                print(f"  -> No feasible solution in sub-solve (stagnation={stagnation}).")

        # ---- restore bounds -----------------------------------------------
        for var in model.getVars():
            lb, ub = orig_bounds[var]
            var.LB, var.UB = lb, ub

        # reload best incumbent as MIP start so we don't drift
        if best_solution_map is not None:
            set_mip_start(best_solution_map)

        # adaptive: grow fraction when stagnating
        if stagnation >= max_stagnation:
            frac = min(0.65, frac + 0.08)
            stagnation = 0
            if verbose:
                print(f"  [adapt] increased destroy fraction to {frac:.2f}")

    # ---- wrap up ----------------------------------------------------------
    if verbose:
        print("Matheuristic finished.")
        if best_obj is not None:
            print(f"Best objective found: {best_obj:.4f}")
        else:
            print("No feasible solution was ever found during matheuristic.")

    if best_solution_map is not None:
        for var, val in best_solution_map.items():
            try:
                var.setAttr("Start", float(val))
            except Exception:
                pass

    return best_obj, best_solution_map

# -------------------------
# extract packing + routes + plotting 
# -------------------------
def extract_packing_solution_by_vehicle(instance, vars_dict):
    n = instance["n"]
    m = instance["m"]
    vehicles = instance["vehicles"]
    items = instance["items"]

    N = list(range(1, n + 1))
    K = list(range(1, m + 1))

    y = vars_dict["y"]
    p_x = vars_dict["p_x"]
    p_y = vars_dict["p_y"]
    p_z = vars_dict["p_z"]
    o = vars_dict["o"]
    R_i = vars_dict["R_i"]

    packed_by_k = {k: [] for k in K}

    for i in N:
        for k in K:
            if y[(i, k)].X > 0.5:
                chosen = None
                for ridx in range(len(R_i[i])):
                    if o[(i, k, ridx)].X > 0.5:
                        chosen = R_i[i][ridx]
                        break
                if chosen is None:
                    chosen = items[i]

                packed_by_k[k].append({
                    "i": i,
                    "pos": (p_x[(i, k)].X, p_y[(i, k)].X, p_z[(i, k)].X),
                    "dims": chosen,
                    "container": vehicles[k]
                })

    return packed_by_k

def extract_routes_by_vehicle(instance, vars_dict):
    n = instance["n"]
    m = instance["m"]
    N0 = list(range(0, n + 1))
    K = list(range(1, m + 1))

    x = vars_dict["x"]
    v = vars_dict["v"]

    routes = {k: [] for k in K}

    for k in K:
        if v[k].X < 0.5:
            continue

        arcs = [(i, j) for i in N0 for j in N0 if i != j and x[(i, j, k)].X > 0.5]
        if not arcs:
            continue

        succ = {}
        for i, j in arcs:
            succ[i] = j

        route = [0]
        cur = 0
        seen = set([0])
        while cur in succ:
            nxt = succ[cur]
            route.append(nxt)
            cur = nxt
            if cur in seen:
                break
            seen.add(cur)
            if cur == 0:
                break

        routes[k] = route

    return routes

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

def plot_all_tours(instance, routes_by_k, out_path=None):
    coords = instance["coords"]

    plt.figure()
    xs = [coords[i][0] for i in coords]
    ys = [coords[i][1] for i in coords]
    plt.scatter(xs, ys)

    for i, (x, y) in coords.items():
        plt.text(x, y, str(i), fontsize=9)

    for k, route in routes_by_k.items():
        if route and len(route) >= 2:
            rx = [coords[i][0] for i in route]
            ry = [coords[i][1] for i in route]
            plt.plot(rx, ry, label=f"truck {k}")

    if any(routes_by_k[k] for k in routes_by_k):
        plt.legend()

    plt.title("Selected tours (multi-truck)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    if out_path:
        dir_name = os.path.dirname(out_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()

# -------------------------
# UPDATED: Color helper for Matplotlib
# -------------------------
def _item_color(i, cmap_name="tab20"):
    """
    Stable color per item id using a matplotlib categorical colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    return cmap((i - 1) % cmap.N)


# -------------------------
# UPDATED: Matplotlib 3D packing with different colors per box
# -------------------------
def plot_packing_for_vehicle(k, packed_items, container_dims, out_path=None):
    L, W, H = container_dims

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Container edges
    edges = [
        ((0,0,0),(L,0,0)), ((L,0,0),(L,W,0)), ((L,W,0),(0,W,0)), ((0,W,0),(0,0,0)),
        ((0,0,H),(L,0,H)), ((L,0,H),(L,W,H)), ((L,W,H),(0,W,H)), ((0,W,H),(0,0,H)),
        ((0,0,0),(0,0,H)), ((L,0,0),(L,0,H)), ((L,W,0),(L,W,H)), ((0,W,0),(0,W,H))
    ]
    for (x1,y1,z1),(x2,y2,z2) in edges:
        ax.plot([x1,x2],[y1,y2],[z1,z2])

    # Items with distinct colors
    for item in packed_items:
        i = item["i"]
        (x, y, z) = item["pos"]
        (dx, dy, dz) = item["dims"]

        faces = _box_faces(x, y, z, dx, dy, dz)

        color = _item_color(i)
        poly = Poly3DCollection(
            faces,
            facecolors=[color],
            edgecolors=[mcolors.to_rgba("black", 0.25)],
            linewidths=0.5,
            alpha=0.35
        )
        ax.add_collection3d(poly)

        cx, cy, cz = x + dx/2, y + dy/2, z + dz/2
        ax.text(cx, cy, cz, str(i), size=9)

    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    ax.set_box_aspect((L, W, H))  # preserve real container proportions
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"3D packing - truck {k}")

    if out_path:
        dir_name = os.path.dirname(out_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()


def volume_utilization_for_vehicle(instance, packed_items):
    if not packed_items:
        return 0.0
    L, W, H = packed_items[0]["container"]
    cap = L * W * H
    vol = sum(it["dims"][0] * it["dims"][1] * it["dims"][2] for it in packed_items)
    return vol / cap

# -------------------------
# main: build model and run matheuristic
# -------------------------
def main(instance_module=None, scenario="P3", timelimit=None, max_stagnation=None):
    if timelimit is None:
        timelimit = globals().get('timelimit', 1800)
    if max_stagnation is None:
        max_stagnation = globals()["max_stagnation"]
    if instance_module is None:
        instance_module = cvrp01
    t_start = time.time()

    instance = generate_instance(
        instance=instance_module,
        gamma=1.0, # travel cost factor
        scenario=scenario,
    )

    model, vars_dict = build_multi_truck_model(instance, verbose=True, symmetry_breaking=True)

    # Run matheuristic
    best_obj, best_map = matheuristic(
        instance, model, vars_dict,
        iterations=iter_number, # number of destroy & reopt iterations
        destroy_fraction=0.25, # portion of customers to destroy per iteration
        initial_time=iter_timelimit, #For the first solve (solution used for the iterations)
        sub_time=iter_timelimit, #Time limit for each subproblem solve
        seed=123,
        max_stagnation=max_stagnation,
        verbose=True
    )

    # After matheuristic we can do a final polishing solve (optional)
    if best_map is not None:
        # set MIP start to best found and run a short polishing solve
        for var, val in best_map.items():
            var.start = float(val)
        model.Params.TimeLimit = timelimit
        model.optimize(subtourelim)

    status = model.Status
    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        elapsed = time.time() - t_start
        print(f"\nTotal matheuristic running time: {elapsed:.2f} seconds")
        print("\nObjective:", model.ObjVal)

        used_trucks = [k for k in range(1, instance["m"] + 1) if vars_dict["v"][k].X > 0.5]
        print("Used trucks:", used_trucks)

        served = [i for i in range(1, instance["n"] + 1) if vars_dict["z"][i].X > 0.5]
        non_served = [i for i in range(1, instance["n"] + 1) if vars_dict["z"][i].X < 0.5]
        print("Served customers:", served)

        routes_by_k = extract_routes_by_vehicle(instance, vars_dict)
        for k in range(1, instance["m"] + 1):
            if routes_by_k.get(k):
                print(f"Route truck {k}:", routes_by_k[k])

        #plot_all_tours(instance, routes_by_k)

        for k in used_trucks:
            total_weight = sum(
                instance["weights"][i]
                for i in range(1, instance["n"] + 1)
                if vars_dict["y"][(i, k)].X > 0.5
            )
            Qk = instance["Q"]
            print(f"Truck {k} weight utilization: {total_weight} / {Qk}")

                # Packing per truck
        packed_by_k = extract_packing_solution_by_vehicle(instance, vars_dict)

        for k in used_trucks:
            packed = packed_by_k.get(k, [])
            print(f"\nPacked items for truck {k} (i, pos, dims):")
            for it in packed:
                print(it["i"], it["pos"], it["dims"])

            if packed:
                Lk, Wk, Hk = instance["vehicles"][k]
                cap = Lk * Wk * Hk
                vol = sum(it["dims"][0] * it["dims"][1] * it["dims"][2] for it in packed)
                print("Oriented-volume utilization:", round(vol / cap, 3))

            # Matplotlib colored plot
            #plot_packing_for_vehicle(k, packed, instance["vehicles"][k])

    else:
        print("No feasible/usable solution. Status code:", status)

    results_dir = Path(__file__).parent / "results"
    os.chdir(results_dir)
    with open('M-' + str(instance["name"]) + '-' + str(instance["scenario"]) + '-' + str(timelimit) + 's-' + 'i' + str(iter_number) + '-' + 't' + str(iter_timelimit) + '-' + 'd' + str(max_stagnation) + '-' + '.lsg', 'w') as f:
        f.write(f"GAP: {model.MIPGap}\n")
        f.write(f"Runtime: {model.Runtime}\n")
        f.write(f"Objective: {model.ObjVal}\n")
        f.write(f"Used vehicles: {used_trucks}\n")
        f.write(f"Served customers: {served}\n")
        f.write(f"Non-Served customers: {non_served}\n")
        for k in range(1, instance["m"] + 1):
            if routes_by_k.get(k):
                f.write(f"Route vehicle {k}: {routes_by_k[k]}\n")
        Qk = instance["Q"]
        for k in used_trucks:
            total_weight = sum(
                instance["weights"][i]
                for i in range(1, instance["n"] + 1)
                if vars_dict["y"][(i, k)].X > 0.5
            )
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
