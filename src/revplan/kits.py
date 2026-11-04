"""Unit-level (kit) formulation of reverse planning and per-unit dispatch of assembly assignments.

The aggregate model in model.py limits assembly use per model in total (constraints (4)-(6)). In general this is
a relaxation: three assemblies AB, AC and BC can be counted against two products that each need A, B and C once,
although no single product can take two of them (see tests). The kit formulation below is exact in general.

A kit of model j is a set K of fitting assemblies that can be installed together in ONE unit:
    sum_{y in K} b_iy <= a_ij  for every component i.
z_jK counts the units of model j built with exactly kit K. Then x_j = sum_K z_jK and w_jy = sum_{K contains y} z_jK.

Proposition (used for the case study). If, for every model j, the set of all fitting assemblies F_j is itself a
kit, every subset of F_j is a kit, and any (x, w) with w_jy <= x_j can be dispatched unit by unit (give assembly y
to units 1..w_jy). The aggregate model is then exact and constraint (6) is implied. `all_fitting_sets_are_kits`
checks the condition; `dispatch` constructs a unit-level assignment for any given (x, w) or proves none exists.
"""
from __future__ import annotations
import itertools
import time
from collections import defaultdict
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from .data import Instance


def kits(inst: Instance, j: int) -> list:
    """All kits (tuples of assembly ids, including the empty kit) of model j."""
    fit = [y for jj, y in inst.pairs if jj == j]
    out = []
    for r in range(len(fit) + 1):
        for K in itertools.combinations(fit, r):
            load = sum((inst.Bm[y] for y in K), np.zeros(len(inst.components)))
            if np.all(load <= inst.A[:, j] + 1e-9):
                out.append(K)
    return out


def all_fitting_sets_are_kits(inst: Instance) -> bool:
    fit = defaultdict(list)
    for j, y in inst.pairs:
        fit[j].append(y)
    return all(np.all(sum(inst.Bm[y] for y in ys) <= inst.A[:, j] + 1e-9) for j, ys in fit.items())


def solve(inst: Instance, lam: float = 1.0, stock_weight: float = 1.0, x_ub=None, fixed_x=None, x_lb=None,
          procurement_cap: float | None = None, stock_floor: float | None = None,
          time_limit: float = 600.0, mip_rel_gap: float = 1e-9) -> dict:
    """max stock_weight*U - lam*P over kit variables z (integer) and stock use u (continuous).

    x_lb, x_ub: optional per-model lower and upper bounds (e.g. a volume band around the programme or sales caps). Returns x, w, z (kit counts), status and gap.
    """
    n, m = len(inst.pool), len(inst.components)
    cols = [(j, K) for j in range(n) for K in kits(inst, j)]
    nz = len(cols)
    # net component requirement of one unit of model j built with kit K
    req = np.zeros((m, nz)); kval = np.zeros(nz)
    for c, (j, K) in enumerate(cols):
        r = inst.A[:, j].copy()
        for y in K:
            r -= inst.Bm[y]; kval[c] += inst.V[y]
        req[:, c] = r
    a = stock_weight
    # U = v.u + kval.z ; P = v.(req z) - v.u ; objective a*U - lam*P
    c_z = -(a * kval - lam * (inst.v @ req))
    c_u = -((a + lam) * inst.v)
    c = np.r_[c_z, c_u]
    rows, cc, vals, lo, hi = [], [], [], [], []

    def add(entries, l, h):
        r = len(lo)
        for ci, val in entries:
            rows.append(r); cc.append(ci); vals.append(val)
        lo.append(l); hi.append(h)

    add([(k, 1.0) for k in range(nz)], inst.Q, inst.Q)                              # sum x = Q
    by_y = defaultdict(list)
    for k, (j, K) in enumerate(cols):
        for y in K:
            by_y[y].append(k)
    for y, ks in by_y.items():                                                       # assembly stock
        add([(k, 1.0) for k in ks], -np.inf, inst.Ys[y])
    by_j = defaultdict(list)
    for k, (j, K) in enumerate(cols):
        by_j[j].append(k)
    for j, ks in by_j.items():                                                       # x bounds / fixed x
        lo_j, hi_j = 0.0, np.inf
        if x_lb is not None:
            lo_j = float(x_lb[j])
        if x_ub is not None:
            hi_j = float(x_ub[j])
        if fixed_x is not None:
            lo_j = hi_j = float(fixed_x[j])
        if lo_j > 0 or np.isfinite(hi_j):
            add([(k, 1.0) for k in ks], lo_j, hi_j)
    R = coo_matrix(req)
    ent = defaultdict(list)
    for i, k, val in zip(R.row, R.col, R.data):
        ent[i].append((k, -val))
    for i in range(m):                                                               # u_i <= d_i
        add([(nz + i, 1.0)] + ent[i], -np.inf, 0.0)
    if procurement_cap is not None:                                                  # P <= cap
        add([(k, float(inst.v @ req[:, k])) for k in range(nz)] + [(nz + i, -inst.v[i]) for i in range(m)],
            -np.inf, procurement_cap)
    if stock_floor is not None:                                                      # U >= floor
        add([(k, kval[k]) for k in range(nz) if kval[k]] + [(nz + i, inst.v[i]) for i in range(m)], stock_floor, np.inf)
    M = coo_matrix((vals, (rows, cc)), shape=(len(lo), nz + m)).tocsc()
    t = time.perf_counter()
    res = milp(c, integrality=np.r_[np.ones(nz), np.zeros(m)],
               bounds=Bounds(np.zeros(nz + m), np.r_[np.full(nz, inst.Q), inst.s]),
               constraints=LinearConstraint(M, lo, hi), options={"mip_rel_gap": mip_rel_gap, "time_limit": time_limit})
    sec = time.perf_counter() - t
    if res.x is None:
        raise RuntimeError(res.message)
    z = np.rint(res.x[:nz]).astype(int)
    x = np.zeros(n, dtype=int); w = defaultdict(int); zk = {}
    for k, (j, K) in enumerate(cols):
        if z[k]:
            x[j] += z[k]; zk[(j, K)] = int(z[k])
            for y in K:
                w[(j, y)] += int(z[k])
    return dict(x=x, w=dict(w), z=zk, seconds=sec, status=int(res.status), mip_gap=float(res.mip_gap or 0.0),
                n_kits=nz)


def dispatch(inst: Instance, x, w: dict):
    """Assign assemblies to individual units. Returns {model index: [kit per unit]} or None if impossible.

    Solved exactly per model as a small integer feasibility problem over kit counts. EVERY fitting assembly of
    the model is fixed to its stored count (zero when unused), so the dispatch reproduces w exactly; the result
    is then re-checked by check_dispatch.
    """
    pairs = set(inst.pairs)
    if any((j, y) not in pairs or k < 0 for (j, y), k in w.items()):
        return None
    out = {}
    for j in range(len(inst.pool)):
        xj = int(x[j])
        fit = sorted(y for jj, y in inst.pairs if jj == j)
        need = {y: int(w.get((j, y), 0)) for y in fit}
        if xj == 0:
            if any(need.values()):
                return None
            out[j] = []
            continue
        K = kits(inst, j)
        Aeq = np.zeros((1 + len(fit), len(K)))
        Aeq[0, :] = 1
        for r, y in enumerate(fit):
            for k, kit in enumerate(K):
                if y in kit:
                    Aeq[1 + r, k] = 1
        b = np.r_[xj, [need[y] for y in fit]]
        res = milp(np.zeros(len(K)), integrality=np.ones(len(K)), bounds=Bounds(0, xj),
                   constraints=LinearConstraint(Aeq, b, b))
        if res.x is None or res.status != 0:
            return None
        units = []
        for k, cnt in enumerate(np.rint(res.x).astype(int)):
            units += [K[k]] * cnt
        out[j] = units
    return out if check_dispatch(inst, x, w, out) else None


def check_dispatch(inst: Instance, x, w: dict, units: dict) -> bool:
    """Independent check of a unit-level dispatch: unit counts, kit validity, exact reproduction of w, stock."""
    used = defaultdict(int)
    for j in range(len(inst.pool)):
        us = units.get(j, [])
        if len(us) != int(x[j]):
            return False
        cnt = defaultdict(int)
        for K in us:
            if len(set(K)) != len(K):
                return False
            load = sum((inst.Bm[y] for y in K), np.zeros(len(inst.components)))
            if not np.all(load <= inst.A[:, j] + 1e-9):
                return False
            for y in K:
                if (j, y) not in set(inst.pairs):
                    return False
                cnt[y] += 1; used[y] += 1
        expect = {y: int(k) for (jj, y), k in w.items() if jj == j and k}
        if dict(cnt) != expect:
            return False
    return all(used[y] <= inst.Ys[y] + 1e-9 for y in used)
