"""Exact MILP for reverse planning with optional semi-finished substitution (Section 3)."""
from __future__ import annotations
import time
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from .data import Instance


def solve(inst: Instance, lam: float = 1.0, fixed_x=None, time_limit: float = 600.0,
          mip_rel_gap: float = 1e-9, stock_weight: float = 1.0, procurement_cap: float | None = None,
          stock_floor: float | None = None) -> dict:
    """max stock_weight*(sum v u + sum V w) - lam * sum v p  s.t. (1)-(6)
    [+ sum v p <= procurement_cap] [+ sum v u + sum V w >= stock_floor].

    Variables x | w | u; p = d - u is implicit. The base model is stock_weight = lam = 1.
    """
    n, P, m = len(inst.pool), len(inst.pairs), len(inst.components)
    Vw = np.array([inst.V[y] for _, y in inst.pairs]) if P else np.zeros(0)
    # sum v p = v.A x - V.w - v.u, so objective = (a+lam) v.u - lam v.A x + (a+lam) V.w with a = stock_weight
    a = stock_weight
    c = -np.r_[-lam * (inst.A.T @ inst.v), (a + lam) * Vw, (a + lam) * inst.v]
    rows, cols, vals, lo, hi = [], [], [], [], []

    def add(entries, l, h):
        r = len(lo)
        for ci, val in entries:
            rows.append(r); cols.append(ci); vals.append(val)
        lo.append(l); hi.append(h)

    add([(j, 1.0) for j in range(n)], inst.Q, inst.Q)                       # (2)
    for k, (j, y) in enumerate(inst.pairs):                                 # (4)
        add([(n + k, 1.0), (j, -1.0)], -np.inf, 0.0)
    by_y = defaultdict(list)
    for k, (j, y) in enumerate(inst.pairs):
        by_y[y].append(k)
    for y, ks in by_y.items():                                              # (5)
        add([(n + k, 1.0) for k in ks], -np.inf, inst.Ys[y])
    by_j = defaultdict(list)
    for k, (j, y) in enumerate(inst.pairs):
        by_j[j].append((k, y))
    for j, lst in by_j.items():                                             # (6), shared components only
        if len(lst) < 2:
            continue
        cnt = Counter(i for _, y in lst for i in np.flatnonzero(inst.Bm[y]))
        for i, nshare in cnt.items():
            if nshare >= 2:
                add([(n + k, inst.Bm[y][i]) for k, y in lst if inst.Bm[y][i] > 0] + [(j, -inst.A[i, j])], -np.inf, 0.0)
    if procurement_cap is not None:                                         # epsilon-constraint on procurement value
        add([(j, float(inst.A[:, j] @ inst.v)) for j in range(n)] + [(n + k, -inst.V[y]) for k, (j, y) in enumerate(inst.pairs)]
            + [(n + P + i, -inst.v[i]) for i in range(m) if inst.v[i] != 0], -np.inf, procurement_cap)
    if stock_floor is not None:                                             # lexicographic second stage
        add([(n + k, inst.V[y]) for k, (j, y) in enumerate(inst.pairs)]
            + [(n + P + i, inst.v[i]) for i in range(m) if inst.v[i] != 0], stock_floor, np.inf)
    Ac = coo_matrix(inst.A)
    dent = defaultdict(list)
    for i, j, a in zip(Ac.row, Ac.col, Ac.data):
        dent[i].append((j, -a))
    for k, (j, y) in enumerate(inst.pairs):
        for i in np.flatnonzero(inst.Bm[y]):
            dent[i].append((n + k, inst.Bm[y][i]))
    for i in range(m):                                                      # u_i <= d_i (net requirement)
        add([(n + P + i, 1.0)] + dent[i], -np.inf, 0.0)
    M = coo_matrix((vals, (rows, cols)), shape=(len(lo), n + P + m)).tocsc()
    lb = np.zeros(n + P + m)
    ub = np.r_[np.full(n + P, inst.Q), inst.s]                              # (3)
    if fixed_x is not None:
        lb[:n] = fixed_x; ub[:n] = fixed_x
    t = time.perf_counter()
    res = milp(c, integrality=np.r_[np.ones(n + P), np.zeros(m)], bounds=Bounds(lb, ub),
               constraints=LinearConstraint(M, lo, hi),
               options={"mip_rel_gap": mip_rel_gap, "time_limit": time_limit})
    sec = time.perf_counter() - t
    if res.x is None:
        raise RuntimeError(res.message)
    x = np.rint(res.x[:n]).astype(int)
    w = {inst.pairs[k]: int(round(res.x[n + k])) for k in range(P) if round(res.x[n + k]) > 0}
    return dict(x=x, w=w, seconds=sec, status=int(res.status), mip_gap=float(res.mip_gap or 0.0))
