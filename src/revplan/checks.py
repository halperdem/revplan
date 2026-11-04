"""Feasibility checks and result fingerprints.

Kept apart from data.py, model.py and heuristics.py so that the fingerprint covers exactly the code that
produces the results, and tightening a check does not invalidate stored results.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .data import Instance, DATA_DIR
def _as_int(v, what):
    """Return v as int if it is a finite non-negative integer value, else raise."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError(f"{what} is not a number: {v!r}")
    f = float(v)
    if not np.isfinite(f) or f < 0 or f != int(f):
        raise ValueError(f"{what} must be a finite non-negative integer, got {v!r}")
    return int(f)


def check_assignment(inst: Instance, x, w: dict, tol: float = 1e-9) -> None:
    """Raise ValueError if a plan or assembly assignment is not a feasible integer solution.

    Checks integrality, finiteness and sign of every x_j and w_jy, constraint (2) sum x = Q, (4) w_jy <= x_j,
    (5) sum_j w_jy <= S_y, and BOM containment for every component of every model (not only shared ones),
    independently of the MILP.
    """
    x = np.asarray(x, dtype=object).ravel()
    if len(x) != len(inst.pool):
        raise ValueError("plan has the wrong length")
    xi = np.array([_as_int(v, f"x[{inst.pool[j]}]") for j, v in enumerate(x)], dtype=int)
    if int(xi.sum()) != inst.Q:
        raise ValueError(f"plan has {int(xi.sum())} units, expected Q = {inst.Q}")
    used, load = {}, {}
    pairs = set(inst.pairs)
    for (j, y), k in w.items():
        k = _as_int(k, f"w[{(j, y)}]")
        if (j, y) not in pairs:
            raise ValueError(f"assembly {y} does not fit model index {j}")
        if k > xi[j]:
            raise ValueError(f"constraint (4) violated for {(j, y)}")
        used[y] = used.get(y, 0) + k
        load[j] = load.get(j, 0) + inst.Bm[y] * k
    for y, k in used.items():
        if k > inst.Ys[y] + tol:
            raise ValueError(f"constraint (5) violated for {y}")
    for j, l in load.items():
        if np.any(l > inst.A[:, j] * xi[j] + tol):
            raise ValueError(f"containment violated for model {inst.pool[j]}")


CORE_MODULES = ("data.py", "model.py", "heuristics.py")
EXTRA_MODULES = CORE_MODULES + ("kits.py", "extra.py", "netting.py", "stage_a.py", "experiment.py")
STAGE_A_MODULES = CORE_MODULES + ("kits.py", "stage_a.py", "netting.py")


def fingerprint(scope: str = "core") -> str:
    """SHA-256 of the plant data and of the modules that determine results (line endings normalised).

    scope "core": results of experiment.py (data.py, model.py, heuristics.py);
    scope "extra": results of extra.py, which also depend on kits.py and extra.py;
    scope "stage_a": results of stage_a.py (core modules, kits.py, stage_a.py and netting.py).
    """
    import hashlib
    h = hashlib.sha256()
    src = Path(__file__).resolve().parent
    mods = {"core": CORE_MODULES, "extra": EXTRA_MODULES, "stage_a": STAGE_A_MODULES}[scope]
    files = sorted(DATA_DIR.glob("*.csv")) + [src / f for f in mods]
    for f in files:
        h.update(f.name.encode()); h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]
