"""Stage A: programme design on the gross stock (before the remaining frozen programme is served).

The remaining programme (Q = all programme units, every programme model in the pool) is re-optimised with the
exact kit formulation (kits.py) under mix-flexibility bands  floor(q_j / f) <= x_j <= ceil(q_j * f),  where q_j
is the programme volume of model j. f = 1 reproduces the actual programme; f = None removes the band. Each case
is solved with and without stocked assemblies, with the assembly stock scaled (availability), and with one
assembly type at a time removed (leave-one-type-out).

python -m revplan.stage_a           # writes results/stage_a.json
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import numpy as np
from . import data, kits, heuristics, checks, netting

RESULTS = Path(__file__).resolve().parents[2] / "results"
BANDS = [1.0, 1.1, 1.25, 1.5, 2.0, None]


def instance(ds=None, use_assemblies=True, drop_kind=None):
    ds = ds or data.load(stock="gross")
    pool = [p for p in ds.products if ds.programme[p] > 0]
    Q = sum(ds.programme[p] for p in pool)
    inst = data.build(ds, len(pool), Q, 1.0, use_assemblies=use_assemblies)
    assert inst.pool == pool
    if drop_kind:
        keep = {y for y in inst.ys if ds.assembly_kind[y] != drop_kind}
        inst.pairs = [(j, y) for j, y in inst.pairs if y in keep]
        inst.ys = sorted(keep & {y for _, y in inst.pairs})
    q = np.array([ds.programme[p] for p in pool])
    return ds, inst, q


def band(q, f):
    if f is None:
        return None, None
    return np.floor(q / f + 1e-9), np.ceil(q * f - 1e-9)


def _pack(ds, inst, q, sol):
    x, w = sol["x"], sol["w"]
    checks.check_assignment(inst, x, w)
    units = kits.dispatch(inst, x, w) if inst.pairs else {}
    if inst.pairs and units is None:
        raise RuntimeError("no unit-level dispatch")
    ev = data.evaluate(inst, x, w)
    kind_units = {}
    for (j, y), k in w.items():
        kind_units[ds.assembly_kind[y]] = kind_units.get(ds.assembly_kind[y], 0) + int(k)
    return dict(**ev, stock_used_total=ev["stock_used"] + ev["assembly_value_used"],
                units_moved=int(np.abs(x - q).sum() // 2), status=sol.get("status", 0), mip_gap=sol.get("mip_gap", 0.0),
                seconds=sol.get("seconds", 0.0), assemblies_by_kind=kind_units,
                mix={inst.pool[j]: int(v) for j, v in enumerate(x) if v},
                assemblies=[[inst.pool[j], y, int(k)] for (j, y), k in sorted(w.items())])


def plant_plan(ds=None) -> dict:
    """The plant's own plan for the remaining programme (main netting rule): the actual mix with the stocked
    assemblies named in its plan rows, allocated in plan order (revplan.netting). Valued like a model plan:
    U = component stock used + value of the assemblies installed, P = value of the components still to buy."""
    ds = ds or data.load(stock="gross")
    v = ds.values["unit_value"]
    n = netting.net(ds.bom, ds.assembly_bom, ds.stock, ds.assembly_stock, netting.read_units(), netting.MAIN_RULE)
    P = sum(v[c] * max(q - ds.stock.get(c, 0.0), 0.0) for c, q in n.requirement.items())
    Uc = sum(v[c] * min(q, ds.stock.get(c, 0.0)) for c, q in n.requirement.items())
    used = [y for a in n.allocation.values() for y in a]
    Uy = sum(sum(v[c] * q for c, q in ds.assembly_bom[y].items()) for y in used)
    return dict(units=len(n.units), assembly_units=len(used), stock_used_total=Uc + Uy, procurement=P, Z=Uc + Uy - P)


def run(out: Path = RESULTS):
    ds = data.load(stock="gross")
    res = dict(fingerprint=checks.fingerprint("stage_a"), bands=[], leave_one_type_out=[], greedy={})
    res["plant_plan"] = plant_plan(ds)
    for label, use in [("components_only", False), ("with_assemblies", True)]:
        _, inst, q = instance(ds, use)
        if use:
            res["all_fitting_sets_are_kits"] = kits.all_fitting_sets_are_kits(inst)
            res["n_assemblies_in_stock"] = len(inst.ys)
            res["assembly_units_in_stock"] = int(sum(inst.Ys.values()))
            res["Q"] = int(inst.Q); res["n_models"] = len(inst.pool)
        for f in BANDS:
            lb, ub = band(q, f)
            sol = kits.solve(inst, x_lb=lb, x_ub=ub)
            res["bands"].append(dict(model=label, band=f, **_pack(ds, inst, q, sol)))
            print("stage A", label, f, round(res["bands"][-1]["Z"]), flush=True)
        gx, gw = heuristics.greedy(inst)
        res["greedy"][label] = _pack(ds, inst, q, dict(x=gx, w=gw))
    _, inst0, q = instance(ds, True)
    res["availability"] = []
    for fa in (0.0, 0.25, 0.5, 0.75, 1.0):                   # assembly stock scaled, component stock fixed
        inst = copy.deepcopy(inst0)
        inst.Ys = {y: float(math.floor(s * fa + 1e-9)) for y, s in inst0.Ys.items()}
        for f in (1.0, None):
            lb, ub = band(q, f)
            res["availability"].append(dict(assembly_stock_factor=fa, band=f, **_pack(ds, inst, q, kits.solve(inst, x_lb=lb, x_ub=ub))))
    for kind in sorted(set(ds.assembly_kind.values())):
        _, inst, q = instance(ds, True, drop_kind=kind)
        for f in (1.0, None):
            lb, ub = band(q, f)
            sol = kits.solve(inst, x_lb=lb, x_ub=ub)
            res["leave_one_type_out"].append(dict(dropped=kind, band=f, **_pack(ds, inst, q, sol)))
            print("stage A drop", kind, f, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage_a.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    run()
