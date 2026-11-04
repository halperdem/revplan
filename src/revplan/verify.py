"""Independent checks of the stored results (python -m revplan.verify).

Writes results/verification.json. Each check recomputes from the data and the stored decisions:
  1. stored MILP and greedy plans: assignment feasibility (checks.check_assignment) and stored values;
  2. greedy re-run: same plan and value as stored;
  3. programme-proportional baseline: stored mix equals the rule, value recomputed, not above the optimum and
     dominated by an optimal plan (marginal value of the last unit is reported, not tested: in Stage B it is
     negative whenever an extra unit needs more procurement than it uses stock);
  4. allocation check: for the best mix of each GA scenario and variant (24 mixes) the heuristic
     assembly allocation gives the same value as the exact allocation with x fixed;
  5. GA records: every stored mix (720 runs) re-evaluated; value equals the stored value and
     does not exceed the MILP optimum;
  6. efficient fronts: solver status and gaps of both stages, monotonicity of the points;
  7. unit level: aggregate model exact on the case (all fitting sets are kits), kit formulation reproduces the
     36 weighted optima with assemblies, every stored assignment can be dispatched to individual units;
  8. robustness analyses (extra.py): all present, current fingerprint, recomputed and compared;
  9. Stage A (stage_a.json): current fingerprint, band 1 equals the actual programme, results recomputed and compared;
 10. plant data: residual stock never exceeds gross stock;
 11. GA convergence traces: present, made with the same data and code, all runs reproduced, each trace
     ends at the stored value and has generations + 1 entries; 24 GA runs re-run from seed with the current code.
Every stored result must carry the fingerprint of the current data and code. Any missing file or failed check
makes the program exit with status 1.
"""
from __future__ import annotations
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from . import data, model, heuristics, checks, kits
from .experiment import RESULTS, POOLS, QS, _mix, proportional


def load_plan(inst, mix: dict, assemblies=()):
    """Turn a stored plan into (x, w) only after validating the raw stored values.

    Raw quantities are passed unchanged to checks.check_assignment, so 20.5, NaN, strings or unknown model ids
    raise ValueError instead of being silently converted. Returns integer x and w.
    """
    idx = {p: j for j, p in enumerate(inst.pool)}
    if not isinstance(mix, dict):
        raise ValueError("mix is not a mapping")
    unknown = [p for p in mix if p not in idx] + [p for p, _, _ in assemblies if p not in idx]
    if unknown:
        raise ValueError(f"unknown model ids {unknown[:3]}")
    x = np.zeros(len(inst.pool), dtype=object)
    for p, q in mix.items():
        x[idx[p]] = q
    w = {}
    for p, y, k in assemblies:
        if (idx[p], y) in w:
            raise ValueError(f"duplicate assignment {(p, y)}")
        w[(idx[p], y)] = k
    checks.check_assignment(inst, x, w)                     # validates the raw values
    return x.astype(int), {key: int(k) for key, k in w.items()}


def _close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def check_deterministic(ds, det, failures: list, zopt: dict | None = None, last_unit: bool = True,
                        expected: dict | None = None, resolve: bool = False) -> dict:
    """Checks 1-3 on the scenarios in det; appends failure messages to failures and returns the report part."""
    expected = EXPECTED if expected is None else expected
    zopt = {} if zopt is None else zopt
    stored_ok, greedy_ok, lu, prop_ok, resolved = [], [], [], [], []
    for row in det["scenarios"]:
        k, Q, sf = row["pool"], row["Q"], row["stock_factor"]
        for label, use in [("components_only", False), ("with_assemblies", True)]:
            inst = data.build(ds, k, Q, sf, use_assemblies=use)
            r = row[label]
            for meth in ("milp", "greedy"):
                try:
                    x, w = load_plan(inst, r[meth]["mix"], r[meth]["assemblies"])
                    stored_ok.append(_close(data.evaluate(inst, x, w)["Z"], r[meth]["Z"]))
                except (ValueError, TypeError, KeyError) as e:
                    stored_ok.append(False); failures.append(f"{k}/{Q}/{sf}/{label}/{meth}: {e}")
            if r["milp"].get("status") != 0:
                failures.append(f"{k}/{Q}/{sf}/{label}: MILP not optimal")
            if "proportional" in r:
                pr = r["proportional"]
                try:
                    x, w = load_plan(inst, pr["mix"], pr["assemblies"])
                    e = data.evaluate(inst, x, w)
                    U = e["stock_used"] + e["assembly_value_used"]
                    ok = (_close(e["Z"], pr["Z"]) and list(x) == list(proportional(ds, inst))
                          and e["Z"] <= r["milp"]["Z"] + 1e-9 * max(1.0, abs(r["milp"]["Z"]))
                          and pr["dominance_status"] == [0, 0])
                    # the two witness plans: feasible, values recomputed, reference constraints met, summary = plan
                    wp = pr["dominance_plans"]
                    xa, wa = load_plan(inst, wp["min_procurement_at_same_stock_use"]["mix"], wp["min_procurement_at_same_stock_use"]["assemblies"])
                    xb, wb = load_plan(inst, wp["max_stock_use_at_same_procurement"]["mix"], wp["max_stock_use_at_same_procurement"]["assemblies"])
                    ea, eb = data.evaluate(inst, xa, wa), data.evaluate(inst, xb, wb)
                    Ua, Ub = ea["stock_used"] + ea["assembly_value_used"], eb["stock_used"] + eb["assembly_value_used"]
                    ok = ok and Ua >= U * (1 - 1e-9) - 1e-9 and eb["procurement"] <= e["procurement"] * (1 + 1e-9) + 1e-9 \
                        and _close(ea["procurement"], pr["min_procurement_at_same_stock_use"]) and _close(Ub, pr["max_stock_use_at_same_procurement"]) \
                        and _close(ea["procurement"], wp["min_procurement_at_same_stock_use"]["procurement"]) and _close(Ua, wp["min_procurement_at_same_stock_use"]["stock_used"]) \
                        and _close(eb["procurement"], wp["max_stock_use_at_same_procurement"]["procurement"]) and _close(Ub, wp["max_stock_use_at_same_procurement"]["stock_used"]) \
                        and ea["procurement"] <= e["procurement"] * (1 + 1e-9) and Ub >= U * (1 - 1e-9)
                    if resolve:                          # optional: solve both problems again and compare optima
                        sa_ = model.solve(inst, lam=1.0, stock_weight=0.0, stock_floor=U * (1 - 1e-9))
                        sb_ = model.solve(inst, lam=0.0, stock_weight=1.0, procurement_cap=e["procurement"] * (1 + 1e-9))
                        ra = data.evaluate(inst, sa_["x"], sa_["w"]); rb = data.evaluate(inst, sb_["x"], sb_["w"])
                        ok = ok and sa_["status"] == 0 and sb_["status"] == 0 and _close(ra["procurement"], ea["procurement"], 1e-7) \
                            and _close(rb["stock_used"] + rb["assembly_value_used"], Ub, 1e-7)
                        resolved.append(ok)
                    prop_ok.append(ok)
                except (ValueError, TypeError, KeyError) as ex:
                    prop_ok.append(False); failures.append(f"{k}/{Q}/{sf}/{label}/proportional: {ex}")
            gx, gw = heuristics.greedy(inst)
            greedy_ok.append(_close(data.evaluate(inst, gx, gw)["Z"], r["greedy"]["Z"]) and
                             {inst.pool[j]: int(q) for j, q in enumerate(gx) if q} == r["greedy"]["mix"])
            if last_unit:
                small = data.build(ds, k, Q - 1, sf, use_assemblies=use)
                s = model.solve(small)
                z1 = data.evaluate(small, s["x"], s["w"])["Z"]
                lu.append(dict(pool=k, Q=Q, stock_factor=sf, model=label, Z_Q=r["milp"]["Z"], Z_Q_minus_1=z1,
                               status=s["status"], last_unit_adds_value=bool(z1 < r["milp"]["Z"])))
                if s["status"] != 0:
                    failures.append(f"{k}/{Q - 1}/{sf}/{label}: MILP not optimal")
            if use and sf == 1.0:
                zopt[(k, Q)] = r["milp"]["Z"]
    if not (len(stored_ok) == expected["plans"] and all(stored_ok)):
        failures.append("stored MILP/greedy plans")
    if not (len(greedy_ok) == expected["greedy"] and all(greedy_ok)):
        failures.append("greedy re-run")
    rep = dict(stored_plans_feasible_and_values_match=dict(checked=len(stored_ok), passed=int(sum(stored_ok))),
               greedy_rerun_matches=dict(checked=len(greedy_ok), passed=int(sum(greedy_ok))))
    if not (len(prop_ok) == expected["proportional"] and all(prop_ok)):
        failures.append("programme-proportional baseline")
    rep["proportional_baseline_and_witness_plans"] = dict(checked=len(prop_ok), passed=int(sum(prop_ok)),
                                                         re_solved=len(resolved), re_solved_match=int(sum(resolved)))
    if last_unit:                                      # reported only (see module docstring)
        rep["marginal_last_unit"] = dict(solved=len(lu), last_unit_adds_value=int(sum(c["last_unit_adds_value"] for c in lu)), details=lu)
    return rep


def check_unit_level(ds, det, fronts, failures: list) -> dict:
    """Unit-level validity of the aggregate model on the case data (kits.py).

    (a) In every instance the full set of fitting assemblies of each model fits into one unit, so the aggregate
    model is exact (Proposition in kits.py); (b) the exact kit formulation reproduces every stored weighted optimum
    with assemblies; (c) every stored assignment (MILP, greedy, lambda variants, front points) can be dispatched
    to individual units.
    """
    cond, same, disp = [], [], []
    for row in det["scenarios"]:
        inst = data.build(ds, row["pool"], row["Q"], row["stock_factor"], use_assemblies=True)
        cond.append(kits.all_fitting_sets_are_kits(inst))
        r = row["with_assemblies"]
        k = kits.solve(inst)
        same.append(k["status"] == 0 and _close(data.evaluate(inst, k["x"], k["w"])["Z"], r["milp"]["Z"]))
        plans = [r["milp"], r["greedy"]] + list(r.get("lambda", {}).values())
        for pl in plans:
            try:
                x, w = load_plan(inst, pl["mix"], pl["assemblies"])
                disp.append(kits.dispatch(inst, x, w) is not None)
            except (ValueError, TypeError, KeyError):
                disp.append(False)
    for c in fronts or []:
        inst = data.build(ds, c["pool"], c["Q"], 1.0, use_assemblies=True)
        for p in c["front"]:
            try:
                x, w = load_plan(inst, p["mix"], p["assemblies"])
                disp.append(kits.dispatch(inst, x, w) is not None)
            except (ValueError, TypeError, KeyError):
                disp.append(False)
    if not (cond and all(cond)):
        failures.append("unit level: a model's fitting assemblies do not fit into one unit")
    if not (len(same) == 36 and all(same)):
        failures.append("unit level: kit formulation optimum differs from stored optimum")
    if not (disp and all(disp)):
        failures.append("unit level: a stored assignment cannot be dispatched to units")
    return dict(unit_level=dict(instances_where_all_fitting_sets_are_kits=int(sum(cond)), instances=len(cond),
                                kit_optimum_matches=int(sum(same)), kit_checked=len(same),
                                assignments_dispatchable=int(sum(disp)), assignments_checked=len(disp)))


EXTRA_FILES = ("sales_caps", "valuation", "composition", "ga_stats", "netting_sensitivity", "plans_report", "composition_limits",
               "pair_diagnostics")


def _same(a, b, rel=1e-9):
    """Recursive comparison of two JSON values with a relative tolerance for floats."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], rel) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return _close(float(a), float(b), rel)
    return a == b


def check_extra(out: Path, failures: list, sample_reruns: bool = True) -> dict:
    """All Stage B robustness analyses must exist, carry the current 'extra' fingerprint and be reproducible:
    each is recomputed into a temporary folder and compared with the stored file."""
    import tempfile
    from . import extra
    fpx = checks.fingerprint("extra")
    exdir = out / "extra"
    rep = dict(present=[], missing=[], stale=[], reproduced=[], differs=[])
    for name in EXTRA_FILES:
        f = exdir / f"{name}.json"
        if not f.exists():
            rep["missing"].append(name); continue
        rep["present"].append(name)
        if json.loads(f.read_text()).get("fingerprint") != fpx:
            rep["stale"].append(name)
    with tempfile.TemporaryDirectory() as tmp:
        for name in EXTRA_FILES:
            if name not in rep["present"]:
                continue
            getattr(extra, name)(out=tmp)
            same = _same(json.loads((Path(tmp) / f"{name}.json").read_text())["results"],
                         json.loads((exdir / f"{name}.json").read_text())["results"], rel=1e-6)
            (rep["reproduced"] if same else rep["differs"]).append(name)
    if rep["missing"]:
        failures.append(f"extra analyses missing: {rep['missing']}")
    if rep["stale"]:
        failures.append(f"extra analyses produced with different data or code: {rep['stale']}")
    if rep["differs"]:
        failures.append(f"extra analyses not reproduced: {rep['differs']}")
    return dict(extra_analyses=rep)


EXPECTED = dict(plans=144, greedy=72, proportional=72, allocation=24, ga_runs=720, front_points=105, ga_scenarios=12,
                stage_a_bands=12, stage_a_availability=10, stage_a_leave_one_out=12)


def check_stage_a(out: Path, failures: list) -> dict:
    """Stage A results: fingerprint, actual programme reproduced at band 1, full recomputation and comparison."""
    import tempfile
    from . import stage_a
    f = out / "stage_a.json"
    if not f.exists():
        failures.append("missing file stage_a.json"); return {}
    A = json.loads(f.read_text())
    rep = dict(fingerprint_current=A.get("fingerprint") == checks.fingerprint("stage_a"))
    ds = data.load(stock="gross")
    prog = {p: q for p, q in ds.programme.items() if q}
    rep["band1_is_programme"] = all(r["mix"] == prog and r["units_moved"] == 0 for r in A["bands"] if r["band"] == 1.0)
    rep["counts_ok"] = (len(A["bands"]) == EXPECTED["stage_a_bands"] and len(A["availability"]) == EXPECTED["stage_a_availability"]
                        and len(A["leave_one_type_out"]) == EXPECTED["stage_a_leave_one_out"])
    rep["all_optimal"] = all(r["status"] == 0 for r in A["bands"] + A["availability"] + A["leave_one_type_out"])
    rep["all_fitting_sets_are_kits"] = bool(A.get("all_fitting_sets_are_kits"))
    with tempfile.TemporaryDirectory() as tmp:
        stage_a.run(Path(tmp))
        new = json.loads((Path(tmp) / "stage_a.json").read_text())
    strip = lambda d: {k: ([{a: b for a, b in r.items() if a != "seconds"} for r in v] if isinstance(v, list) else
                           {kk: {a: b for a, b in vv.items() if a != "seconds"} for kk, vv in v.items()} if k == "greedy" else v)
                       for k, v in d.items()}
    rep["reproduced"] = _same(strip(new), strip(A), rel=1e-6)
    for key, v in rep.items():
        if not v:
            failures.append(f"stage A: {key}")
    return dict(stage_a=rep)


def check_release_data(failures: list) -> dict:
    """Residual stock and programme are reproduced from the published gross stock and programme units
    (revplan.netting, main rule); inferred assembly compatibility is compared with the plan rows."""
    from . import netting
    g, r = data.load(stock="gross"), data.load(stock="residual")
    comp = all(r.stock.get(c, 0) <= g.stock.get(c, 0) + 1e-9 and r.stock.get(c, 0) >= 0 for c in g.stock)
    asm = all(0 <= r.assembly_stock[y] <= g.assembly_stock[y] for y in g.assemblies)
    units = netting.read_units()
    n = netting.net(g.bom, g.assembly_bom, g.stock, g.assembly_stock, units, netting.MAIN_RULE)
    same_c = all(n.residual[c] == r.stock[c] for c in r.stock)
    same_a = all(n.assembly_residual[y] == r.assembly_stock[y] for y in r.assemblies)
    same_p = all(n.programme.get(p, 0) == g.programme[p] for p in g.products)
    if not (comp and asm):
        failures.append("plant data: residual stock exceeds gross stock or is negative")
    if not (same_c and same_a and same_p):
        failures.append("plant data: residual stock or programme not reproduced by revplan.netting")
    inst = data.build(g, len(g.products), 1, use_assemblies=True)
    inferred = {(inst.pool[j], y) for j, y in inst.pairs}
    planned = {(u["product_id"], y) for u in units for y in u["assemblies"]}
    # BOM coverage rule (data/README.md): every model's BOM must contain at least half of the components of
    # every assembly its plan rows name, in at least the assembly's quantity
    cov = {(p, y): sum(1 for c, q in g.assembly_bom[y].items() if g.bom[p].get(c, 0) >= q - 1e-9) / len(g.assembly_bom[y])
           for p, y in planned}
    if min(cov.values()) < 0.5:
        failures.append("plant data: a model's BOM fails the assembly coverage rule")
    return dict(release_data=dict(bom_coverage_min=round(min(cov.values()), 4),
                                  bom_coverage_pairs_below_0_95=sorted(f"{p}/{y}" for (p, y), v in cov.items() if v < 0.95),residual_le_gross_components=comp, residual_le_gross_assemblies=asm,
                                  netting_reproduces_residual_components=same_c, netting_reproduces_residual_assemblies=same_a,
                                  netting_reproduces_programme=same_p, programme_units=int(sum(g.programme.values())),
                                  assembly_units_gross=int(sum(g.assembly_stock.values())),
                                  assembly_units_residual=int(sum(r.assembly_stock.values())),
                                  compatibility_pairs_inferred=len(inferred), inferred_pairs_in_plans=len(inferred & planned),
                                  plan_pairs_not_inferred=len(planned - inferred)))


def main(out: Path = RESULTS, sample_reruns: bool = True, resolve: bool = False) -> bool:
    """Run all checks; write results/verification.json; return True only if every check passes."""
    ds = data.load()
    fp = checks.fingerprint()
    rep, failures = {"fingerprint": fp}, []

    def need(cond, msg):
        if not cond:
            failures.append(msg)

    def load(path):
        if not path.exists():
            failures.append(f"missing file {path.relative_to(out)}")
            return None
        return json.loads(path.read_text())

    det = load(out / "deterministic.json")
    zopt = {}
    if det is not None:
        need(det.get("fingerprint") == fp, "deterministic.json: fingerprint differs from current data/code")
        rep.update(check_deterministic(ds, det, failures, zopt, resolve=resolve))

    # GA records, exact allocation, sampled re-runs, convergence traces
    alloc, ga_ok, ga_le, reruns, traces = [], [], [], [], []
    hist = load(out / "ga" / "history_check.json") or {}
    for i, (k, Q) in enumerate((k, Q) for k in POOLS for Q in QS):
        rec = load(out / "ga" / f"ga_{k}_{Q}.json")
        if rec is None or (k, Q) not in zopt:
            continue
        need(rec.get("fingerprint") == fp, f"ga_{k}_{Q}.json: fingerprint differs from current data/code")
        inst = data.build(ds, k, Q, 1.0, use_assemblies=True)
        dec = heuristics.Decoder(inst)
        gx = heuristics.greedy(inst)[0]
        for variant in ("random", "greedy_seeded"):
            runs = rec["runs"].get(variant, [])
            need(len(runs) == 30, f"ga_{k}_{Q}.json: {variant} has {len(runs)} runs, expected 30")
            if not runs:
                continue
            plans = {}
            for r in runs:
                try:
                    plans[r["run"]] = load_plan(inst, r["mix"])[0]
                except (ValueError, TypeError) as e:
                    ga_ok.append(False); ga_le.append(False)
                    failures.append(f"ga_{k}_{Q}.json {variant} run {r.get('run')}: {e}")
                    continue
                z = float(dec.score(plans[r["run"]])[0])
                ga_ok.append(_close(z, r["Z"]))
                ga_le.append(z <= zopt[(k, Q)] + 1e-9 * max(1.0, abs(zopt[(k, Q)])))      # Z can be negative
            valid = [r for r in runs if r["run"] in plans]
            if not valid:
                continue
            best = max(valid, key=lambda r: r["Z"])
            x = plans[best["run"]]
            s = model.solve(inst, fixed_x=x)
            ze = data.evaluate(inst, x, s["w"])["Z"]
            zh = float(dec.score(x)[0])
            alloc.append(dict(pool=k, Q=Q, variant=variant, heuristic=zh, exact=ze, status=s["status"], equal=_close(zh, ze)))
            if sample_reruns:                              # first run of every scenario and variant, from its seed
                r = runs[0]
                g = heuristics.ga(inst, r["seed"], seed_x=gx if variant == "greedy_seeded" else None)
                reruns.append(dict(pool=k, Q=Q, variant=variant, run=r["run"],
                                   reproduced=_close(g["Z"], r["Z"]) and _mix(inst, g["x"]) == r["mix"]
                                   and g["generations"] == r["generations"]))
        hf = out / "ga" / f"history_{k}_{Q}.csv.gz"
        h = hist.get(f"{k}_{Q}")
        if not hf.exists() or h is None:
            failures.append(f"missing convergence trace for {k}/{Q}")
            continue
        need(h.get("fingerprint") == fp, f"history {k}/{Q}: fingerprint differs from current data/code")
        need(h["runs_reproduced"] == 60 and not h["mismatches"], f"history {k}/{Q}: not all runs reproduced")
        last, count = {}, {}
        with gzip.open(hf, "rt", newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row["variant"], int(row["run"]))
                last[key] = float(row["best_Z"]); count[key] = count.get(key, 0) + 1
        ok = all(key in last and _close(last[key], r["Z"]) and count[key] == r["generations"] + 1
                 for variant in ("random", "greedy_seeded") for r in rec["runs"][variant] for key in [(variant, r["run"])])
        traces.append(ok)
        need(ok, f"history {k}/{Q}: trace does not end at the stored value or has the wrong length")
    rep["allocation_exact_on_best_ga_mixes"] = dict(checked=len(alloc), passed=int(sum(a["equal"] for a in alloc)), details=alloc)
    rep["ga_records_reevaluated"] = dict(checked=len(ga_ok), value_matches=int(sum(ga_ok)), not_above_milp=int(sum(ga_le)))
    rep["ga_convergence_traces"] = dict(scenarios=len(traces), consistent=int(sum(traces)),
                                        runs_reproduced_when_traced=int(sum(v["runs_reproduced"] for v in hist.values())))
    if sample_reruns:
        rep["ga_sample_reruns_with_current_code"] = dict(checked=len(reruns), reproduced=int(sum(r["reproduced"] for r in reruns)),
                                                         details=reruns)
        need(len(reruns) == 2 * EXPECTED["ga_scenarios"] and all(r["reproduced"] for r in reruns), "GA sample re-runs")
    need(len(alloc) == 2 * EXPECTED["ga_scenarios"] and all(a["equal"] for a in alloc), "exact allocation check")
    need(len(ga_ok) == EXPECTED["ga_runs"] and all(ga_ok) and all(ga_le), "GA record re-evaluation")
    need(len(traces) == EXPECTED["ga_scenarios"] and all(traces), "GA convergence traces")

    # efficient fronts
    fronts = load(out / "pareto.json")
    if fronts is not None:
        pts, bad_pts = [], 0
        for c in fronts:
            need(c.get("fingerprint") == fp, f"pareto {c['pool']}/{c['Q']}: fingerprint differs from current data/code")
            inst = data.build(ds, c["pool"], c["Q"], 1.0, use_assemblies=True)
            for p in c["front"]:
                pts.append(p)
                try:
                    x, w = load_plan(inst, p["mix"], p["assemblies"])
                    e = data.evaluate(inst, x, w)
                    if not (_close(e["procurement"], p["procurement"]) and _close(e["stock_used"] + e["assembly_value_used"], p["stock_used_total"])):
                        bad_pts += 1
                except (ValueError, TypeError):
                    bad_pts += 1
        mono = all(all(a["procurement"] <= b["procurement"] + 1e-6 and a["stock_used_total"] <= b["stock_used_total"] + 1e-6
                       for a, b in zip(c["front"], c["front"][1:])) for c in fronts)
        g1 = max(p["stage1"]["mip_gap"] for p in pts); g2 = max(p["stage2"]["mip_gap"] for p in pts)
        s1 = sum(p["stage1"]["status"] == 0 for p in pts); s2 = sum(p["stage2"]["status"] == 0 for p in pts)
        rep["efficient_fronts"] = dict(points=len(pts), stage1_optimal=int(s1), stage2_optimal=int(s2),
                                       max_gap_stage1=g1, max_gap_stage2=g2, monotone=bool(mono),
                                       points_feasible_and_values_match=len(pts) - bad_pts)
        need(len(pts) == EXPECTED["front_points"] and s1 == s2 == len(pts) and max(g1, g2) <= 1e-6 * (1 + 1e-9)
             and mono and bad_pts == 0, "efficient fronts")

    if det is not None:
        rep.update(check_unit_level(ds, det, fronts, failures))

    rep.update(check_extra(out, failures, sample_reruns))
    rep.update(check_stage_a(out, failures))
    rep.update(check_release_data(failures))

    rep["passed"] = not failures
    rep["failures"] = failures
    (out / "verification.json").write_text(json.dumps(rep, indent=1))
    for key, v in rep.items():
        if isinstance(v, dict):
            print(key, {a: b for a, b in v.items() if a != "details"})
    print("ALL CHECKS PASSED" if not failures else "FAILED:\n  " + "\n  ".join(failures))
    return not failures


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--no-reruns", action="store_true", help="skip the 24 GA re-runs from seed (faster)")
    ap.add_argument("--resolve", action="store_true", help="also re-solve the 144 dominance problems (both model variants) and compare optima")
    a = ap.parse_args()
    sys.exit(0 if main(a.out, sample_reruns=not a.no_reruns, resolve=a.resolve) else 1)
