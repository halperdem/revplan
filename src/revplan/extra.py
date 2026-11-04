"""Robustness analyses for Stage B, the main case (python -m revplan.extra <name> | all).

Stage B plans additional production on the residual stock (stock left after the remaining frozen programme).
All exact solves use the unit-level kit formulation (kits.py), which equals the aggregate model on these data.
Assembly-related analyses (recognition vs. re-mixing, availability, leave-one-type-out) belong to Stage A
(stage_a.py), because the programme consumes almost all stocked assemblies.
Outputs: results/extra/<name>.json
  sales_caps     per-model upper bounds from the remaining programme (its full volume, and half of it)
  valuation      valuation-date and imputation checks
  composition    changes in stock composition: high-value components +-50%, 20 random perturbations
  ga_stats       absolute GA gaps, procurement differences and 95% confidence intervals from the stored runs
  ga_budget      (optional, not in "all") both GA variants with an equal budget and population diversity
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from . import data, heuristics, kits, checks, model, experiment
from .experiment import RESULTS, POOLS, QS, SEED_BASE, _mix

OUT = RESULTS / "extra"
CASES = [(9, 20), (9, 130), (25, 130), (35, 150)]          # S1, S2, S3 and the widest pool


def _ev(inst, sol):
    e = data.evaluate(inst, sol["x"], sol["w"])
    return dict(Z=e["Z"], stock_used=e["stock_used"] + e["assembly_value_used"], procurement=e["procurement"],
                assembly_units=e["assembly_units"], status=sol["status"], mix=_mix(inst, sol["x"]))


def _moved(a, b):
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) // 2


def _save(name, obj, out=None):
    out = OUT if out is None else Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(dict(fingerprint=checks.fingerprint("extra"), results=obj), indent=1))
    print("written", name, flush=True)


def sales_caps(out=None):
    ds = data.load(); rows = []
    for k, Q in CASES + [(25, 150)]:
        inst = data.build(ds, k, Q, 1.0, use_assemblies=True)
        base = _ev(inst, kits.solve(inst))
        r = dict(pool=k, Q=Q, uncapped=base)
        px = experiment.proportional(ds, inst)              # programme-proportional mix, best assembly use
        pe = _ev(inst, kits.solve(inst, fixed_x=px))
        r["proportional"] = dict(stock_used=pe["stock_used"], procurement=pe["procurement"])
        for label, f in [("programme_volume", 1.0), ("half_programme_volume", 0.5)]:
            ub = [math.floor(ds.programme[p] * f) for p in inst.pool]
            if sum(ub) < Q:
                r[label] = dict(infeasible=True, capacity=sum(ub)); continue
            e = _ev(inst, kits.solve(inst, x_ub=ub))
            e["Z_change"] = e["Z"] - base["Z"]                  # absolute: Z can be close to zero in Stage B
            e["procurement_change"] = e["procurement"] - base["procurement"]
            e["units_moved"] = _moved(base["mix"], e["mix"])
            e["binding_caps"] = int(sum(e["mix"].get(p, 0) >= u for p, u in zip(inst.pool, ub) if u > 0))
            # least procurement under the caps with at least the proportional mix's stock use (iso-stock comparison)
            e["proportional_within_caps"] = bool(all(x <= u for x, u in zip(px, ub)))
            dP = kits.solve(inst, lam=1.0, stock_weight=0.0, x_ub=ub, stock_floor=pe["stock_used"] * (1 - 1e-9))
            ed = _ev(inst, dP)
            e["iso_stock"] = dict(stock_used=ed["stock_used"], procurement=ed["procurement"], status=ed["status"], mix=ed["mix"],
                                  procurement_saving_pct=100.0 * (1.0 - ed["procurement"] / pe["procurement"]))
            r[label] = e
        rows.append(r)
    _save("sales_caps", rows, out)


def valuation(out=None):
    ds = data.load(); rows = []
    for k, Q in CASES:
        base = None
        for col, imp in [("unit_value", 1.0), ("unit_value_alt_20231229", 1.0), ("unit_value_alt_20240131", 1.0),
                         ("unit_value", 0.5), ("unit_value", 2.0)]:
            inst = data.build(ds, k, Q, 1.0, use_assemblies=True, value_column=col, imputed_factor=imp)
            e = _ev(inst, kits.solve(inst))
            base = base or e
            rows.append(dict(pool=k, Q=Q, value_column=col, imputed_factor=imp, Z=e["Z"], procurement=e["procurement"],
                             units_moved_vs_base=_moved(base["mix"], e["mix"]),
                             assembly_units=e["assembly_units"]))
    _save("valuation", rows, out)


def composition(reps: int = 20, sigma: float = 0.5, out=None):
    ds = data.load(); rows = []
    for k, Q in [(25, 130), (35, 150)]:
        inst0 = data.build(ds, k, Q, 1.0, use_assemblies=True)
        base = _ev(inst0, kits.solve(inst0))
        val = inst0.v * inst0.s
        top = val >= np.quantile(val[val > 0], 0.9)            # top decile of stock value
        for f in (0.5, 1.5):
            inst = copy.deepcopy(inst0); inst.s = np.where(top, inst0.s * f, inst0.s)
            e = _ev(inst, kits.solve(inst))
            rows.append(dict(pool=k, Q=Q, perturbation=f"top-decile stock x{f}", Z=e["Z"], procurement=e["procurement"],
                             Z_change=e["Z"] - base["Z"], units_moved=_moved(base["mix"], e["mix"])))
        rng = np.random.default_rng(SEED_BASE)
        for r in range(reps):                                  # mean-preserving lognormal noise per component
            inst = copy.deepcopy(inst0)
            inst.s = inst0.s * rng.lognormal(-sigma ** 2 / 2, sigma, size=len(inst0.s))
            e = _ev(inst, kits.solve(inst))
            rows.append(dict(pool=k, Q=Q, perturbation=f"random lognormal sigma={sigma} #{r + 1}", Z=e["Z"],
                             procurement=e["procurement"], Z_change=e["Z"] - base["Z"],
                             units_moved=_moved(base["mix"], e["mix"])))
    _save("composition", rows, out)


def _ci(xs):
    m = st.mean(xs)
    if len(xs) < 2:
        return m, m, m
    h = 2.045 * st.stdev(xs) / math.sqrt(len(xs))           # t(0.975, 29) for 30 runs
    return m, m - h, m + h


def ga_stats(out=None):
    ds = data.load(); det = json.loads((RESULTS / "deterministic.json").read_text())
    zopt = {(r["pool"], r["Q"]): r["with_assemblies"] for r in det["scenarios"] if r["stock_factor"] == 1.0}
    rows = []
    for k in POOLS:
        for Q in QS:
            rec = json.loads((RESULTS / "ga" / f"ga_{k}_{Q}.json").read_text())
            inst = data.build(ds, k, Q, 1.0, use_assemblies=True)
            dec = heuristics.Decoder(inst)
            opt = zopt[(k, Q)]["milp"]; gr = zopt[(k, Q)]["greedy"]
            r = dict(pool=k, Q=Q, Z_opt=opt["Z"], greedy_abs_gap=opt["Z"] - gr["Z"],
                     greedy_procurement_diff=gr["procurement"] - opt["procurement"])
            for variant, runs in rec["runs"].items():
                gaps, pdiff = [], []
                for run in runs:
                    x = np.zeros(len(inst.pool), dtype=int)
                    for p, q in run["mix"].items():
                        x[inst.pool.index(p)] = q
                    W = dec.allocate(x)
                    d = inst.A @ x - dec.Bmat @ W
                    u = np.minimum(d, inst.s)
                    proc = float(inst.v @ (d - u))
                    gaps.append(opt["Z"] - run["Z"]); pdiff.append(proc - opt["procurement"])
                m, lo, hi = _ci(gaps); pm, plo, phi = _ci(pdiff)
                r[variant] = dict(abs_gap_mean=m, abs_gap_ci95=[lo, hi], abs_gap_max=max(gaps),
                                  procurement_diff_mean=pm, procurement_diff_ci95=[plo, phi])
            rows.append(r)
    _save("ga_stats", rows, out)


def _ga_div(inst, seed, seed_x=None, population=100, generations=1000, crossover=0.8, mutation=0.05, elites=2,
            checkpoints=(0, 10, 25, 50, 100, 250, 500, 750, 1000)):
    """Same operators as heuristics.ga, fixed number of generations (no early stop), records population diversity.

    Diversity = mean pairwise L1 distance between chromosomes divided by 2Q (0 = identical, 1 = disjoint plans).
    """
    dec = heuristics.Decoder(inst)
    n, Q = len(inst.pool), inst.Q
    rng = np.random.default_rng(seed)
    X = np.array([rng.multinomial(Q, rng.dirichlet(np.ones(n))) for _ in range(population)])
    if seed_x is not None:
        X[0] = seed_x
    trace = {}
    t0 = time.perf_counter()
    for gen in range(generations + 1):
        f = dec.score(X)
        order = np.argsort(-f, kind="stable")
        if gen in checkpoints:
            D = np.abs(X[:, None, :] - X[None, :, :]).sum(-1)
            trace[gen] = dict(best=float(f[order[0]]), diversity=float(D.sum() / (population * (population - 1)) / (2 * Q)),
                              distinct=int(len({r.tobytes() for r in X})))
        if gen == generations:
            break
        wts = f - f.min()
        wts += max(float(wts.max()), 1.0) * 1e-12
        par = rng.choice(population, size=(population - elites, 2), p=wts / wts.sum())
        ch = X[par[:, 0]].copy()
        nc = len(ch)
        if n > 1:
            cuts = rng.integers(1, n, size=nc)
            cr = rng.random(nc) < crossover
            mask = (np.arange(n)[None, :] >= cuts[:, None]) & cr[:, None]
            ch = np.where(mask, X[par[:, 1]], ch)
        for r in np.flatnonzero(ch.sum(1) != Q):
            tot = int(ch[r].sum())
            if tot == 0:
                ch[r] = rng.multinomial(Q, np.ones(n) / n); continue
            sc = ch[r] * Q / tot
            fx = np.floor(sc).astype(int)
            left = Q - int(fx.sum())
            fr = sc - fx
            if left:
                fx[rng.choice(n, left, replace=False, p=fr / fr.sum())] += 1
            ch[r] = fx
        for j in range(n):
            sel = np.flatnonzero((rng.random(nc) < mutation) & (ch[:, j] > 0))
            if not len(sel):
                continue
            dest = rng.integers(n, size=len(sel))
            lim = np.minimum(ch[sel, j], max(1, Q // 10))
            amt = (rng.random(len(sel)) * lim).astype(int) + 1
            ch[sel, j] -= amt
            ch[sel, dest] += amt
        X = np.vstack([X[order[:elites]], ch])
    return dict(trace=trace, Z=trace[generations]["best"], evaluations=dec.evals, seconds=time.perf_counter() - t0)


BUDGET_CASES = [(25, 130), (25, 150), (35, 130), (35, 150)]


def _budget_job(args):
    k, Q, variant, run = args
    f = OUT / "ga_budget_parts" / f"{k}_{Q}_{variant}_{run}.json"
    fp = checks.fingerprint("extra")
    if f.exists():                                     # resume only from runs made with the same data and code
        if json.loads(f.read_text()).get("fingerprint") != fp:
            raise RuntimeError(f"{f} was produced with different data or code; delete ga_budget_parts or use a fresh run")
        return
    ds = data.load()
    inst = data.build(ds, k, Q, 1.0, use_assemblies=True)
    idx = [(kk, qq) for kk in POOLS for qq in QS].index((k, Q))
    seed = SEED_BASE + 1000 * idx + run
    sx = heuristics.greedy(inst)[0] if variant == "greedy_seeded" else None
    g = _ga_div(inst, seed, seed_x=sx)
    f.write_text(json.dumps(dict(pool=k, Q=Q, variant=variant, run=run + 1, seed=seed, fingerprint=fp, **g)))


def ga_budget(workers: int = 2, runs: int = 30):
    (OUT / "ga_budget_parts").mkdir(parents=True, exist_ok=True)
    jobs = [(k, Q, v, r) for k, Q in BUDGET_CASES for v in ("random", "greedy_seeded") for r in range(runs)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_budget_job, jobs))
    ga_budget_summary(OUT / "ga_budget_parts", runs=runs)


def _holm(ps):
    """Holm (1979) step-down adjusted p-values, in the original order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def ga_budget_summary(parts_dir, runs: int = 30, out=None):
    """Aggregate the equal-budget runs; paired Wilcoxon tests with Holm adjustment over the four scenarios."""
    from scipy.stats import wilcoxon
    parts_dir = Path(parts_dir)
    det = json.loads((RESULTS / "deterministic.json").read_text())
    zopt = {(r["pool"], r["Q"]): r["with_assemblies"]["milp"]["Z"] for r in det["scenarios"] if r["stock_factor"] == 1.0}
    # gaps are normalised by the material value of the optimal plan (stock use + procurement), because Z itself
    # can be close to zero or negative in Stage B
    scale = {(r["pool"], r["Q"]): r["with_assemblies"]["milp"]["stock_used"] + r["with_assemblies"]["milp"]["assembly_value_used"]
             + r["with_assemblies"]["milp"]["procurement"] for r in det["scenarios"] if r["stock_factor"] == 1.0}
    rows = []
    for k, Q in BUDGET_CASES:
        r = dict(pool=k, Q=Q)
        P = {v: [json.loads((parts_dir / f"{k}_{Q}_{v}_{i}.json").read_text()) for i in range(runs)]
             for v in ("random", "greedy_seeded")}
        for v, parts in P.items():
            gaps = [100 * (zopt[(k, Q)] - p["Z"]) / scale[(k, Q)] for p in parts]
            m, lo, hi = _ci(gaps)
            cps = sorted(parts[0]["trace"], key=int)
            r[v] = dict(gap_pct_mean=m, gap_pct_ci95=[lo, hi], at_optimum=int(sum(g < 1e-9 for g in gaps)),
                        evaluations_mean=st.mean(p["evaluations"] for p in parts),
                        diversity={c: st.mean(p["trace"][c]["diversity"] for p in parts) for c in cps},
                        gap_pct_by_generation={c: st.mean(100 * (zopt[(k, Q)] - p["trace"][c]["best"]) / scale[(k, Q)] for p in parts) for c in cps})
        a = [p["Z"] for p in P["random"]]; b = [p["Z"] for p in P["greedy_seeded"]]
        r["paired_wilcoxon_p"] = float(wilcoxon(a, b).pvalue)
        r["seeded_better_runs"] = int(sum(y > x + 1e-9 for x, y in zip(a, b)))
        r["random_better_runs"] = int(sum(x > y + 1e-9 for x, y in zip(a, b)))
        rows.append(r)
    for r, h in zip(rows, _holm([r["paired_wilcoxon_p"] for r in rows])):
        r["paired_wilcoxon_p_holm"] = h
    _save("ga_budget", rows, out)
    return rows


# ga_budget (equal-budget GA with diversity) is kept for reference but not part of "all": in Stage B the greedy plan is
# optimal in every scenario, so a greedy-seeded GA starts at the optimum and the comparison has no content.
# ------------------------------------------------------------------ netting rules and stock corrections
NETTING_VARIANTS = [("bs_entry", (), "plan"), ("bo_entry", (), "plan"), ("bs_exit", (), "plan"),
                    ("bs_entry", ("unposted_backflush",), "plan"), ("bs_entry", (), "verified")]


def netting_sensitivity(out=None):
    """Re-run the key results of both stages under other netting rules (revplan.netting) and with the unposted
    back-flush quantities of the days before the cut-off deducted from the residual stock."""
    from . import netting, stage_a
    rows = []
    for rule, deduct, pairs in NETTING_VARIANTS:
        g, r = netting.dataset(rule=rule, deduct=deduct, pairs=pairs)
        row = dict(rule=rule, deduct=list(deduct), pairs=pairs, programme_units=int(sum(g.programme.values())),
                   programme_models=sum(1 for p in g.products if g.programme[p] > 0),
                   residual_value=float(sum(r.stock[c] * r.values["unit_value"][c] for c in r.stock)),
                   residual_assembly_units=int(sum(r.assembly_stock.values())))
        if not deduct and pairs == "plan":               # Stage A depends on the programme, not on deductions or allocation
            sa = {}
            for use in (False, True):
                _, inst, q = stage_a.instance(g, use)
                for f in (1.0, None):
                    lb, ub = stage_a.band(q, f)
                    sol = kits.solve(inst, x_lb=lb, x_ub=ub); e = _ev(inst, sol)
                    sa[f"{'with' if use else 'without'}_assemblies_band_{f}"] = dict(
                        procurement=e["procurement"], stock_used=e["stock_used"], Z=e["Z"], assembly_units=e["assembly_units"], status=sol["status"])
            row["stage_a"] = sa
        nm = sum(1 for p in r.products if r.programme[p] > 0)
        sb = []
        for k, Q in CASES:
            k = min(k, nm)
            inst = data.build(r, k, Q, 1.0, use_assemblies=True)
            sol = model.solve(inst); e = _ev(inst, sol)
            gx, gw = heuristics.greedy(inst); ge = data.evaluate(inst, gx, gw)
            px = experiment.proportional(r, inst); ps = model.solve(inst, fixed_x=px); pe = _ev(inst, ps)
            dP = model.solve(inst, lam=1.0, stock_weight=0.0, stock_floor=pe["stock_used"] * (1 - 1e-9)); de = _ev(inst, dP)
            sb.append(dict(pool=k, Q=Q, Z=e["Z"], stock_used=e["stock_used"], procurement=e["procurement"], status=sol["status"],
                           greedy_abs_gap=e["Z"] - ge["Z"], proportional_stock_used=pe["stock_used"], proportional_procurement=pe["procurement"],
                           min_procurement_at_proportional_stock_use=de["procurement"], dominance_status=dP["status"],
                           saving_pct=100 * (1 - de["procurement"] / pe["procurement"])))
        row["stage_b"] = sb
        rows.append(row)
        print("netting", rule, deduct, pairs, flush=True)
    _save("netting_sensitivity", rows, out)


def pair_diagnostics(out=None):
    """Model-assembly pairs named in the plan rows of the remaining programme that the containment rule does not
    infer: how many of the assembly's components the model's BOM lacks (or holds in a smaller quantity), their value
    share, and how many assembly units the main netting allocates through the pair."""
    from . import netting
    from collections import Counter
    g = data.load(stock="gross"); v = g.values["unit_value"]
    units = netting.read_units()
    n = netting.net(g.bom, g.assembly_bom, g.stock, g.assembly_stock, units, netting.MAIN_RULE)
    planned = Counter((u["product_id"], y) for u in units if netting.remaining(u, netting.MAIN_RULE) for y in u["assemblies"])
    alloc = Counter((u["product_id"], y) for u in units if u["unit_id"] in n.allocation for y in n.allocation[u["unit_id"]])
    rows = []
    for (p, y), k in sorted(planned.items()):
        A, B = g.assembly_bom[y], g.bom[p]
        if netting.fits(B, A):
            continue
        short = {c: q - B.get(c, 0.0) for c, q in A.items() if B.get(c, 0.0) < q - 1e-9}
        val = sum(v[c] * d for c, d in short.items()); tot = sum(v[c] * q for c, q in A.items())
        rows.append(dict(product_id=p, assembly_id=y, kind=g.assembly_kind[y], planned_units=k, allocated_units=alloc[(p, y)],
                         components=len(A), components_short=len(short), value_short=val, value_share_short=val / tot,
                         short=sorted(short)))
    _save("pair_diagnostics", rows, out)


# ------------------------------------------------------------------ plant-facing plan reports
def _report(inst, ds, x, w, name, folder):
    """Mix and part-level view of a plan: stock at start, used, left, bought (quantity and value)."""
    import csv as _csv
    d = inst.A @ np.asarray(x, dtype=float)
    for (j, y), k in w.items():
        d = d - inst.Bm[y] * k
    u = np.minimum(d, inst.s); b = d - u
    imp = {r["component_id"]: r for r in _csv.DictReader(open(data.DATA_DIR / "components.csv", encoding="utf-8"))}
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{name}_mix.csv", "w", newline="", encoding="utf-8") as f:
        wr = _csv.writer(f); wr.writerow(["product_id", "units", "assemblies_installed"])
        for j, p in enumerate(inst.pool):
            if x[j]:
                wr.writerow([p, int(x[j]), ";".join(f"{y}x{k}" for (jj, y), k in sorted(w.items()) if jj == j and k)])
    rows = []
    for i, c in enumerate(inst.components):
        if d[i] > 0 or inst.s[i] > 0:
            rows.append([c, imp[c]["production_impact"], inst.s[i], d[i], u[i], inst.s[i] - u[i], b[i], b[i] * inst.v[i]])
    rows.sort(key=lambda r: -r[7])
    with open(folder / f"{name}_parts.csv", "w", newline="", encoding="utf-8") as f:
        wr = _csv.writer(f); wr.writerow(["component_id", "production_impact", "stock_start", "requirement", "from_stock", "stock_left",
                                          "to_buy_qty", "to_buy_value"])
        wr.writerows([[r[0], r[1]] + [repr(float(v)) for v in r[2:8]] for r in rows])
    bought = [r for r in rows if r[6] > 1e-9]
    ev = data.evaluate(inst, x, w)
    return dict(name=name, units=int(np.sum(x)), models=int(np.sum(np.asarray(x) > 0)), stock_used=ev["stock_used"] + ev["assembly_value_used"],
                procurement=ev["procurement"], parts_required=int(np.sum(d > 1e-9)), parts_to_buy=len(bought),
                critical_parts_to_buy=sum(1 for r in bought if r[1] == "critical"),
                may_run_short_parts_to_buy=sum(1 for r in bought if r[1] == "may_run_short"),
                unclassified_parts_to_buy=sum(1 for r in bought if not r[1]),          # not on the plant's impact list
                value_to_buy_unclassified=float(sum(r[7] for r in bought if not r[1])),
                value_to_buy_critical=float(sum(r[7] for r in bought if r[1] == "critical")),
                top10_value_share=float(sum(r[7] for r in bought[:10]) / max(ev["procurement"], 1e-12)))


def plans_report(out=None):
    from . import stage_a
    folder = (OUT if out is None else Path(out)) / "plans"
    res = []
    g = data.load(stock="gross")
    _, inst, q = stage_a.instance(g, True)
    for f in (1.0, 1.1):
        lb, ub = stage_a.band(q, f)
        sol = kits.solve(inst, x_lb=lb, x_ub=ub)
        res.append(dict(stage="A", band=f, **_report(inst, g, sol["x"], sol["w"], f"stageA_band{f}", folder)))
    r = data.load()
    for k, Q in [(25, 130), (35, 150)]:
        inst = data.build(r, k, Q, 1.0, use_assemblies=True)
        sol = model.solve(inst)
        res.append(dict(stage="B", pool=k, Q=Q, plan="optimal", **_report(inst, r, sol["x"], sol["w"], f"stageB_{k}_{Q}_optimal", folder)))
        px = experiment.proportional(r, inst); ps = model.solve(inst, fixed_x=px)
        res.append(dict(stage="B", pool=k, Q=Q, plan="proportional", **_report(inst, r, ps["x"], ps["w"], f"stageB_{k}_{Q}_proportional", folder)))
        pe = data.evaluate(inst, ps["x"], ps["w"])                # same stock use as the proportional mix, least procurement
        dp = model.solve(inst, lam=1.0, stock_weight=0.0, stock_floor=(pe["stock_used"] + pe["assembly_value_used"]) * (1 - 1e-9))
        res.append(dict(stage="B", pool=k, Q=Q, plan="least_procurement_at_proportional_stock_use",
                        **_report(inst, r, dp["x"], dp["w"], f"stageB_{k}_{Q}_least_procurement_same_stock_use", folder)))
    _save("plans_report", res, out)


# ------------------------------------------------------------------ where simple rules start to fail
def composition_limits(reps: int = 40, ga_instances: int = 10, ga_runs: int = 5, out=None):
    """Stress the residual stock where models compete: scale the stock of components shared by at least half of the
    candidate models by U(0, 1) factors, or perturb every component lognormally (sigma = 1). Greedy and GA are
    compared with the MILP on every instance (GA on the first instances only)."""
    r = data.load(); rows = []
    for k, Q in [(25, 130), (35, 150)]:
        base = data.build(r, k, Q, 1.0, use_assemblies=True)
        shared = (base.A > 0).sum(1) >= 0.5 * len(base.pool)
        for family in ("shared_scarcity", "lognormal_all"):
            rng = np.random.default_rng(SEED_BASE + 7 * k + Q + (0 if family == "shared_scarcity" else 1))
            for rep in range(reps):
                inst = copy.deepcopy(base)
                if family == "shared_scarcity":
                    inst.s = np.where(shared, base.s * rng.random(len(base.s)), base.s)
                else:
                    inst.s = base.s * rng.lognormal(-0.5, 1.0, size=len(base.s))
                sol = model.solve(inst); e = data.evaluate(inst, sol["x"], sol["w"])
                gx, gw = heuristics.greedy(inst); ge = data.evaluate(inst, gx, gw)
                scale_ = e["stock_used"] + e["assembly_value_used"] + e["procurement"]
                row = dict(pool=k, Q=Q, family=family, rep=rep + 1, Z_opt=e["Z"], status=sol["status"],
                           greedy_gap_pct=100 * (e["Z"] - ge["Z"]) / scale_, greedy_optimal=bool(e["Z"] - ge["Z"] <= 1e-9 * scale_))
                if rep < ga_instances:
                    zs = [heuristics.ga(inst, SEED_BASE + 100000 + 1000 * rep + i)["Z"] for i in range(ga_runs)]
                    row["ga_gap_pct"] = [100 * (e["Z"] - z) / scale_ for z in zs]
                rows.append(row)
            print("limits", k, Q, family, flush=True)
    summ = []
    for k, Q in [(25, 130), (35, 150)]:
        for family in ("shared_scarcity", "lognormal_all"):
            rr = [x for x in rows if (x["pool"], x["Q"], x["family"]) == (k, Q, family)]
            gg = [x["greedy_gap_pct"] for x in rr]; ga = [v for x in rr if "ga_gap_pct" in x for v in x["ga_gap_pct"]]
            summ.append(dict(pool=k, Q=Q, family=family, instances=len(rr), greedy_optimal=sum(x["greedy_optimal"] for x in rr),
                             greedy_gap_mean=st.mean(gg), greedy_gap_max=max(gg), ga_runs=len(ga),
                             ga_at_optimum=sum(v <= 1e-7 for v in ga), ga_gap_mean=st.mean(ga) if ga else None, ga_gap_max=max(ga) if ga else None))
    _save("composition_limits", dict(instances=rows, summary=summ), out)


ALL = dict(sales_caps=sales_caps, valuation=valuation, composition=composition, ga_stats=ga_stats,
           netting_sensitivity=netting_sensitivity, plans_report=plans_report, composition_limits=composition_limits,
           pair_diagnostics=pair_diagnostics)
OPTIONAL = dict(ga_budget=ga_budget)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=list(ALL) + list(OPTIONAL) + ["all", "fast"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    names = list(ALL) if a.what in ("all", "fast") else [a.what]
    for nme in names:
        OPTIONAL[nme](a.workers) if nme in OPTIONAL else ALL[nme]()
