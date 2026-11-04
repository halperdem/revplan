"""Reproduce the paper's experiments.

Stage B (main case): stock left after the remaining frozen programme (data.load default "residual").

python -m revplan.experiment deterministic   # MILP + greedy + programme-proportional mix, 36 scenarios x 2 models, lambda, valuation date
python -m revplan.experiment pareto            # stock use vs procurement front (epsilon-constraint)
python -m revplan.experiment ga --workers 2  # GA + greedy-seeded GA, 12 base-stock scenarios x 30 runs
python -m revplan.experiment ga-history       # re-run all GA seeds, full convergence traces, reproduction check
"""
from __future__ import annotations
import argparse
import json
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from . import data, model, heuristics, checks

POOLS = [9, 15, 25, 35]               # 35 = every model of the remaining programme (main netting rule)
QS = [20, 130, 150]
STOCK = [0.8, 1.0, 1.2]
LAMBDAS = [0.5, 2.0]
SEED_BASE = 20261022
RESULTS = Path(__file__).resolve().parents[2] / "results"


def proportional(ds, inst):
    """Baseline mix: Q units split over the pool in proportion to the remaining programme (largest remainder)."""
    q = np.array([ds.programme[p] for p in inst.pool], dtype=float)
    share = inst.Q * q / q.sum()
    x = np.floor(share).astype(int)
    for j in np.argsort(-(share - x), kind="stable")[: inst.Q - x.sum()]:
        x[j] += 1
    return x


def _mix(inst, x):
    return {inst.pool[j]: int(q) for j, q in enumerate(x) if q}


def run_deterministic(out: Path = RESULTS):
    ds = data.load()
    rows = []
    for k in POOLS:
        for Q in QS:
            for sf in STOCK:
                row = dict(pool=k, Q=Q, stock_factor=sf)
                for label, use in [("components_only", False), ("with_assemblies", True)]:
                    inst = data.build(ds, k, Q, sf, use_assemblies=use)
                    sol = model.solve(inst)
                    ev = data.evaluate(inst, sol["x"], sol["w"])
                    times = []
                    for _ in range(3):
                        t = time.perf_counter(); gx, gw = heuristics.greedy(inst); times.append(time.perf_counter() - t)
                    gev = data.evaluate(inst, gx, gw)
                    checks.check_assignment(inst, sol["x"], sol["w"]); checks.check_assignment(inst, gx, gw)
                    px = proportional(ds, inst)
                    ps = model.solve(inst, fixed_x=px)              # best assembly use for the proportional mix
                    checks.check_assignment(inst, ps["x"], ps["w"])
                    pev = data.evaluate(inst, ps["x"], ps["w"])
                    pu = pev["stock_used"] + pev["assembly_value_used"]
                    # dominance check: least procurement with at least the proportional mix's stock use, and
                    # most stock use with at most its procurement
                    dP = model.solve(inst, lam=1.0, stock_weight=0.0, stock_floor=pu * (1 - 1e-9))
                    dU = model.solve(inst, lam=0.0, stock_weight=1.0, procurement_cap=pev["procurement"] * (1 + 1e-9))
                    eP, eU = data.evaluate(inst, dP["x"], dP["w"]), data.evaluate(inst, dU["x"], dU["w"])
                    checks.check_assignment(inst, dP["x"], dP["w"]); checks.check_assignment(inst, dU["x"], dU["w"])
                    pev["min_procurement_at_same_stock_use"] = eP["procurement"]
                    pev["max_stock_use_at_same_procurement"] = eU["stock_used"] + eU["assembly_value_used"]
                    pev["dominance_status"] = [dP["status"], dU["status"]]
                    # the two witness plans themselves, so that verify can recompute them
                    pev["dominance_plans"] = {
                        name: dict(mix=_mix(inst, s_["x"]), assemblies=_w_list(inst, s_["w"]), stock_used=e_["stock_used"] + e_["assembly_value_used"],
                                   procurement=e_["procurement"], status=s_["status"], mip_gap=s_["mip_gap"])
                        for name, s_, e_ in (("min_procurement_at_same_stock_use", dP, eP), ("max_stock_use_at_same_procurement", dU, eU))}
                    r = dict(proportional=dict(**pev, status=ps["status"], mix=_mix(inst, ps["x"]), assemblies=_w_list(inst, ps["w"])),
                             milp=dict(**ev, seconds=sol["seconds"], status=sol["status"], mip_gap=sol["mip_gap"], mix=_mix(inst, sol["x"]),
                                       assemblies=_w_list(inst, sol["w"])),
                             greedy=dict(**gev, seconds=st.median(times), mix=_mix(inst, gx), assemblies=_w_list(inst, gw)))
                    if use and sf == 1.0:
                        r["lambda"] = {}
                        for lam in LAMBDAS:
                            sl = model.solve(inst, lam=lam)
                            e = data.evaluate(inst, sl["x"], sl["w"])
                            r["lambda"][str(lam)] = dict(**e, units_moved=int(np.abs(sl["x"] - sol["x"]).sum() // 2), status=sl["status"],
                                                     mip_gap=sl["mip_gap"], mix=_mix(inst, sl["x"]), assemblies=_w_list(inst, sl["w"]))
                    row[label] = r
                rows.append(row)
                print(k, Q, sf, "Z(with assemblies) = %.3f" % row["with_assemblies"]["milp"]["Z"], flush=True)
    # valuation-date check (components-only model, S1-S3, base stock)
    val = []
    for k, Q in [(9, 20), (9, 130), (25, 130)]:
        base = None
        for col, imp in [("unit_value", 1.0), ("unit_value_alt_20231229", 1.0), ("unit_value_alt_20240131", 1.0), ("unit_value", 0.5), ("unit_value", 2.0)]:
            inst = data.build(ds, k, Q, 1.0, use_assemblies=False, value_column=col, imputed_factor=imp)
            sol = model.solve(inst)
            base = sol["x"] if base is None else base
            gx, gw = heuristics.greedy(inst)
            val.append(dict(pool=k, Q=Q, value_column=col, imputed_factor=imp, milp_Z=data.evaluate(inst, sol["x"], {})["Z"],
                            greedy_Z=data.evaluate(inst, gx, {})["Z"], units_moved_vs_base=int(np.abs(sol["x"] - base).sum() // 2)))
    out.mkdir(parents=True, exist_ok=True)
    (out / "deterministic.json").write_text(json.dumps(dict(fingerprint=checks.fingerprint(), scenarios=rows, valuation_check=val), indent=1))


PARETO_CASES = [(9, 20), (9, 130), (25, 130), (9, 150), (15, 150), (25, 150), (35, 150)]


def _w_list(inst, w):
    return [[inst.pool[j], y, int(k)] for (j, y), k in sorted(w.items())]


def pareto_front(inst, points: int = 15, gap: float = 1e-6, time_limit: float = 600.0) -> list:
    """Epsilon-constraint front with lexicographic (two-stage) solves; no small tie-break weights.

    Every point: stage 1 optimises the primary criterion, stage 2 optimises the secondary one with the
    primary fixed at its stage-1 value (absolute tolerance 1e-9 * value). Interior points maximise stock use
    subject to procurement <= eps and then minimise procurement, so each point is efficient, not only
    weakly efficient. Status and relative MIP gap are stored for both stages.
    """
    def tol(z):
        return 1e-9 * max(1.0, abs(z))

    def pack(sol, s1, kind, eps):
        e = data.evaluate(inst, sol["x"], sol["w"])
        checks.check_assignment(inst, sol["x"], sol["w"])
        return dict(eps=eps, kind=kind, stock_used_total=e["stock_used"] + e["assembly_value_used"],
                    procurement=e["procurement"], Z=e["Z"], mix=_mix(inst, sol["x"]), assemblies=_w_list(inst, sol["w"]),
                    stage1=dict(status=s1["status"], mip_gap=s1["mip_gap"], seconds=s1["seconds"]),
                    stage2=dict(status=sol["status"], mip_gap=sol["mip_gap"], seconds=sol["seconds"]))

    def stock_of(sol):
        e = data.evaluate(inst, sol["x"], sol["w"]); return e["stock_used"] + e["assembly_value_used"]

    def proc_of(sol):
        return data.evaluate(inst, sol["x"], sol["w"])["procurement"]

    opt = dict(mip_rel_gap=gap, time_limit=time_limit)
    # minimum procurement, then maximum stock use at that procurement
    s1 = model.solve(inst, lam=1.0, stock_weight=0.0, **opt)
    p1 = proc_of(s1)
    s2 = model.solve(inst, lam=0.0, stock_weight=1.0, procurement_cap=p1 + tol(p1), **opt)
    lo = pack(s2, s1, "min_procurement", None)

    def max_stock(cap):
        a = model.solve(inst, lam=0.0, stock_weight=1.0, procurement_cap=cap, **opt)
        u = stock_of(a)
        b = model.solve(inst, lam=1.0, stock_weight=0.0, procurement_cap=cap, stock_floor=u - tol(u), **opt)
        return b, a

    b, a = max_stock(None)
    hi = pack(b, a, "max_stock_use", None)
    front = [lo]
    for t in np.linspace(0, 1, points)[1:-1]:
        eps = lo["procurement"] + t * (hi["procurement"] - lo["procurement"])
        b, a = max_stock(eps)
        front.append(pack(b, a, "interior", eps))
    front.append(hi)
    return front


def run_pareto(out: Path = RESULTS):
    ds = data.load()
    res = []
    for k, Q in PARETO_CASES:
        inst = data.build(ds, k, Q, 1.0, use_assemblies=True)
        res.append(dict(pool=k, Q=Q, fingerprint=checks.fingerprint(), front=pareto_front(inst)))
        print("Pareto", k, Q, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pareto.json").write_text(json.dumps(res, indent=1))


def _stale(path: Path, fp: str) -> bool:
    """True if a stored result exists but was produced with other data or code."""
    try:
        return json.loads(path.read_text()).get("fingerprint") != fp
    except (ValueError, AttributeError):
        return True


def _ga_scenario(args):
    idx, k, Q, runs, out, force = args
    f = Path(out, f"ga_{k}_{Q}.json")
    fp = checks.fingerprint()
    if f.exists() and not force:                       # resume only from results of the same data and code
        if _stale(f, fp):
            raise RuntimeError(f"{f} was produced with different data or code; delete it or use --force")
        return k, Q
    ds = data.load()
    inst = data.build(ds, k, Q, 1.0, use_assemblies=True)
    gx, _ = heuristics.greedy(inst)
    res = dict(pool=k, Q=Q, stock_factor=1.0, fingerprint=fp, runs={})
    for variant, sx in [("random", None), ("greedy_seeded", gx)]:
        rr = []
        for r in range(runs):
            seed = SEED_BASE + 1000 * idx + r
            g = heuristics.ga(inst, seed, seed_x=sx)
            h = g["history"]
            rr.append(dict(run=r + 1, seed=seed, Z=g["Z"], generations=g["generations"], seconds=g["seconds"],
                           evaluations=g["evaluations"], mix=_mix(inst, g["x"]),
                           best_at_generation={str(t): h[min(t, len(h) - 1)] for t in (0, 50, 100, 200, 400, 800, 1200, 1600, 2000)}))
        res["runs"][variant] = rr
    Path(out, f"ga_{k}_{Q}.json").write_text(json.dumps(res, indent=1))
    return k, Q


def run_ga(out: Path = RESULTS, workers: int = 1, runs: int = 30, only=None, force: bool = False):
    gdir = out / "ga"
    gdir.mkdir(parents=True, exist_ok=True)
    grid = [(i, k, Q, runs, str(gdir), force) for i, (k, Q) in enumerate((k, Q) for k in POOLS for Q in QS)
            if only is None or (k, Q) in only]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, Q in ex.map(_ga_scenario, grid):
            print("GA done", k, Q, flush=True)


def _ga_history(args):
    """Re-run every stored GA run from its seed, write the full best-so-far trajectory and check reproduction."""
    idx, k, Q, gdir = args
    import gzip, csv
    rec = json.loads(Path(gdir, f"ga_{k}_{Q}.json").read_text())
    ds = data.load()
    inst = data.build(ds, k, Q, 1.0, use_assemblies=True)
    gx, _ = heuristics.greedy(inst)
    bad = []
    tmp = Path(gdir, f"history_{k}_{Q}.csv.gz.part")
    with gzip.open(tmp, "wt", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["variant", "run", "generation", "best_Z"])
        for variant, sx in [("random", None), ("greedy_seeded", gx)]:
            for r in rec["runs"][variant]:
                g = heuristics.ga(inst, r["seed"], seed_x=sx)
                for t, z in enumerate(g["history"]):
                    wr.writerow([variant, r["run"], t, repr(z)])
                if abs(g["Z"] - r["Z"]) > 1e-9 * max(1.0, abs(r["Z"])) or _mix(inst, g["x"]) != r["mix"] or g["generations"] != r["generations"]:
                    bad.append(dict(variant=variant, run=r["run"]))
    tmp.replace(Path(gdir, f"history_{k}_{Q}.csv.gz"))          # only complete files count as done
    return k, Q, bad


def run_ga_history(out: Path = RESULTS, workers: int = 1, force: bool = False):
    gdir = out / "ga"
    fp = checks.fingerprint()
    chk = Path(gdir, "history_check.json")
    done = json.loads(chk.read_text()) if chk.exists() else {}
    if force:
        done = {}
    for key, v in list(done.items()):                  # keep only traces made with the same data and code
        if v.get("fingerprint") != fp:
            raise RuntimeError(f"history {key} was produced with different data or code; use --force")
    grid = [(i, k, Q, str(gdir)) for i, (k, Q) in enumerate((k, Q) for k in POOLS for Q in QS)
            if f"{k}_{Q}" not in done or not Path(gdir, f"history_{k}_{Q}.csv.gz").exists()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, Q, bad in ex.map(_ga_history, grid):
            done[f"{k}_{Q}"] = dict(runs_reproduced=60 - len(bad), mismatches=bad, fingerprint=fp)
            chk.write_text(json.dumps(done, indent=1))
            print("history", k, Q, "mismatches", len(bad), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["deterministic", "pareto", "ga", "ga-history", "all"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--force", action="store_true", help="recompute stored GA results even if they exist")
    a = ap.parse_args()
    if a.what in ("deterministic", "all"):
        run_deterministic(a.out)
    if a.what in ("pareto", "all"):
        run_pareto(a.out)
    if a.what in ("ga", "all"):
        run_ga(a.out, a.workers, a.runs, force=a.force)
    if a.what in ("ga-history", "all"):
        run_ga_history(a.out, a.workers, force=a.force)
