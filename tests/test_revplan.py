"""Small synthetic checks of the model and heuristics (python -m pytest tests)."""
import itertools
import numpy as np
from revplan import data, model, heuristics, checks

HAVE_DATA = (data.DATA_DIR / "products.csv").exists()   # the industrial data are not part of the public repository


def needs_data(fn):
    """Tests on the confidential plant data run only where the data folder is present."""
    def wrapper():
        if not HAVE_DATA:
            try:
                import pytest
                pytest.skip("industrial data not available")
            except ImportError:
                print("skipped", fn.__name__, "(industrial data not available)"); return "skipped"
        return fn()
    wrapper.__name__ = fn.__name__
    return wrapper



def toy(use_assemblies=True):
    ds = data.Dataset(
        products=["P1", "P2", "P3"], programme={"P1": 3, "P2": 2, "P3": 1},
        bom={"P1": {"c1": 2, "c2": 1, "c3": 1}, "P2": {"c1": 1, "c3": 2, "c4": 1}, "P3": {"c2": 2, "c4": 1, "c5": 3}},
        stock={"c1": 5, "c2": 1, "c3": 3, "c4": 0, "c5": 4},
        values={"unit_value": {"c1": 1.0, "c2": 4.0, "c3": 2.0, "c4": 3.0, "c5": 0.5}},
        imputed={c: False for c in ["c1", "c2", "c3", "c4", "c5"]},
        assemblies=["A1", "A2"], assembly_kind={"A1": "engine", "A2": "cab"}, assembly_stock={"A1": 2, "A2": 1},
        assembly_bom={"A1": {"c3": 1, "c4": 1}, "A2": {"c2": 1}})
    return ds


def _valid(inst, x, w):
    """Feasibility written out from the model definition, without the MILP or checks.check_assignment."""
    used = {}
    for (j, y), k in w.items():
        if k > x[j]:
            return False
        used[y] = used.get(y, 0) + k
    if any(used[y] > inst.Ys[y] for y in used):
        return False
    for j in range(len(inst.pool)):
        load = sum((inst.Bm[y] * k for (jj, y), k in w.items() if jj == j), np.zeros(len(inst.components)))
        if np.any(load > inst.A[:, j] * x[j] + 1e-9):
            return False
    return True


def brute_force(inst):
    """Full enumeration of plans x and assembly assignments w; no solver involved."""
    best = -np.inf
    n = len(inst.pool)
    for x in itertools.product(range(inst.Q + 1), repeat=n):
        if sum(x) != inst.Q:
            continue
        x = np.array(x)
        ranges = [range(min(x[j], int(inst.Ys[y])) + 1) for j, y in inst.pairs]
        for ks in itertools.product(*ranges):
            w = {p: k for p, k in zip(inst.pairs, ks) if k}
            if _valid(inst, x, w):
                best = max(best, data.evaluate(inst, x, w)["Z"])
    return best


def test_milp_matches_enumeration():
    ds = toy()
    for use in (False, True):
        for Q in (1, 3, 5):
            inst = data.build(ds, 3, Q, use_assemblies=use)
            sol = model.solve(inst)
            z = data.evaluate(inst, sol["x"], sol["w"])["Z"]
            assert abs(z - brute_force(inst)) < 1e-6
            assert sol["x"].sum() == Q


def test_heuristics_feasible_and_bounded():
    ds = toy()
    inst = data.build(ds, 3, 5)
    sol = model.solve(inst)
    zo = data.evaluate(inst, sol["x"], sol["w"])["Z"]
    x, w = heuristics.greedy(inst)
    assert x.sum() == 5
    used = {}
    for (j, y), k in w.items():
        assert k <= x[j]
        used[y] = used.get(y, 0) + k
    assert all(used[y] <= inst.Ys[y] for y in used)
    assert data.evaluate(inst, x, w)["Z"] <= zo + 1e-9
    g = heuristics.ga(inst, seed=1, max_gen=100, stall=50)
    assert g["x"].sum() == 5 and g["Z"] <= zo + 1e-9


def test_decoder_equals_greedy_allocation():
    ds = toy()
    inst = data.build(ds, 3, 4)
    x, w = heuristics.greedy(inst)
    assert abs(heuristics.Decoder(inst).score(x)[0] - data.evaluate(inst, x, w)["Z"]) < 1e-9


def test_value_monotone_in_stock():
    ds = toy()
    zs = []
    for sf in (0.5, 1.0, 2.0):
        inst = data.build(ds, 3, 4, stock_factor=sf)
        sol = model.solve(inst)
        zs.append(data.evaluate(inst, sol["x"], sol["w"])["Z"])
    assert zs[0] <= zs[1] + 1e-9 <= zs[2] + 2e-9


def test_assemblies_never_reduce_optimum():
    ds = toy()
    for Q in (2, 4):
        a = data.build(ds, 3, Q, use_assemblies=False); b = data.build(ds, 3, Q, use_assemblies=True)
        za = data.evaluate(a, model.solve(a)["x"], {})["Z"]
        sb = model.solve(b)
        assert data.evaluate(b, sb["x"], sb["w"])["Z"] >= za - 1e-9


def test_lambda_one_matches_base_and_cap_binds():
    ds = toy()
    inst = data.build(ds, 3, 4)
    a = model.solve(inst); b = model.solve(inst, lam=1.0, stock_weight=1.0)
    assert abs(data.evaluate(inst, a["x"], a["w"])["Z"] - data.evaluate(inst, b["x"], b["w"])["Z"]) < 1e-9
    lo = model.solve(inst, lam=1.0, stock_weight=1e-6)
    bmin = data.evaluate(inst, lo["x"], lo["w"])["procurement"]
    c = model.solve(inst, lam=1e-6, stock_weight=1.0, procurement_cap=bmin + 1e-9)
    assert data.evaluate(inst, c["x"], c["w"])["procurement"] <= bmin + 1e-6


def test_pareto_front_monotone():
    from revplan.experiment import pareto_front
    ds = toy()
    f = pareto_front(data.build(ds, 3, 5), points=5)
    b = [p["procurement"] for p in f]; u = [p["stock_used_total"] for p in f]
    assert all(b[i] <= b[i + 1] + 1e-9 for i in range(len(b) - 1))
    assert all(u[i] <= u[i + 1] + 1e-9 for i in range(len(u) - 1))




def test_check_assignment_rejects_infeasible():
    ds = toy()
    inst = data.build(ds, 3, 2)
    j, y = inst.pairs[0]
    x = np.zeros(3, dtype=int); x[j] = 2
    checks.check_assignment(inst, x, {(j, y): 1})
    for bad in ({(j, y): 3}, {(j, y): int(inst.Ys[y]) + 1}, {(j, y): 0.5}, {(j, y): float("nan")}, {(j, y): -1}):
        try:
            checks.check_assignment(inst, x, bad)
        except ValueError:
            continue
        raise AssertionError(f"infeasible assignment accepted: {bad}")
    xf = x.astype(float); xf[j] = 1.5; xf[(j + 1) % 3] = 0.5      # sums to Q but fractional
    xs = x.astype(float); xs[j] = 2.5                              # wrong total, fractional
    xn = x.astype(float); xn[(j + 1) % 3] = float("nan")
    for badx in (xf, xs, xn):
        try:
            checks.check_assignment(inst, badx, {})
        except ValueError:
            continue
        raise AssertionError(f"invalid plan accepted: {badx}")


def test_lexicographic_front_endpoints():
    from revplan.experiment import pareto_front
    ds = toy()
    inst = data.build(ds, 3, 4)
    f = pareto_front(inst, points=4)
    best_p = min(data.evaluate(inst, np.array(x), {})["procurement"] for x in itertools.product(range(5), repeat=3) if sum(x) == 4)
    assert f[0]["procurement"] <= best_p + 1e-9
    assert all(a["stock_used_total"] <= b["stock_used_total"] + 1e-9 for a, b in zip(f, f[1:]))


@needs_data
def test_verify_rejects_fractional_stored_records():
    """A fractional quantity in a stored JSON record must fail verification, not be truncated."""
    import copy, json
    from revplan import verify
    from revplan.experiment import RESULTS
    ds = data.load()
    det = json.loads((RESULTS / "deterministic.json").read_text())
    row = next(r for r in det["scenarios"] if (r["pool"], r["Q"], r["stock_factor"]) == (25, 130, 1.0))
    assert row["with_assemblies"]["milp"]["assemblies"], "test needs a scenario with assembly use"
    exp = dict(plans=4, greedy=2, proportional=2)
    fails = []
    verify.check_deterministic(ds, dict(scenarios=[row]), fails, last_unit=False, expected=exp)
    assert fails == [], fails                                # the stored record itself passes
    for field, bad in [("mix", 0.5), ("mix", float("nan")), ("mix", "1"), ("assemblies", 0.5)]:
        r2 = copy.deepcopy(row)
        m = r2["with_assemblies"]["milp"]
        if field == "mix":
            p0 = next(iter(m["mix"])); m["mix"][p0] = m["mix"][p0] + bad if isinstance(bad, float) else bad
        else:
            m["assemblies"][0][2] = m["assemblies"][0][2] + bad
        json.loads(json.dumps(r2, allow_nan=True))          # still valid JSON as stored on disk
        fails = []
        verify.check_deterministic(ds, dict(scenarios=[r2]), fails, last_unit=False, expected=exp)
        assert any("milp" in f for f in fails) and "stored MILP/greedy plans" in fails, (field, bad, fails)
    inst = data.build(ds, 9, 20)
    for bad_mix in ({next(iter(row["components_only"]["milp"]["mix"])): 20.5}, {"UNKNOWN": 20}):
        try:
            verify.load_plan(inst, bad_mix)
        except ValueError:
            continue
        raise AssertionError(f"load_plan accepted {bad_mix}")


def test_proportional_rule():
    from revplan.experiment import proportional
    ds = toy()
    for Q in (1, 4, 7, 20):
        inst = data.build(ds, 3, Q)
        x = proportional(ds, inst)
        q = np.array([ds.programme[p] for p in inst.pool], dtype=float)
        assert x.sum() == Q and np.all(np.abs(x - Q * q / q.sum()) < 1)


@needs_data
def test_netting_rules_and_allocation():
    import datetime as dt
    from revplan import netting
    ds = toy()
    d = dt.date
    units = [dict(unit_id="U1", product_id="P1", plan_month="dec", bo_entry=d(2023, 12, 13), bs_entry=d(2023, 12, 15), bs_exit=d(2023, 12, 18), assemblies=["A1"]),
             dict(unit_id="U2", product_id="P2", plan_month="dec", bo_entry=d(2023, 12, 15), bs_entry=d(2023, 12, 18), bs_exit=d(2023, 12, 19), assemblies=["A1", "A2"]),
             dict(unit_id="U3", product_id="P2", plan_month="jan", bo_entry=None, bs_entry=d(2024, 1, 9), bs_exit=d(2024, 1, 10), assemblies=["A1"])]
    n = {r: netting.net(ds.bom, ds.assembly_bom, ds.stock, ds.assembly_stock, units, r) for r in netting.RULES}
    assert n["bs_exit"].units == ["U1", "U2", "U3"] and n["bs_entry"].units == ["U2", "U3"] and n["bo_entry"].units == ["U3"]
    m = n["bs_entry"]                       # allocation follows the plan rows in plan order while stock lasts
    assert m.allocation == {"U2": ["A1", "A2"], "U3": ["A1"]} and m.assembly_residual == {"A1": 0, "A2": 0}
    # requirement = BOM minus assembly content, floored at zero per unit (A2 supplies c2, which P2 does not need)
    assert {c: q for c, q in m.requirement.items() if q > 0} == {"c1": 2, "c3": 2}
    assert all(m.residual[c] == max(ds.stock[c] - m.requirement.get(c, 0), 0) for c in ds.stock)
    assert all(n["bo_entry"].residual[c] >= m.residual[c] >= n["bs_exit"].residual[c] for c in ds.stock)


def counterexample():
    """Reviewer's example: each product needs A, B, C once; stock holds one AB, one AC and one BC assembly."""
    return data.Dataset(
        products=["P"], programme={"P": 1}, bom={"P": {"A": 1, "B": 1, "C": 1}},
        stock={"A": 0, "B": 0, "C": 0}, values={"unit_value": {"A": 1.0, "B": 1.0, "C": 1.0}},
        imputed={"A": False, "B": False, "C": False},
        assemblies=["AB", "AC", "BC"], assembly_kind={"AB": "x", "AC": "x", "BC": "x"},
        assembly_stock={"AB": 1, "AC": 1, "BC": 1},
        assembly_bom={"AB": {"A": 1, "B": 1}, "AC": {"A": 1, "C": 1}, "BC": {"B": 1, "C": 1}})


def test_aggregate_model_is_a_relaxation_and_kits_are_exact():
    from revplan import kits
    inst = data.build(counterexample(), 1, 2)
    agg = model.solve(inst)
    za = data.evaluate(inst, agg["x"], agg["w"])
    assert abs(za["Z"] - 6) < 1e-9 and abs(za["procurement"]) < 1e-9      # aggregate: all three used
    assert kits.dispatch(inst, agg["x"], agg["w"]) is None                 # ...but not dispatchable
    assert not kits.all_fitting_sets_are_kits(inst)
    k = kits.solve(inst)
    zk = data.evaluate(inst, k["x"], k["w"])
    assert abs(zk["Z"] - 2) < 1e-9 and abs(zk["procurement"] - 2) < 1e-9   # unit-level optimum
    assert kits.dispatch(inst, k["x"], k["w"]) is not None


def test_kits_match_aggregate_when_fitting_sets_are_kits():
    from revplan import kits
    ds = toy()
    for Q in (2, 4, 5):
        inst = data.build(ds, 3, Q)
        assert kits.all_fitting_sets_are_kits(inst)
        a, k = model.solve(inst), kits.solve(inst)
        assert abs(data.evaluate(inst, a["x"], a["w"])["Z"] - data.evaluate(inst, k["x"], k["w"])["Z"]) < 1e-9
        assert kits.dispatch(inst, a["x"], a["w"]) is not None


def test_diversity_ga_reproduces_package_ga():
    from revplan import extra
    inst = data.build(toy(), 3, 5)
    for sx in (None, heuristics.greedy(inst)[0]):
        g = heuristics.ga(inst, 7, seed_x=sx, max_gen=30, stall=10 ** 9)
        d = extra._ga_div(inst, 7, seed_x=sx, generations=30, checkpoints=(0, 30))
        assert g["history"][0] == d["trace"][0]["best"] and g["Z"] == d["Z"]


def test_dispatch_reproduces_assignment_exactly():
    from revplan import kits
    inst = data.build(toy(), 3, 4)
    sol = model.solve(inst)
    d = kits.dispatch(inst, sol["x"], sol["w"])
    assert d is not None and kits.check_dispatch(inst, sol["x"], sol["w"], d)
    used = sum(len(K) for us in d.values() for K in us)
    assert used == sum(sol["w"].values())                     # no assembly added beyond the stored plan
    j = next(j for j in range(3) if sol["x"][j] > 0 and any(jj == j for jj, _ in inst.pairs))
    d0 = kits.dispatch(inst, sol["x"], {})                     # plan without assemblies stays without
    assert d0 is not None and all(K == () for us in d0.values() for K in us)
    y = next(y for jj, y in inst.pairs if jj == j)
    assert kits.dispatch(inst, sol["x"], {(j, y): int(sol["x"][j]) + 1}) is None   # more than x_j units


def test_holm_adjustment():
    from revplan.extra import _holm
    adj = _holm([0.0145, 0.0473, 0.404, 3.79e-06])
    assert [round(a, 4) for a in adj] == [0.0435, 0.0946, 0.404, 0.0]



@needs_data
def test_verified_pairs_netting():
    from revplan import netting, data
    g = data.load(stock="gross")
    units = netting.read_units()
    vu = netting.verified_units(units, g.bom, g.assembly_bom)
    assert all(set(b["assemblies"]) <= set(a["assemblies"]) for a, b in zip(units, vu))
    assert all(netting.fits(g.bom[u["product_id"]], g.assembly_bom[y]) for u in vu for y in u["assemblies"])
    _, r_plan = netting.dataset()
    _, r_ver = netting.dataset(pairs="verified")
    # fewer allocations leave at least as many assemblies in stock
    assert all(r_ver.assembly_stock[y] >= r_plan.assembly_stock[y] for y in r_plan.assembly_stock)

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            if fn() != "skipped":
                print("passed", name)
