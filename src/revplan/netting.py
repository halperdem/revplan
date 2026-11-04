"""Netting the remaining frozen programme from the gross stock (reproducible from the plant data).

The plant assembles tractors on two consecutive lines: the pre-paint line (BO) and the post-paint line (BS).

data/programme_units.csv lists every programme tractor that had not left the BS line at the stock cut-off
(16 December 2023), in plan order, with its line dates and the stocked assemblies its plan row names. A unit
still has to draw its materials from the stock if the date of the selected rule is on or after the cut-off:

  bs_entry  materials are issued when the unit enters the BS line (main case; the plant issues material at line
            entry / work-order release, so units already on the line have drawn theirs);
  bo_entry  materials are issued when the unit enters the BO (pre-paint) line (more units count as drawn);
  bs_exit   materials are back-flushed when the unit leaves the BS line (fewest units count as drawn).

Stocked assemblies are allocated to the remaining units in plan order (an assembly named in a unit's plan row is
taken if a unit of it is still in stock). Each unit's leaf requirement is its exploded BOM minus the content of the
assemblies it receives (never below zero), and the residual stock of a component is
max(gross stock - total requirement, 0); for assemblies, gross stock minus the units allocated.

pairs="verified" (sensitivity) allocates a planned assembly only if the model's BOM contains every component of
the assembly in at least the same quantity (the compatibility rule of the model); other plan pairs are ignored,
the assembly stays in stock and the tractor's whole leaf requirement is netted from the component stock.
"""
from __future__ import annotations
import csv
import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from .data import DATA_DIR, Dataset, _read, load

CUT = dt.date(2023, 12, 16)
RULES = ("bs_entry", "bo_entry", "bs_exit")
MAIN_RULE = "bs_entry"


@dataclass
class Netting:
    rule: str
    units: list                 # unit ids still drawing material, in plan order
    programme: dict             # product -> remaining units
    programme_by_month: dict    # (month, product) -> units
    allocation: dict            # unit id -> list of allocated assembly ids
    requirement: dict           # component -> leaf requirement of the remaining programme
    residual: dict              # component -> residual stock
    assembly_residual: dict     # assembly -> residual stock


def read_units(data_dir: Path | str = DATA_DIR) -> list:
    out = []
    for r in _read(Path(data_dir) / "programme_units.csv"):
        out.append(dict(unit_id=r["unit_id"], product_id=r["product_id"], plan_month=r["plan_month"],
                        bo_entry=_date(r["bo_entry"]), bs_entry=_date(r["bs_entry"]), bs_exit=_date(r["bs_exit"]),
                        assemblies=[a for a in r["assemblies"].split(";") if a]))
    return out


def _date(s):
    return dt.date.fromisoformat(s) if s else None


def remaining(unit: dict, rule: str, cut: dt.date = CUT) -> bool:
    """True if the unit still draws its materials from the stock. A missing date counts as not yet drawn."""
    d = unit[rule]
    return d is None or d >= cut


def net(bom: dict, assembly_bom: dict, gross: dict, assembly_gross: dict, units: list, rule: str = MAIN_RULE,
        cut: dt.date = CUT) -> Netting:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    left = {y: int(round(s)) for y, s in assembly_gross.items()}
    req = Counter(); alloc = {}; keep = []; prog = Counter(); by_month = Counter()
    for u in units:
        if not remaining(u, rule, cut):
            continue
        keep.append(u["unit_id"]); prog[u["product_id"]] += 1; by_month[(u["plan_month"], u["product_id"])] += 1
        need = Counter(bom[u["product_id"]])
        got = []
        for y in u["assemblies"]:
            if left.get(y, 0) >= 1:
                left[y] -= 1; got.append(y)
                for c, q in assembly_bom[y].items():
                    need[c] = max(need[c] - q, 0.0)
        alloc[u["unit_id"]] = got
        for c, q in need.items():
            if q > 0:
                req[c] += q
    residual = {c: max(s - req.get(c, 0.0), 0.0) for c, s in gross.items()}
    return Netting(rule, keep, dict(prog), dict(by_month), alloc, dict(req), residual, left)


def from_release(data_dir: Path | str = DATA_DIR, rule: str = MAIN_RULE, cut: dt.date = CUT) -> Netting:
    g = load(data_dir, stock="gross")
    return net(g.bom, g.assembly_bom, g.stock, g.assembly_stock, read_units(data_dir), rule, cut)


def fits(bom_p: dict, assembly_content: dict) -> bool:
    """Containment rule: the model's BOM covers every component of the assembly in at least the same quantity."""
    return all(bom_p.get(c, 0.0) >= q - 1e-9 for c, q in assembly_content.items())


def verified_units(units: list, bom: dict, assembly_bom: dict) -> list:
    """Programme units with only those planned assemblies that pass the containment rule."""
    return [dict(u, assemblies=[y for y in u["assemblies"] if fits(bom[u["product_id"]], assembly_bom[y])]) for u in units]


def dataset(data_dir: Path | str = DATA_DIR, rule: str = MAIN_RULE, deduct: tuple = (), cut: dt.date = CUT,
            pairs: str = "plan") -> tuple:
    """(gross dataset, residual dataset) for a netting rule, optionally deducting further per-component quantities
    from the residual stock (columns of components.csv, e.g. 'open_service_orders', 'unposted_backflush').
    Products are re-ranked by the programme volume under the rule."""
    g = load(data_dir, stock="gross")
    units = read_units(data_dir)
    if pairs == "verified":
        units = verified_units(units, g.bom, g.assembly_bom)
    elif pairs != "plan":
        raise ValueError("pairs must be 'plan' or 'verified'")
    n = net(g.bom, g.assembly_bom, g.stock, g.assembly_stock, units, rule, cut)
    extra = Counter()
    if deduct:
        for r in _read(Path(data_dir) / "components.csv"):
            for col in deduct:
                extra[r["component_id"]] += float(r[col])
    stock = {c: max(s - extra.get(c, 0.0), 0.0) for c, s in n.residual.items()}
    rank = {p: i for i, p in enumerate(g.products)}
    prods = sorted(g.products, key=lambda p: (-n.programme.get(p, 0), rank[p]))
    prog = {p: n.programme.get(p, 0) for p in prods}
    gross = replace(g, products=prods, programme=prog)
    resid = replace(g, products=prods, programme=prog, stock=stock, assembly_stock=dict(n.assembly_residual))
    return gross, resid
