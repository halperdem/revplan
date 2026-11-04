"""Load the anonymised plant data and build planning instances."""
from __future__ import annotations
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@dataclass
class Dataset:
    products: list            # product ids in Pareto rank order of the remaining programme
    programme: dict           # product -> remaining frozen programme units (Dec + Jan) after the stock date
    bom: dict                 # product -> {component: qty per unit}
    stock: dict               # component -> stock qty
    values: dict              # value column -> {component: unit value}
    imputed: dict             # component -> bool
    assemblies: list          # assembly ids (release order)
    assembly_kind: dict
    assembly_stock: dict
    assembly_bom: dict        # assembly -> {component: qty per assembly}


STOCK_STATES = ("residual", "gross")


def load(data_dir: Path | str = DATA_DIR, stock: str = "residual") -> Dataset:
    """Load the plant data.

    stock="residual": stock left after the remaining frozen programme has been served (main case, Stage B);
    stock="gross": ERP stock at the snapshot date, before the remaining programme (Stage A, programme design).
    """
    if stock not in STOCK_STATES:
        raise ValueError(f"stock must be one of {STOCK_STATES}")
    d = Path(data_dir)
    prods = sorted(_read(d / "products.csv"), key=lambda r: int(r["pareto_rank"]))
    bom = defaultdict(dict)
    for r in _read(d / "bom.csv"):
        bom[r["product_id"]][r["component_id"]] = float(r["qty_per_unit"])
    comps = _read(d / "components.csv")
    cols = [c for c in comps[0] if c.startswith("unit_value")]
    asm = _read(d / "assemblies.csv")
    abom = defaultdict(dict)
    for r in _read(d / "assembly_bom.csv"):
        abom[r["assembly_id"]][r["component_id"]] = float(r["qty_per_assembly"])
    return Dataset(
        products=[r["product_id"] for r in prods],
        programme={r["product_id"]: int(r["programme_dec_qty"]) + int(r["programme_jan_qty"]) for r in prods},
        bom=dict(bom),
        stock={r["component_id"]: float(r[f"stock_{stock}"]) for r in comps},
        values={c: {r["component_id"]: float(r[c]) for r in comps} for c in cols},
        imputed={r["component_id"]: r["value_imputed"] == "True" for r in comps},
        assemblies=[r["assembly_id"] for r in asm],
        assembly_kind={r["assembly_id"]: r["kind"] for r in asm},
        assembly_stock={r["assembly_id"]: int(r[f"stock_{stock}"]) for r in asm},
        assembly_bom=dict(abom),
    )


@dataclass
class Instance:
    """One planning instance: pool of models, quantity Q, stock level, optional assemblies."""
    pool: list
    Q: int
    components: list
    A: np.ndarray            # components x models, exploded BOM
    s: np.ndarray            # component stock
    v: np.ndarray            # component unit values
    ys: list = field(default_factory=list)       # usable assemblies
    Bm: dict = field(default_factory=dict)       # assembly -> component vector
    V: dict = field(default_factory=dict)        # assembly -> value
    Ys: dict = field(default_factory=dict)       # assembly -> stock (integer)
    pairs: list = field(default_factory=list)    # (model index, assembly) fits


def build(ds: Dataset, pool_size: int, Q: int, stock_factor: float = 1.0,
          use_assemblies: bool = True, value_column: str = "unit_value",
          imputed_factor: float = 1.0) -> Instance:
    pool = ds.products[:pool_size]
    comps = sorted({c for j in pool for c in ds.bom[j]})
    ix = {c: i for i, c in enumerate(comps)}
    A = np.zeros((len(comps), len(pool)))
    for j, p in enumerate(pool):
        for c, q in ds.bom[p].items():
            A[ix[c], j] = q
    vals = ds.values[value_column]
    s = np.array([ds.stock.get(c, 0.0) * stock_factor for c in comps])
    v = np.array([vals[c] * (imputed_factor if ds.imputed[c] else 1.0) for c in comps])
    inst = Instance(pool, int(Q), comps, A, s, v)
    if use_assemblies:
        def fits(p):
            B = ds.bom[p]
            return [y for y in ds.assemblies if ds.assembly_stock[y] > 0 and
                    all(c in B and B[c] >= q - 1e-9 for c, q in ds.assembly_bom[y].items())]
        inst.pairs = [(j, y) for j, p in enumerate(pool) for y in fits(p)]
        inst.ys = sorted({y for _, y in inst.pairs})
        for y in inst.ys:
            b = np.zeros(len(comps))
            for c, q in ds.assembly_bom[y].items():
                b[ix[c]] = q
            inst.Bm[y] = b
            inst.V[y] = float(b @ v)
            inst.Ys[y] = float(np.floor(ds.assembly_stock[y] * stock_factor + 1e-9))
    return inst


def evaluate(inst: Instance, x, w: dict) -> dict:
    """Value a plan x (units per model) with assembly use w {(j, y): units}."""
    d = inst.A @ np.asarray(x, dtype=float)
    for (j, y), k in w.items():
        d = d - inst.Bm[y] * k
    u = np.minimum(d, inst.s)
    b = d - u
    uy = sum(inst.V[y] * k for (j, y), k in w.items())
    return dict(Z=float(inst.v @ u - inst.v @ b + uy), stock_used=float(inst.v @ u),
                assembly_value_used=float(uy), procurement=float(inst.v @ b),
                assembly_units=int(sum(w.values())), units=int(np.sum(x)))
