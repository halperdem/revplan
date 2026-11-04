# revplan — inventory-driven reverse production planning (code release v9.1)

Source code for the article *"Inventory-driven reverse production planning with semi-finished assemblies: programme design and residual-stock planning in a tractor plant"* ([JOURNAL], [YEAR], DOI: [DOI]).

Forward planning derives materials from demand. **Reverse planning** derives a product mix from the materials already held. The case is studied in two stages:

* **Stage A – programme design (gross stock, counterfactual).** The remaining frozen programme (701 units, 35 models) is re-optimised on the ERP stock snapshot, with stocked semi-finished assemblies (engines, transmissions, power-trains, cabs, axles, platforms), under mix-flexibility bands around the actual programme: what could have been done before the freeze.
* **Stage B – additional production (residual stock, main case).** After the frozen programme has drawn its materials (netting in `revplan.netting`), choose how many units of each candidate model to add (Q = 20, 130, 150) so that the stock left over is used and external procurement is limited; compared with the exact optimum, a greedy rule, genetic algorithms and a programme-proportional mix.

## Contents

```
src/revplan/     data.py (instances; stock = residual | gross) · checks.py (feasibility, fingerprints) · model.py (MILP) · heuristics.py (greedy, GA)
                 kits.py (unit-level kit formulation, dispatch) · netting.py (programme netting, three rules) · experiment.py (Stage B)
                 stage_a.py (Stage A) · extra.py (robustness, netting rules, plan reports, method limits) · analysis.py · verify.py
tests/           synthetic checks (MILP vs solver-free enumeration of x and w, feasibility, monotonicity, front endpoints)
```

## Data

The industrial data used in the article are not publicly available due to commercial confidentiality restrictions, and they are not part of this repository. The code expects them in a folder `data/` next to `src/` with the files `products.csv`, `programme_units.csv`, `bom.csv`, `components.csv`, `assemblies.csv` and `assembly_bom.csv` (column layout in `src/revplan/data.py` and `src/revplan/netting.py`). Without that folder the synthetic tests run (tests that need the plant data are skipped), but the experiments cannot be repeated.

## Reproduce (with the data folder in place)

Python ≥ 3.11.

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src            # Windows PowerShell: $env:PYTHONPATH="src"
python tests/test_revplan.py     # or: python -m pytest tests
python -m revplan.stage_a                           # Stage A: programme design on gross stock, bands, availability, leave-one-type-out (~5 min)
python -m revplan.experiment deterministic          # Stage B: MILP + greedy + proportional mix, 36 scenarios x 2 models, λ, valuation (~15 min)
python -m revplan.experiment pareto                 # Stage B: stock use vs procurement front, 7 cases, two-stage solves
python -m revplan.experiment ga --workers 2         # Stage B: GA + greedy-seeded GA, 12 scenarios x 30 runs (resumable)
python -m revplan.experiment ga-history --workers 2 # re-runs every GA seed: full convergence traces + reproduction check (resumable)
python -m revplan.extra all                         # sales caps, valuation, stock composition, GA statistics, netting rules,
                                                    # plant-facing plan reports (results/extra/plans/), where simple rules fail
python -m revplan.verify                            # independent checks of all stored results -> results/verification.json
                                                    # exits with status 1 on any failed check or missing file (--no-reruns skips the GA re-runs;
                                                    # --resolve also re-solves the 144 dominance problems)
python -m revplan.analysis                          # tables and figures
```

For runtime measurements set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. Objective values and plans are deterministic for given seeds; runtimes depend on the machine. The stored results were produced with the versions pinned in `requirements.txt` (Python 3.13) on a cloud virtual machine with 2 virtual cores of an Intel Xeon processor at 2.1 GHz and 8 GB RAM.

Every stored result carries a fingerprint (SHA-256 of `data/*.csv` and of the modules that produced it, line endings normalised; scopes in `checks.py`). Resumed GA runs reuse a stored file only if its fingerprint matches the current data and code, and otherwise stop with an error (use `--force` to recompute). `revplan.verify` fails if any stored result has a different fingerprint.

## Experimental design

**Stage A** (gross stock, Q = 701 = remaining programme, all 35 programme models, exact kit formulation):

| Factor | Levels |
| --- | --- |
| Mix-flexibility band | floor(q_j/f) ≤ x_j ≤ ceil(q_j·f), f = 1 (actual programme), 1.1, 1.25, 1.5, 2, none |
| Model | components only; with stocked assemblies |
| Assembly availability | assembly stock ×0, 0.25, 0.5, 0.75, 1 (f = 1 and none) |
| Compatibility | one assembly type removed at a time (f = 1 and none) |

**Stage B** (residual stock, main case):

| Factor | Levels |
| --- | --- |
| Candidate pool (Pareto) | top 9, 15, 25, 35 models (61.6%, 84.0%, 97.1%, 100% of programme volume) |
| Additional quantity Q | 20, 130, 150 units (exact) |
| Stock level | ×0.8, ×1.0, ×1.2 (assemblies rounded down) |
| Model | components only; with semi-finished substitution |
| Methods | MILP (HiGHS via SciPy), greedy, GA, greedy-seeded GA; baseline: programme-proportional mix (largest remainder) with optimal assembly use |
| GA | population 100, crossover 0.8, mutation 0.05/gene, 2 elites, ≤2000 generations, stop after 200 without improvement, 30 runs, seed = 20261022 + 1000·scenario + run |
| Procurement weight λ | 0.5, 1, 2 (MILP, base stock) |
| Efficient front | ε-constraint, 15 points, 7 cases; every point solved lexicographically in two stages; relative MIP gap 1e-6 per stage, status and gap stored |

Objective: `Z = Σ v_i u_i + Σ V_y w_jy − λ Σ v_i p_i` (stock use plus value of installed assemblies minus weighted procurement). Z is a net inventory-utilisation value at standard cost, not profit; it can be negative. Gaps are therefore reported in absolute terms and as a percentage of the optimal plan's material value (stock use + procurement). GA–greedy differences below 1e-9 of that value are set to zero before testing (Wilcoxon signed-rank, zeros dropped; exact sign-flip test; Holm adjustment).

## Licence

Code: MIT (see `LICENSE`). The industrial data are not covered by this licence and are not distributed.
