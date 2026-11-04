"""Tables and figures of the paper from results/ (python -m revplan.analysis).

Gaps are reported in two forms: absolute (monetary units of the release) and as a percentage of the material
value of the optimal plan, D = stock use + procurement. Z itself is close to zero or negative in several Stage B
scenarios, so a gap relative to Z would be meaningless.
"""
from __future__ import annotations
import csv
import glob
import json
import statistics as st
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt
from . import data
from .experiment import RESULTS, POOLS, QS

plt.rcParams.update({"font.size": 9, "legend.fontsize": 8,   # figures are drawn at print width (15 cm)
                      "axes.spines.top": False, "axes.spines.right": False})
C = {"milp": "#2E7D32", "greedy": "#335C81", "ga": "#A24936", "gas": "#D08C60", "prop": "#9AA5B1", "opt": "#335C81"}


def U(e):
    return e["stock_used"] + e["assembly_value_used"]


def scale(e):
    return U(e) + e["procurement"]


TOL = 1e-9          # relative to the plan's material value: differences below this are rounding, not loss


def equalise(d, opt):
    """Set differences within the numerical tolerance to exactly zero before any test."""
    d = np.asarray(d, dtype=float).copy()
    d[np.abs(d) <= TOL * scale(opt)] = 0.0
    return d


def wilcoxon_p(d):
    """Two-sided Wilcoxon signed-rank test on GA - greedy differences; zero differences are dropped
    (zero_method='wilcox'); with ties SciPy uses the normal approximation. 1.0 if every difference is zero."""
    d = np.asarray(d, dtype=float)
    return 1.0 if not np.any(d) else float(wilcoxon(d, zero_method="wilcox").pvalue)


def signflip_p(d, draws: int = 200_000, seed: int = 20261022):
    """Two-sided sign-flip permutation test of the mean of the non-zero differences (exact up to 20 of them)."""
    d = np.asarray(d, dtype=float); d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    obs = abs(d.sum())
    if n <= 20:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(draws, n))
    return float(np.mean(np.abs(signs @ d) >= obs - 1e-12))


def gap(opt, z):
    """Gap to the optimum in % of the optimal plan's material value (0 within 1e-9 relative tolerance)."""
    d = opt["Z"] - z
    return 0.0 if abs(d) <= 1e-9 * scale(opt) else 100 * d / scale(opt)


def _holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = float("%.3g" % running)
    return adj


def stage_b_tables(res, D, G):
    ds = data.load()
    tot = sum(ds.programme.values())
    share = {k: sum(ds.programme[p] for p in ds.products[:k]) / tot for k in POOLS}
    with open(res / "table_stage_b.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "Q", "stock_factor", "pool_share_of_programme_pct", "Z_opt", "stock_use_opt", "procurement_opt",
                    "Z_proportional", "stock_use_proportional", "procurement_proportional", "procurement_saving_vs_proportional",
                    "procurement_saving_vs_proportional_pct", "assembly_units_opt", "Z_opt_components_only", "greedy_abs_gap",
                    "greedy_gap_pct", "milp_seconds", "greedy_seconds", "milp_status"])
        for r in D.values():
            a = r["with_assemblies"]; m, g, p = a["milp"], a["greedy"], a["proportional"]
            w.writerow([r["pool"], r["Q"], r["stock_factor"], round(100 * share[r["pool"]], 1), round(m["Z"], 3), round(U(m), 3),
                        round(m["procurement"], 3), round(p["Z"], 3), round(U(p), 3), round(p["procurement"], 3),
                        round(p["procurement"] - m["procurement"], 3), round(100 * (1 - m["procurement"] / p["procurement"]), 2),
                        m["assembly_units"], round(r["components_only"]["milp"]["Z"], 3), round(m["Z"] - g["Z"], 3),
                        round(gap(m, g["Z"]), 5), round(m["seconds"], 3), round(g["seconds"], 4), m["status"]])
    rows = []
    for (k, Q), g in sorted(G.items()):
        y = D[(k, Q, 1.0)]["with_assemblies"]; opt = y["milp"]; zg = y["greedy"]["Z"]
        row = dict(pool=k, Q=Q, milp_Z=round(opt["Z"], 3), plan_value=round(scale(opt), 3), greedy_gap_pct=round(gap(opt, zg), 5),
                   milp_seconds=round(opt["seconds"], 3), greedy_seconds=round(y["greedy"]["seconds"], 4))
        for v, tag in (("random", "GA"), ("greedy_seeded", "GAseeded")):
            zs = np.array([x["Z"] for x in g["runs"][v]])
            d = equalise(zs - zg, opt)
            row.update({f"{tag}_best_gap_pct": round(gap(opt, zs.max()), 5), f"{tag}_mean_gap_pct": round(gap(opt, zs.mean()), 5),
                        f"{tag}_worst_gap_pct": round(gap(opt, zs.min()), 5),
                        f"{tag}_runs_at_optimum": int(np.sum(np.abs(zs - opt["Z"]) <= TOL * scale(opt))),
                        f"{tag}_runs_better_than_greedy": int(np.sum(d > 0)), f"{tag}_runs_worse_than_greedy": int(np.sum(d < 0)),
                        f"{tag}_wilcoxon_p": float("%.3g" % wilcoxon_p(d)), f"{tag}_signflip_p": float("%.3g" % signflip_p(d)),
                        f"{tag}_mean_seconds": round(st.mean(x["seconds"] for x in g["runs"][v]), 2),
                        f"{tag}_mean_generations": round(st.mean(x["generations"] for x in g["runs"][v])),
                        f"{tag}_mean_evaluations": round(st.mean(x["evaluations"] for x in g["runs"][v]))})
        rows.append(row)
    for tag in ("GA", "GAseeded"):                    # Holm adjustment over the twelve scenarios
        for test in ("wilcoxon", "signflip"):
            for r, a in zip(rows, _holm([r[f"{tag}_{test}_p"] for r in rows])):
                r[f"{tag}_{test}_p_holm"] = a
    if rows:
        with open(res / "table_methods.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return share


def stage_a_tables(res):
    f = res / "stage_a.json"
    if not f.exists():
        return None
    A = json.loads(f.read_text())
    with open(res / "table_stage_a.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["analysis", "model", "band", "assembly_stock_factor", "dropped", "stock_use", "procurement", "Z",
                    "assembly_units", "units_moved", "status", "seconds"])
        for r in A["bands"]:
            w.writerow(["band", r["model"], r["band"], "", "", round(r["stock_used_total"], 3), round(r["procurement"], 3),
                        round(r["Z"], 3), r["assembly_units"], r["units_moved"], r["status"], round(r["seconds"], 3)])
        for k, r in A["greedy"].items():
            w.writerow(["greedy", k, None, "", "", round(r["stock_used_total"], 3), round(r["procurement"], 3), round(r["Z"], 3),
                        r["assembly_units"], r["units_moved"], "", ""])
        for r in A["availability"]:
            w.writerow(["availability", "with_assemblies", r["band"], r["assembly_stock_factor"], "", round(r["stock_used_total"], 3),
                        round(r["procurement"], 3), round(r["Z"], 3), r["assembly_units"], r["units_moved"], r["status"], round(r["seconds"], 3)])
        for r in A["leave_one_type_out"]:
            w.writerow(["leave_one_type_out", "with_assemblies", r["band"], "", r["dropped"], round(r["stock_used_total"], 3),
                        round(r["procurement"], 3), round(r["Z"], 3), r["assembly_units"], r["units_moved"], r["status"], round(r["seconds"], 3)])
    return A


def figures(res, D, G, A):
    fig_dir = res / "figures"
    fig_dir.mkdir(exist_ok=True)
    # Stage A: procurement of the programme design by mix-flexibility band
    if A:
        bands = [b["band"] for b in A["bands"] if b["model"] == "with_assemblies"]
        lab = ["programme" if b == 1.0 else ("free" if b is None else f"×{b:g}") for b in bands]
        fig, ax = plt.subplots(figsize=(5.9, 3.4)); xs = np.arange(len(bands))
        for off, model, col, name in [(-.2, "components_only", C["prop"], "Components only"),
                                      (.2, "with_assemblies", C["opt"], "Components and stocked assemblies")]:
            P = [b["procurement"] for b in A["bands"] if b["model"] == model]
            ax.bar(xs + off, P, .4, color=col, label=name)
        ax.set(xticks=xs, xticklabels=lab, xlabel="Mix-flexibility band around the programme", ylabel="External procurement")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False)
        fig.tight_layout(); fig.savefig(fig_dir / "fig_stage_a.png", dpi=220); plt.close(fig)
    # Stage B: procurement of the programme-proportional mix vs the least procurement with the same stock use
    fig, axs = plt.subplots(2, 2, figsize=(5.9, 5.4))
    axes = axs.ravel()[:len(QS)]; axs.ravel()[-1].axis("off")
    for ax, Q in zip(axes, QS):
        xs = np.arange(len(POOLS))
        pp = [D[(k, Q, 1.0)]["with_assemblies"]["proportional"]["procurement"] for k in POOLS]
        pd = [D[(k, Q, 1.0)]["with_assemblies"]["proportional"]["min_procurement_at_same_stock_use"] for k in POOLS]
        ax.bar(xs - .2, pp, .4, color=C["prop"], label="Programme-proportional mix")
        ax.bar(xs + .2, pd, .4, color=C["opt"], label="Efficient mix with the same stock use")
        for i, (a, b) in enumerate(zip(pp, pd)):
            ax.text(i + .2, b, f"−{100 * (1 - b / a):.0f}%", ha="center", va="bottom", fontsize=7.5)
        ax.set(title=f"Q = {Q}", xticks=xs, xticklabels=POOLS, xlabel="Candidate models", ylabel="External procurement")
    h, l = axes[0].get_legend_handles_labels()
    axs.ravel()[-1].legend(h, l, loc="center", frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir / "fig_stage_b.png", dpi=220); plt.close(fig)
    # GA gap by pool
    if G:
        fig, axes = plt.subplots(1, 2, figsize=(5.9, 3.3), sharey=True)
        for ax, Q in zip(axes, [q for q in QS if q != 20]):
            xs = np.arange(len(POOLS)); gg, m1, lo1, hi1, m2 = [], [], [], [], []
            for k in POOLS:
                y = D[(k, Q, 1.0)]["with_assemblies"]; opt = y["milp"]; gg.append(gap(opt, y["greedy"]["Z"]))
                if (k, Q) in G:
                    z = [gap(opt, x["Z"]) for x in G[(k, Q)]["runs"]["random"]]
                    z2 = [gap(opt, x["Z"]) for x in G[(k, Q)]["runs"]["greedy_seeded"]]
                    m1.append(np.mean(z)); lo1.append(min(z)); hi1.append(max(z)); m2.append(np.mean(z2))
                else:
                    m1.append(np.nan); lo1.append(np.nan); hi1.append(np.nan); m2.append(np.nan)
            m1, lo1, hi1 = map(np.array, (m1, lo1, hi1))
            ax.plot(xs, gg, "o-", color=C["greedy"], label="Greedy")
            ax.errorbar(xs + .07, m1, yerr=[np.clip(m1 - lo1, 0, None), np.clip(hi1 - m1, 0, None)], fmt="s--", color=C["ga"], capsize=3, label="GA (mean, best-worst)")
            ax.plot(xs - .07, m2, "^:", color=C["gas"], label="Greedy-seeded GA (mean)")
            ax.axhline(0, color="gray", lw=.7)
            ax.set(title=f"Q = {Q}", xticks=xs, xticklabels=POOLS, xlabel="Candidate models", ylabel="Gap (% of plan material value)")
        axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(fig_dir / "fig_gap_by_pool.png", dpi=220); plt.close(fig)
    # runtime
    fig, ax = plt.subplots(figsize=(5.9, 3.6))
    for i, k in enumerate(POOLS):
        for Q in QS:
            y = D[(k, Q, 1.0)]["with_assemblies"]; first = (i == 0 and Q == QS[0])
            ax.scatter(k, y["milp"]["seconds"], color=C["milp"], marker="o", label="MILP" if first else None)
            ax.scatter(k, y["greedy"]["seconds"], color=C["greedy"], marker="D", label="Greedy" if first else None)
            if (k, Q) in G:
                ax.scatter(k, st.mean(x["seconds"] for x in G[(k, Q)]["runs"]["random"]), color=C["ga"], marker="s", label="GA (one run)" if first else None)
    ax.set(yscale="log", xlabel="Candidate models", ylabel="Seconds per solve (log)", xticks=POOLS); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig_runtime.png", dpi=220); plt.close(fig)
    # stock use vs procurement fronts with the lambda points
    pf = res / "pareto.json"
    if pf.exists():
        P = json.loads(pf.read_text())
        with open(res / "table_pareto.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pool", "Q", "point", "kind", "eps", "procurement", "stock_used_total", "Z_lambda1",
                                           "stage1_status", "stage1_gap", "stage2_status", "stage2_gap"])
            for c in P:
                for i, x in enumerate(c["front"]):
                    w.writerow([c["pool"], c["Q"], i, x["kind"], "" if x["eps"] is None else round(x["eps"], 3), round(x["procurement"], 3),
                                round(x["stock_used_total"], 3), round(x["Z"], 3), x["stage1"]["status"], x["stage1"]["mip_gap"],
                                x["stage2"]["status"], x["stage2"]["mip_gap"]])
        cases = [c for c in P if (c["pool"], c["Q"]) in [(9, 130), (25, 130), (15, 150), (35, 150)]]
        fig, axes = plt.subplots(2, 2, figsize=(5.9, 5.4))
        for ax, c in zip(axes.ravel(), cases):
            ax.plot([x["procurement"] for x in c["front"]], [x["stock_used_total"] for x in c["front"]], "o-", color=C["greedy"], ms=3, label="Efficient plans")
            y = D[(c["pool"], c["Q"], 1.0)]["with_assemblies"]
            for key, mk, lab_ in [(None, "*", "λ = 1"), ("0.5", "v", "λ = 0.5"), ("2.0", "^", "λ = 2")]:
                pt = y["milp"] if key is None else y["lambda"][key]
                ax.plot(pt["procurement"], U(pt), mk, color=C["ga"], ms=9 if key is None else 6, label=lab_)
            pr = y["proportional"]
            ax.plot(pr["procurement"], U(pr), "x", color="k", ms=7, label="Proportional mix")
            ax.set(title=f"{c['pool']} models / Q = {c['Q']}", xlabel="External procurement", ylabel="Stock used")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        axes.ravel()[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(fig_dir / "fig_pareto.png", dpi=220); plt.close(fig)


def main(res: Path = RESULTS):
    det = json.loads((res / "deterministic.json").read_text())
    D = {(r["pool"], r["Q"], r["stock_factor"]): r for r in det["scenarios"]}
    G = {}
    for fpath in sorted(glob.glob(str(res / "ga" / "ga_*.json"))):
        g = json.loads(Path(fpath).read_text()); G[(g["pool"], g["Q"])] = g
    stage_b_tables(res, D, G)
    A = stage_a_tables(res)
    figures(res, D, G, A)
    print("tables and figures written to", res)


if __name__ == "__main__":
    main()
