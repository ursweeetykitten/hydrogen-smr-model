# -*- coding: utf-8 -*-
"""Generate portfolio figures from the computed CSV results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
OUT = Path("figures"); OUT.mkdir(exist_ok=True)

NAVY = "#1F4E79"
ACC = "#C0504D"
GREY = "#595959"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#B0B0B0", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})

# ---------- 1. Dry gas composition through the process ----------
m = pd.read_csv(R / "03a_material_balance_by_stream.csv")
stages = ["feed_to_SMR", "SMR_out", "HTS_out", "LTS_out"]
labels = ["Feed", "After SMR", "After HTS", "After LTS"]
comps = ["CH4", "CO", "CO2", "H2"]
colors = {"CH4": "#8C8C8C", "CO": ACC, "CO2": "#E8A33D", "H2": NAVY}
data = {c: [] for c in comps}
for s in stages:
    sub = m[(m.stream == s)]
    for c in comps:
        v = sub[sub.component == c]["dry_mol_frac"]
        data[c].append(float(v.iloc[0]) * 100 if len(v) else 0.0)

x = np.arange(len(stages)); w = 0.2
fig, ax = plt.subplots(figsize=(7.2, 4.3))
for i, c in enumerate(comps):
    ax.bar(x + (i - 1.5) * w, data[c], w, label=c, color=colors[c])
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Dry mole fraction, %")
ax.set_title("Gas composition through SMR → HTS → LTS", color=NAVY, fontweight="bold")
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.savefig(OUT / "01_composition_through_stages.png"); plt.close(fig)

# ---------- 2. H2 build-up through the chain ----------
h2 = []
for s in ["SMR_out", "HTS_out", "LTS_out"]:
    sub = m[(m.stream == s) & (m.component == "H2")]
    h2.append(float(sub["Nm3_h"].iloc[0]))
prod = pd.read_csv(R / "04c_final_product_and_tail_gas_summary.csv").iloc[0]
h2_product = float(prod["actual_H2_product_Nm3_h"])

fig, ax = plt.subplots(figsize=(7.2, 4.3))
xs = ["After\nSMR", "After\nHTS", "After\nLTS", "Product\n(after PSA)"]
ys = h2 + [h2_product]
bars = ax.bar(xs, ys, color=[NAVY, NAVY, NAVY, ACC], width=0.6)
for b, y in zip(bars, ys):
    ax.text(b.get_x() + b.get_width() / 2, y + 120, f"{y:,.0f}", ha="center", fontsize=10, color=GREY)
ax.set_ylabel("H₂ flow, Nm³/h")
ax.set_title("Hydrogen build-up and final product", color=NAVY, fontweight="bold")
ax.set_ylim(0, max(ys) * 1.15)
fig.savefig(OUT / "02_h2_buildup.png"); plt.close(fig)

# ---------- 3. Design space explored by the optimizer ----------
hist = pd.read_csv(R / "02a_operating_optimization_history.csv")
hist = hist.dropna(subset=["SC", "X_CH4_pct", "score"])
feas = hist[hist["score"] < 20.0]
infeas = hist[hist["score"] >= 20.0]
best = pd.read_csv(R / "02b_best_operating_point.csv").iloc[0]
fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.scatter(infeas["SC"], infeas["X_CH4_pct"], s=14, color="#CFCFCF", label="infeasible / penalized")
ax.scatter(feas["SC"], feas["X_CH4_pct"], s=22, color=NAVY, label="feasible candidates")
ax.scatter([best["SC"]], [best["X_CH4_pct"]], s=180, marker="*", color=ACC,
           edgecolor="white", linewidth=0.8, zorder=5, label="selected optimum")
ax.axhline(90, color=GREY, ls="--", lw=1.0)
ax.set_xlabel("Steam-to-carbon ratio S/C"); ax.set_ylabel("CH₄ conversion, %")
ax.set_title("Operating points explored by differential evolution", color=NAVY, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.savefig(OUT / "03_design_space.png"); plt.close(fig)

# ---------- 4. CH4 conversion vs SMR catalyst mass ----------
sweep = pd.read_csv(R / "01c_catalyst_chain_mass_sweep.csv")
g = sweep.groupby("W_smr_loaded_kg")["X_CH4_pct"].max().reset_index()
g = g.sort_values("W_smr_loaded_kg")
fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.plot(g["W_smr_loaded_kg"], g["X_CH4_pct"], color=NAVY, lw=2)
ax.axhline(90, color=ACC, ls="--", lw=1.3, label="90 % target")
ax.set_xlabel("SMR catalyst (loaded), kg"); ax.set_ylabel("CH₄ conversion, %")
ax.set_title("CH₄ conversion vs SMR catalyst mass", color=NAVY, fontweight="bold")
ax.legend(frameon=False)
fig.savefig(OUT / "04_conversion_vs_catalyst.png"); plt.close(fig)

print("Figures saved:", sorted(p.name for p in OUT.glob("*.png")))
