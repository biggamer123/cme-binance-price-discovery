"""Fig. 1 of the paper: share of the basis-adjusted CME-Binance gap still open after the reopen (writes results/fig5_reopen_convergence.png)."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"GC": "#2a78d6", "SI": "#eb6834", "PL": "#1baf7a", "PA": "#eda100", "HG": "#e87ba4",
       "CL": "#008300", "BZ": "#6250d6", "NG": "#e34948"}
NAME = {"GC": "Gold", "SI": "Silver", "PL": "Platinum", "PA": "Palladium", "HG": "Copper",
        "CL": "WTI", "BZ": "Brent", "NG": "Nat. gas", "ES": "S&P 500", "BTC": "Bitcoin"}
METALS, ENERGY = ["GC", "SI", "PL", "PA", "HG"], ["CL", "BZ", "NG"]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False})
def grid(ax, axis="y"): ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)

# ---------- Fig 5: how fast the reopen gap closes
cm = pd.read_csv("results/q3_halflife_curves.csv"); ct = pd.read_csv("results/q3b_tick_curves_post.csv")
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.75), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
for g, c, lab in [("energy", COL["CL"], "Energy"), ("metals", COL["GC"], "Metals")]:
    for kind, ls in [("weekend", "-"), ("daily_break", "--")]:
        d = ct[(ct.group == g) & (ct.kind == kind) & (ct.s <= 60)]
        ax[0].plot(d.s, d.frac_open, ls=ls, lw=2, color=c, label=f"{lab}, {'weekends' if kind=='weekend' else 'daily breaks'}")
ax[0].set(xlabel="seconds after CME reopen", ylabel="share of gap still open", title="(a) First minute (ticks)", ylim=(-0.4, 1.3))
ax[0].axhline(0, color=INK2, lw=0.6); ax[0].axhline(0.5, color=INK2, lw=0.6, ls=":"); ax[0].legend(fontsize=6, loc="upper right", handlelength=2.2); grid(ax[0])
ends = []
for a_, assets, title in [(ax[1], ENERGY, "(b) Energy, weekends"), (ax[2], METALS, "(c) Metals, weekends")]:
    for a in assets:
        d = cm[(cm.asset == a) & (cm.kind == "weekend") & (cm.k_min >= 0) & (cm.k_min <= 30)]
        a_.plot(d.k_min, d.frac_open, lw=2, color=COL[a], marker="o", ms=2.5, label=NAME[a])
        if assets is METALS:
            ends.append([d.frac_open.iloc[-1], NAME[a], d.k_min.iloc[-1]])
    if assets is METALS:
        ends.sort(); 
        for j in range(1, len(ends)):
            if ends[j][0] - ends[j-1][0] < 0.09: ends[j][0] = ends[j-1][0] + 0.09
        for yv, nm, xv in ends:
            a_.text(xv + 0.6, yv, nm, va="center", fontsize=6.5, color=INK)
    if assets is ENERGY: a_.legend(fontsize=7, loc="upper right")
    a_.axhline(0, color=INK2, lw=0.6); a_.axhline(0.5, color=INK2, lw=0.6, ls=":")
    a_.set(xlabel="minutes after CME reopen", title=title, xlim=(0, 39), ylim=(-0.4, 1.3)); grid(a_)
ax[1].set_ylabel("share of gap still open")
fig.tight_layout(); fig.savefig("results/fig5_reopen_convergence.png", dpi=200); plt.close(fig)
