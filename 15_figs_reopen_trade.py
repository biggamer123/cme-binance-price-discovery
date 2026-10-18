"""Figures for reopen convergence, who closes the gap, two-sided trade economics, data quality."""
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

# ---------- Fig 6: who closes the gap (weekends, 5 min)
s = pd.read_csv("results/q3_halflife_summary.csv"); w = s[(s.kind == "weekend") & s.asset.isin(COL)].set_index("asset")
order = ["CL", "BZ", "NG", "GC", "SI", "PL", "PA", "HG"]
fig, ax = plt.subplots(figsize=(5.4, 2.9)); y = np.arange(len(order))[::-1]; h = 0.36
ax.barh(y + h / 2, w.loc[order, "bin_share_5m"], height=h, color="#2a78d6", label="Binance moved towards CME")
ax.barh(y - h / 2, w.loc[order, "cme_share_5m"], height=h, color="#eb6834", label="CME moved towards Binance")
for yy, a in zip(y, order):
    for v, off in [(w.loc[a, "bin_share_5m"], h / 2), (w.loc[a, "cme_share_5m"], -h / 2)]:
        ax.text(max(v, 0) + 0.02, yy + off, f"{v:.2f}", va="center", fontsize=6.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([NAME[a] for a in order]); ax.axvline(0, color=INK2, lw=0.6)
ax.axhline(4.5, color=GRID, lw=1); ax.text(1.18, 6, "energy", fontsize=7, color=INK2, ha="right"); ax.text(1.18, 2, "metals", fontsize=7, color=INK2, ha="right")
ax.set_xlabel("share of the weekend reopen gap closed within 5 min"); ax.set_xlim(-0.1, 1.2)
ax.legend(loc="lower right", fontsize=7); grid(ax, "x")
fig.tight_layout(); fig.savefig("results/fig6_who_closes.png", dpi=200); plt.close(fig)

# ---------- Fig 7: two-sided trade, gross P&L vs retail cost
f = pd.read_csv("results/two_sided_fixed_rule.csv"); cost = pd.read_csv("results/costs_two_sided_native.csv").set_index("asset")
cost["cost_taker"] = cost["cost_taker_native"]   # native Binance spreads (Tardis book_ticker)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
order = ["GC", "SI", "PL", "PA", "HG", "CL", "BZ", "NG"]; x = np.arange(len(order))
for ax, kind, title in [(axs[0], "weekend", "(a) Weekend reopens"), (axs[1], "daily_break", "(b) Daily-break reopens")]:
    d = f[f.kind == kind].set_index("asset")
    for i, a in enumerate(order):
        if a in d.index:
            m, t, n = d.loc[a, "hedged_mean_bps"], d.loc[a, "hedged_t"], int(d.loc[a, "trades"])
            se = abs(m / t) if t else 0
            ax.bar(i, m, width=0.6, color=COL[a], yerr=1.96 * se, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
            ax.text(i, -8.5, f"n={n}", ha="center", fontsize=6, color=INK2)
        else:
            ax.text(i, -8.5, "no trades", ha="center", fontsize=6, color=INK2)
        ax.hlines(cost.loc[a, "cost_taker"], i - 0.38, i + 0.38, color=INK, lw=1.6)
        ax.hlines(cost.loc[a, "cost_maker"], i - 0.38, i + 0.38, color=INK, lw=1.0, ls=(0, (2, 1.5)))
    ax.set_xticks(x); ax.set_xticklabels([NAME[a] for a in order], rotation=30, ha="right")
    ax.axhline(0, color=INK2, lw=0.6); ax.set_title(title); ax.set_ylim(-11, 60); grid(ax)
axs[0].set_ylabel("bps per trade")
axs[1].plot([], [], color=INK, lw=1.6, label="retail cost, taker"); axs[1].plot([], [], color=INK, lw=1.0, ls=(0, (2, 1.5)), label="retail cost, maker")
from matplotlib.patches import Patch
h_, l_ = axs[1].get_legend_handles_labels(); h_.append(Patch(color="#9aa5b1")); l_.append("gross P&L, 95% CI")
axs[1].legend(h_, l_, fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig("results/fig7_two_sided_trade.png", dpi=200); plt.close(fig)

# ---------- Fig 8: data quality - Binance quote staleness, CME spreads
q = pd.read_csv("results/costs_staleness.csv").set_index("asset")
order = ["BTC", "CL", "GC", "BZ", "SI", "PL", "NG", "PA", "ES", "HG"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
y = np.arange(len(order))[::-1]
dense = q.loc[order, "bin_quote_age_median_s"] <= 15
ax[0].barh(y, q.loc[order, "bin_quote_age_median_s"], height=0.55, color=np.where(dense, "#2a78d6", "#9aa5b1"))
ax[0].errorbar(q.loc[order, "bin_quote_age_p90_s"], y, fmt="|", color=INK2, ms=7)
import glob as _g
nat = pd.concat(pd.read_csv(f_) for f_ in _g.glob("results/q5_real_1may_*.csv")).set_index("asset")["true_quote_age_med_s"]
ax[0].plot(nat.loc[order].values, y, ls="", marker="D", ms=3.6, color=INK, mec="white", mew=0.5, zorder=5)
from matplotlib.lines import Line2D
ax[0].legend([Line2D([], [], color="#2a78d6", lw=5), Line2D([], [], ls="", marker="|", color=INK2, ms=7),
              Line2D([], [], ls="", marker="D", ms=3.6, color=INK)],
             ["rebuilt, median", "rebuilt, 90th pct", "native quote, median (1 May)"], fontsize=6,
             loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, handlelength=1.5, columnspacing=1.0)
for yy, a in zip(y, order): ax[0].text(q.loc[a, "bin_quote_age_median_s"] * 1.15, yy, f"{q.loc[a,'bin_quote_age_median_s']:.1f}s", va="center", fontsize=6.5, color=INK2)
ax[0].set_xscale("log"); ax[0].set_yticks(y); ax[0].set_yticklabels([NAME[a] for a in order])
ax[0].set(xlabel="Binance quote age (s, log scale)", title="(a) Binance quote age")
ax[0].axvline(15, color=INK2, lw=0.6, ls=":"); ax[0].text(16, y[0] + 0.1, "sparse >15 s", fontsize=6.5, color=INK2); grid(ax[0], "x")
o2 = ["GC", "SI", "PL", "PA", "HG", "CL", "BZ", "NG", "ES", "BTC"]; x = np.arange(len(o2)); wdt = 0.38
ax[1].bar(x - wdt / 2, q.loc[o2, "cme_spread_bps"], width=wdt, color="#2a78d6", label="normal hours")
ax[1].bar(x + wdt / 2, q.loc[o2, "cme_spread_post_reopen_bps"], width=wdt, color="#eb6834", label="1-6 min after reopen")
ax[1].set_xticks(x); ax[1].set_xticklabels([NAME[a] for a in o2], rotation=30, ha="right")
ax[1].set(ylabel="CME bid-ask spread (bps)", title="(b) CME spread, median"); ax[1].legend(fontsize=6.5); grid(ax[1])
fig.tight_layout(); fig.savefig("results/fig8_data_quality.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok")
