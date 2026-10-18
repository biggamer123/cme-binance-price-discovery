"""Cross-correlation of 100 ms returns, full CME quotes (mbp-1). Sized for a 12 cm LNCS text width."""
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["BTC", "ES", "GC", "SI", "PL", "PA", "HG", "CL", "BZ", "NG"]
NAMES = {"GC": "Gold", "SI": "Silver", "PL": "Platinum", "PA": "Palladium", "HG": "Copper", "CL": "WTI",
         "BZ": "Brent", "NG": "Nat. gas", "ES": "S&P 500", "BTC": "Bitcoin"}
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8,
                     "axes.titleweight": "bold"})
fig, axs = plt.subplots(2, 5, figsize=(7.4, 3.05), sharex=True)
for ax, a in zip(axs.ravel(), ORDER):
    x = pd.read_csv(f"results/q2_xcorr_{a}_mbp1.csv"); x = x[x.lag_s.abs() <= 2]
    ax.bar(x.lag_s, x.rho, width=0.085, color=["#1f4e79" if l > 0 else "#9aa5b1" for l in x.lag_s])
    ax.axvline(0, lw=0.6, color=INK2); ax.axhline(0, lw=0.5, color=INK2)
    ax.set_title(NAMES[a]); ax.yaxis.set_major_locator(MaxNLocator(3)); ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_xticks([-2, -1, 0, 1, 2])
fig.supxlabel("lag k (s);  k > 0: Binance moves after CME (CME leads)", fontsize=7.5, color=INK)
fig.supylabel(r"corr($r^{CME}_t$, $r^{Bin}_{t+k}$)", fontsize=7.5, color=INK)
fig.tight_layout(); fig.savefig("results/fig3_xcorr.png", dpi=220); print("ok")
