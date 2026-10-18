"""Hayashi-Yoshida lead-lag on 1 May 2026 (00:00-21:00 UTC), CME mbp-1 vs (a) trade-reconstructed Binance mid,
(b) native Binance best bid/offer (Tardis book_ticker). Same day, same window - only the Binance quote differs."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8, "axes.titleweight": "bold",
                     "legend.frameon": False})
SER = [("GC", "Gold", "#2a78d6", "o", "-"), ("PL", "Platinum", "#1baf7a", "s", "-"), ("BZ", "Brent", "#6250d6", "^", "-"),
       ("HG", "Copper", "#e87ba4", "D", "-"), ("BTC", "Bitcoin", INK2, "", "--")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.75), sharey=True)
for ax, col, title in [(axs[0], "rec", "(a) Rebuilt Binance quote (from trades)"), (axs[1], "true", "(b) Native Binance quote (bookTicker)")]:
    ax.axvspan(50, 100, color="#eeeeea", zorder=0)
    for root, name, c, mk, ls in SER:
        d = pd.read_csv(f"results/q5_hy_1may_{root}.csv")
        ax.plot(d.lag_ms, d[col], color=c, marker=mk, ms=2.6, lw=1.4, ls=ls, label=name)
        pk = d.loc[d[col].idxmax()]
        ax.plot([pk.lag_ms], [pk[col]], marker=mk or "o", ms=5.5, color=c, mec="white", mew=0.8, zorder=5)
    ax.set_xscale("symlog", linthresh=50); ax.axvline(0, color=INK2, lw=0.6)
    ticks = [-5000, -500, -50, 0, 50, 500, 5000]; ax.set_xticks(ticks); ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.set_title(title); ax.set_xlabel("lag L (ms);  L > 0: Binance moves after CME")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel("Hayashi-Yoshida correlation")
axs[1].text(60, 0.03, "50-100 ms", fontsize=6.5, color=INK2)
axs[0].legend(loc="upper left", fontsize=6.5, ncol=1, handlelength=2.2)
fig.tight_layout(); fig.savefig("results/fig9_hy_native.png", dpi=220); print("ok")
