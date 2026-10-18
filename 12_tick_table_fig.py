"""Table + figure for the tick-time robustness (Hayashi-Yoshida + directional response)."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ORDER = ["BTC", "ES", "GC", "SI", "PL", "PA", "HG", "CL", "BZ", "NG"]
t = pd.concat(pd.read_csv(f"results/q2_tick_{a}.csv") for a in ORDER).set_index("asset").loc[ORDER]
cols = ["cme_moves", "binance_moves", "hy_peak_lag_ms", "hy_peak_rho", "hy_LLR",
        "bin_after_cme_100ms", "bin_after_cme_1000ms", "cme_after_bin_100ms", "cme_after_bin_1000ms"]
t[cols].to_csv("results/table_tick.csv"); print(t[cols].round(3).to_string())
fig, ax = plt.subplots(figsize=(7, 3.4))
style = {"GC": ("Gold", "-"), "CL": ("WTI", "--"), "BZ": ("Brent", "-."), "BTC": ("Bitcoin", ":"), "PA": ("Palladium", (0, (1, 1)))}
for a, (name, ls) in style.items():
    x = pd.read_csv(f"results/q2_hy_{a}.csv")
    ax.plot(x.lag_ms, x.hy_rho, ls=ls, marker="o", ms=3, lw=1.3, label=name)
ax.set_xscale("symlog", linthresh=50); ax.axvline(0, color="k", lw=0.6)
ticks = [-5000, -1000, -200, -50, 0, 50, 200, 1000, 5000]; ax.set_xticks(ticks); ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=8)
ax.axvspan(50, 100, color="0.85", zorder=0, label="~Chicago-Tokyo latency")
ax.set_xlabel("lag L (ms); L>0: Binance after CME"); ax.set_ylabel("Hayashi-Yoshida correlation")
ax.legend(fontsize=8, ncol=3, frameon=False); fig.tight_layout(); fig.savefig("results/fig4_hy.png", dpi=200); print("ok")
