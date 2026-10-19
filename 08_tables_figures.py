"""Combine Q1/Q2 results into paper tables and figures."""
import glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ORDER = ["BTC", "ES", "GC", "SI", "PL", "PA", "HG", "CL", "BZ", "NG"]
NAMES = {"GC": "Gold", "SI": "Silver", "PL": "Platinum", "PA": "Palladium", "HG": "Copper", "CL": "WTI crude",
         "BZ": "Brent crude", "NG": "Natural gas", "ES": "S&P 500", "BTC": "Bitcoin"}
BIN = {"GC": "XAUUSDT", "SI": "XAGUSDT", "PL": "XPTUSDT", "PA": "XPDUSDT", "HG": "COPPERUSDT", "CL": "CLUSDT",
       "BZ": "BZUSDT", "NG": "NATGASUSDT", "ES": "SPYUSDT", "BTC": "BTCUSDT"}
# liquidity in the tick window (CME open hours not filtered: simple daily averages over 24 Apr - 5 May)
liq = []
for r, b in BIN.items():
    c = pd.read_parquet(f"data/cme/{r}_tbbo.parquet", columns=["size"])
    nb = sum(sum(1 for _ in open(f)) - 1 for f in glob.glob(f"data/binance/{b}/aggTrades/*.csv"))
    import pyarrow.parquet as pq
    nq = pq.ParquetFile(f"data/cme/{r}_mbp1.parquet").metadata.num_rows
    liq.append(dict(asset=r, cme_trades_per_day=len(c) / 10, cme_quote_updates_per_day=nq / 10,
                    binance_aggtrades_per_day=nb / 12))
liq = pd.DataFrame(liq)
q1 = pd.read_csv("results/q1_summary.csv"); cu = pd.read_csv("results/q1b_catchup.csv")
q2 = pd.concat(pd.read_csv(f) for f in glob.glob("results/q2_summary_*.csv") if "mbp1" not in f)
w = q1[q1.kind == "weekend"].set_index("asset"); d = q1[q1.kind == "daily_break"].set_index("asset")
cw = cu[cu.kind == "weekend"].set_index("asset")
t = pd.DataFrame(index=ORDER)
t["Market"] = [NAMES[a] for a in ORDER]; t["Binance perp"] = [BIN[a] for a in ORDER]
t["Weekends"] = w.n; t["Weekend slope"] = w.slope.round(2); t["Weekend slope SE"] = w.slope_se.round(2)
t["Weekend R2"] = w.r2.round(2); t["Sign hit rate"] = w.hit_rate.round(2)
t["Break slope"] = d.slope.round(2); t["Break R2"] = d.r2.round(2)
t["Catch-up 5m"] = cw.catchup_5.round(2); t["Catch-up 30m"] = cw.catchup_30.round(2)
q2 = q2.set_index("asset")
t["LLR (CME leads >1)"] = q2.LLR.round(2); t["Peak lag (s)"] = q2.peak_lag_s
t["IS CME low"] = q2.IS_cme_lo.round(2); t["IS CME high"] = q2.IS_cme_hi.round(2); t["CS CME"] = q2.CS_cme.round(2)
t["Binance quote coverage"] = q2.coverage.round(2)
q2m = pd.concat(pd.read_csv(f) for f in glob.glob("results/q2_summary_*_mbp1.csv")).set_index("asset")
t["LLR full quotes"] = q2m.LLR.round(1); t["Peak lag full (s)"] = q2m.peak_lag_s
t["IS CME low full"] = q2m.IS_cme_lo.round(2); t["IS CME high full"] = q2m.IS_cme_hi.round(2)
t["CS CME full"] = q2m.CS_cme.round(2); t["Gap half-life full (s)"] = q2m.gap_halflife_s.round(1)
t = t.join(liq.set_index("asset").round(0))
t.to_csv("results/table_main.csv"); print(t.T.to_string())

# Figure 1: weekend slope with 95% CI, and R2
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
y = np.arange(len(ORDER))[::-1]
ax[0].errorbar(t["Weekend slope"], y, xerr=1.96 * t["Weekend slope SE"], fmt="o", capsize=3)
ax[0].axvline(1, ls="--", lw=0.8); ax[0].axvline(0, lw=0.5, color="k")
ax[0].set_yticks(y); ax[0].set_yticklabels([f"{NAMES[a]} ({a})" for a in ORDER])
ax[0].set_xlabel("slope of CME reopen gap on Binance weekend return (95% CI)")
ax[1].barh(y, t["Weekend R2"]); ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xlim(0, 1)
ax[1].set_xlabel("R² (share of CME gap explained by Binance)")
fig.tight_layout(); fig.savefig("results/fig1_weekend_pricing.png", dpi=200)
# Figure 2: information share under both CME quote measurements
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for yy, a in zip(y, ORDER):
    ax.plot([t.loc[a, "IS CME low"], t.loc[a, "IS CME high"]], [yy + 0.15, yy + 0.15], lw=5,
            solid_capstyle="butt", color="#9aa5b1", label="CME quotes at trades only (tbbo)" if a == ORDER[0] else None)
    ax.plot([t.loc[a, "IS CME low full"], t.loc[a, "IS CME high full"]], [yy - 0.15, yy - 0.15], lw=5,
            solid_capstyle="butt", color="#1f4e79", label="every CME quote change (mbp-1)" if a == ORDER[0] else None)
ax.axvline(0.5, ls="--", lw=0.8, color="k"); ax.set_xlim(0, 1)
ax.set_yticks(y); ax.set_yticklabels([f"{NAMES[a]} ({a})" for a in ORDER])
ax.set_xlabel("CME Hasbrouck information share (lower-upper bound), 100 ms grid")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("results/fig2_information_share.png", dpi=200)
print("saved")
