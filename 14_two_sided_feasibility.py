"""Is the CME-Binance reopen convergence tradable for a RETAIL trader, two-sided, after all costs?
Gross P&L: 'hedged' trade from 11_q3_reopen_halflife.py (results/q3_trade_test.csv): at reopen+L min, if the
basis-adjusted gap |g| > threshold, short the rich venue and buy the cheap one; exit after H min.
Gross = -sign(g_L) * (g_{L+H} - g_L), on last-trade prices (Binance 1m close, CME 1s close). Jan-Sep 2026.
Round-trip cost (bps) = Binance fees (2 legs) + Binance spread + CME fees (2 legs) + CME spread
  CME spread = half the post-reopen spread (entry) + half the normal spread (exit), measured from mbp-1
  Binance spread = trade-based effective-spread proxy (likely optimistic - no real quotes)
Scenarios: (A) retail taker on both venues; (B) Binance maker (limit order, assumes it fills; no Binance spread
cost); CME is always taker. Fees: Binance USD-M VIP0 assumed 5.0/2.0 bps taker/maker per side (check the actual
TradFi-perp fee schedule); CME retail all-in ~$1.50/side per micro contract ($3 where no micro exists)."""
import pandas as pd, numpy as np
c = pd.read_csv("results/costs_staleness.csv").set_index("asset")
CONTRACT = {"GC": ("MGC micro", 10, 1.5), "SI": ("SIL micro", 1000, 1.5), "PL": ("PL standard", 50, 3.0),
            "PA": ("PA standard", 100, 3.0), "HG": ("MHG micro", 2500, 1.5), "CL": ("MCL micro", 100, 1.5),
            "BZ": ("BZ standard", 1000, 3.0), "NG": ("NG standard", 10000, 3.0)}
FEE_T, FEE_M = 5.0, 2.0
rows = []
for a, (name, mult, usd) in CONTRACT.items():
    notional = c.loc[a, "price_level"] * mult
    cme_fee = usd / notional * 1e4
    cme_spread = 0.5 * c.loc[a, "cme_spread_post_reopen_bps"] + 0.5 * c.loc[a, "cme_spread_bps"]
    bin_spread = c.loc[a, "bin_eff_spread_bps"]
    rows.append(dict(asset=a, cme_contract=name, notional_usd=round(notional), cme_fee_rt=2 * cme_fee,
                     cme_spread_rt=cme_spread, bin_spread_rt=bin_spread,
                     cost_taker=2 * FEE_T + bin_spread + 2 * cme_fee + cme_spread,
                     cost_maker=2 * FEE_M + 2 * cme_fee + cme_spread))
cost = pd.DataFrame(rows).set_index("asset"); cost.round(2).to_csv("results/costs_two_sided.csv")
print(cost.round(2).to_string(), "\n")
t = pd.read_csv("results/q3_trade_test.csv")
t = t[t.asset.isin(CONTRACT)].copy()
t = t.join(cost[["cost_taker", "cost_maker"]], on="asset")
for s in ["taker", "maker"]:
    t[f"net_{s}"] = t.hedged_mean_bps - t[f"cost_{s}"]
    t[f"t_net_{s}"] = np.where(t.hedged_mean_bps != 0, t.hedged_t * t[f"net_{s}"] / t.hedged_mean_bps, np.nan)
t.to_csv("results/two_sided_net_all.csv", index=False)
# pre-specified rule (not optimised): enter 1 min after reopen, hold 15 min, trade only if |gap| > 20 bps
fixed = t[(t.entry_min == 1) & (t.hold_min == 15) & (t.min_gap_bps == 20)]
cols = ["asset", "kind", "trades", "mean_abs_signal_bps", "hedged_mean_bps", "hedged_t", "cost_taker", "net_taker", "t_net_taker", "cost_maker", "net_maker", "t_net_maker"]
fixed[cols].to_csv("results/two_sided_fixed_rule.csv", index=False)
print("FIXED RULE (entry 1m, hold 15m, |gap|>20bps):"); print(fixed[cols].round(2).to_string(index=False))
# best in-sample cell per asset/kind among cells with >=8 trades (optimistic upper bound - data-snooped)
b = t[t.trades >= 8].sort_values("net_maker", ascending=False).groupby(["asset", "kind"]).head(1)
print("\nBEST IN-SAMPLE CELL (>=8 trades; optimistic):"); print(b[["asset", "kind", "entry_min", "hold_min", "min_gap_bps"] + cols[2:]].round(2).to_string(index=False))
b.to_csv("results/two_sided_best_insample.csv", index=False)
