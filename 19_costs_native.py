"""Retail two-sided cost model re-done with NATIVE Binance spreads (Tardis book_ticker, 7 reopen days),
plus Binance top-of-book depth vs the CME hedge contract size, and a summary of the stale-price check.
Binance spread round trip = half the median spread 1-6 min after the reopen (entry) + half the normal spread (exit).
Outputs: results/costs_two_sided_native.csv, results/two_sided_fixed_rule_native.csv, results/q6_gap_check_summary.csv"""
import numpy as np, pandas as pd
c = pd.read_csv("results/costs_two_sided.csv").set_index("asset")
s = pd.read_csv("results/q6_spread_state.csv").pivot(index="asset", columns="state", values="spread_bps")
dep = pd.read_csv("results/q6_spread_state.csv").pivot(index="asset", columns="state", values="depth_usd")
c["bin_spread_native_rt"] = 0.5 * s["post_reopen_1_6m"] + 0.5 * s["open_normal"]
c["cost_taker_native"] = c.cost_taker - c.bin_spread_rt + c.bin_spread_native_rt
c["bin_depth_usd_normal"] = dep["open_normal"]; c["bin_depth_usd_post"] = dep["post_reopen_1_6m"]
c["hedge_vs_depth"] = c.notional_usd / c.bin_depth_usd_post
c.round(2).to_csv("results/costs_two_sided_native.csv")
print(c[["cme_contract", "notional_usd", "bin_spread_rt", "bin_spread_native_rt", "cost_taker", "cost_taker_native",
         "cost_maker", "bin_depth_usd_normal", "bin_depth_usd_post", "hedge_vs_depth"]].round(2).to_string())
f = pd.read_csv("results/two_sided_fixed_rule.csv").join(c[["cost_taker_native"]], on="asset")
f["net_taker_native"] = f.hedged_mean_bps - f.cost_taker_native
f.to_csv("results/two_sided_fixed_rule_native.csv", index=False)
print(f[["asset", "kind", "trades", "hedged_mean_bps", "net_taker", "net_taker_native", "net_maker"]].round(1).to_string(index=False))
g = pd.read_csv("results/q6_gap_check.csv")
dense = g.asset.isin(["GC", "SI", "CL", "BZ"])
sig_last, sig_true = g.g1_last.abs() > 20, g.g1_true.abs() > 20
summ = dict(events=len(g), med_abs_diff_g1_bps=(g.g1_last - g.g1_true).abs().median(),
            med_abs_diff_dense=(g.g1_last - g.g1_true)[dense].abs().median(),
            med_abs_diff_sparse=(g.g1_last - g.g1_true)[~dense].abs().median(),
            max_abs_diff=(g.g1_last - g.g1_true).abs().max(),
            sign_agree=(np.sign(g.g1_last) == np.sign(g.g1_true)).mean(),
            signals_last=int(sig_last.sum()), signals_true=int(sig_true.sum()), signals_both=int((sig_last & sig_true).sum()),
            pnl_last_signals=g.loc[sig_last, "pnl_last"].mean(), pnl_true_signals=g.loc[sig_true, "pnl_true"].mean(),
            corr_pnl=np.corrcoef(g.pnl_last, g.pnl_true)[0, 1], entry_spread_med=g.spread1_bps.median())
pd.DataFrame([summ]).to_csv("results/q6_gap_check_summary.csv", index=False)
print(pd.Series(summ).round(2).to_string())
print(g.loc[sig_last | sig_true, ["asset", "kind", "reopen", "g1_last", "g1_true", "pnl_last", "pnl_true"]].round(1).to_string(index=False))
