"""Summaries of 24_native_multiday.py across days (for Table 4 / Table 5 and the text).
  results/q7_rq3_summary.csv     per market: days with CME ticks, median and range of LLR, IS^C bounds,
                                 HY peak, D both ways, native spread
  results/q7_reopen_summary.csv  per kind of reopen: events, stale-price check (native mid vs last trade at 1 min),
                                 signals, gross at mid, executed net at real bid/ask
  results/q7_trades.csv          every trade signal with its numbers"""
import os
import numpy as np, pandas as pd

ORDER = ["GC", "SI", "PL", "PA", "HG", "CL", "BZ", "NG", "ES", "BTC"]

if os.path.exists("results/q7_rq3_days.csv"):
    d = pd.read_csv("results/q7_rq3_days.csv")
    d = d[d.cme_ticks.astype(str).str.lower() == "true"]
    rows = []
    for a, g in d.groupby("asset"):
        r = dict(asset=a, days=len(g), open_hours=g.open_hours.sum(),
                 LLR_med=g.LLR.median(), LLR_min=g.LLR.min(), LLR_max=g.LLR.max(),
                 peak_ms_mode=g.peak_ms.mode().iloc[0], share_days_bin_lags=(g.LLR > 1).mean(),
                 IS_lo_med=g.IS_lo.median(), IS_hi_med=g.IS_hi.median(), IS_lo_min=g.IS_lo.min(),
                 HY_peak_med=g.HY_peak_ms.median(), HY_peak_min=g.HY_peak_ms.min(), HY_peak_max=g.HY_peak_ms.max(),
                 D_bin_after_cme=g.D_bin_after_cme.median(), D_cme_after_bin=g.D_cme_after_bin.median(),
                 spread_bps=g.native_spread_bps.median())
        rows.append(r)
    s = pd.DataFrame(rows)
    s["o"] = s.asset.map({k: i for i, k in enumerate(ORDER)}); s = s.sort_values("o").drop(columns="o")
    s.to_csv("results/q7_rq3_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("RQ3 with native quotes, across days\n" + s.round(3).to_string(index=False))

if os.path.exists("results/q7_reopen_native.csv"):
    r = pd.read_csv("results/q7_reopen_native.csv")
    r["d_g60"] = (r.g60 - r.g60_last).abs()
    out = []
    for kind, g in [("all", r)] + list(r.groupby("kind")):
        t = g[g.signal_native != 0]
        out.append(dict(kind=kind, reopens=len(g), days=g.day.nunique(),
                        g60_abs_diff_med_bps=g.d_g60.median(), g60_abs_diff_max_bps=g.d_g60.max(),
                        sign_agree=(np.sign(g.g60) == np.sign(g.g60_last)).mean(),
                        signals_native=int((g.signal_native != 0).sum()), signals_last=int((g.signal_last != 0).sum()),
                        signals_both=int(((g.signal_native != 0) & (g.signal_native == g.signal_last)).sum()),
                        gross_mid_mean=t.gross_mid_bps.mean() if len(t) else np.nan,
                        exec_trades=int(t.exec_net_bps.notna().sum()) if "exec_net_bps" in t else 0,
                        exec_gross_mean=t.exec_gross_bps.mean() if "exec_gross_bps" in t and len(t) else np.nan,
                        exec_net_mean=t.exec_net_bps.mean() if "exec_net_bps" in t and len(t) else np.nan,
                        bin_depth_entry_med_usd=t.bin_depth_entry_usd.median() if len(t) else np.nan,
                        bin_spread_entry_med_bps=g.bin_spread_entry_bps.median()))
    o = pd.DataFrame(out); o.to_csv("results/q7_reopen_summary.csv", index=False)
    print("\nReopens with native quotes\n" + o.round(2).to_string(index=False))
    cols = [c for c in ["day", "asset", "kind", "cme_quotes", "g0", "g60", "g60_last", "g960", "trade_side", "gross_mid_bps",
                        "exec_gross_bps", "fees_bps", "exec_net_bps", "bin_spread_entry_bps", "bin_depth_entry_usd"] if c in r]
    tr = r[r.signal_native != 0][cols]; tr.to_csv("results/q7_trades.csv", index=False)
    print("\nTrade signals\n" + tr.round(1).to_string(index=False))
