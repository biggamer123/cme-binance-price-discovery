"""Summaries behind Table 3 and Table 5a (run after 24_native_multiday.py and 29_timestamp_shift.py --merge).

Uses the same day rules as 26_native_days_table.py: full days only (>= 12 h of CME-open time with native quotes),
and no VECM shares on the energy market-days of 19 Aug 2026, when the two venues referenced different delivery months.

Writes
  results/q7_native_shares_by_asset.csv  IS bounds, CS, information leadership share (ILS) and half-life per market
  results/q13_day_level.csv              between-session variation: share of days with LLR > 1, HY > 0, IS_lo > 0.5,
                                         ILS > 0.5, D(B|C) > D(C|B), and quartiles of LLR and ILS
  results/q9_shift_summary.csv           Table 5a: share of market-days on which each measure favours CME, per shift
"""
import os
import numpy as np, pandas as pd

OUT = os.environ.get("OUT", "results")
FULL_HOURS = 12
ROLL = {("CL", "2026-08-19"), ("BZ", "2026-08-19"), ("NG", "2026-08-19")}
ORDER = ["BTC", "ES", "GC", "SI", "PL", "PA", "HG", "CL", "BZ", "NG"]


def ils(is_c, cs_c):
    """Putnins (2013) information leadership share of CME: x^2 / (1 + x^2), x = |IS_C CS_B / (IS_B CS_C)|."""
    x = np.abs(is_c * (1 - cs_c) / ((1 - is_c) * cs_c))
    return x ** 2 / (1 + x ** 2)


d = pd.read_csv("results/q7_rq3_days.csv")
d = d[d.cme_ticks.astype(str).str.lower() == "true"].copy()
roll = d.apply(lambda r: (r.asset, r.day) in ROLL, axis=1)
d.loc[roll, ["IS_lo", "IS_hi", "CS"]] = np.nan
d = d[d.open_hours >= FULL_HOURS].copy()
d["IS_mid"] = (d.IS_lo + d.IS_hi) / 2
d["ILS"] = ils(d.IS_mid, d.CS)
d["ILS_lo"] = ils(d.IS_lo, d.CS)
d["ILS_hi"] = ils(d.IS_hi, d.CS)

shares, level = [], []
for a in sorted(d.asset.unique(), key=ORDER.index):
    g = d[d.asset == a]
    v = g[g.IS_lo.notna()]
    shares.append(dict(asset=a, n=len(v), IS_lo=v.IS_lo.median(), IS_hi=v.IS_hi.median(), CS=v.CS.median(),
                       ILS=v.ILS.median(), ILS_q1=v.ILS.quantile(0.25), ILS_q3=v.ILS.quantile(0.75),
                       ILSlo=v.ILS_lo.median(), ILShi=v.ILS_hi.median(), hl=v.halflife_s.median(),
                       hl_min=v.halflife_s.min(), hl_max=v.halflife_s.max(), vh=v.vecm_hours.median()))
    level.append(dict(asset=a, days=len(g), is_days=len(v), LLR_gt1=(g.LLR > 1).mean(), LLR_min=g.LLR.min(),
                      LLR_q1=g.LLR.quantile(0.25), LLR_med=g.LLR.median(), LLR_q3=g.LLR.quantile(0.75),
                      LLR_max=g.LLR.max(), HY_pos=(g.HY_peak_ms > 0).mean(), HY_min=g.HY_peak_ms.min(),
                      HY_med=g.HY_peak_ms.median(), HY_max=g.HY_peak_ms.max(), ISlo_gt05=(v.IS_lo > 0.5).mean(),
                      ISlo_min=v.IS_lo.min(), ISlo_med=v.IS_lo.median(), ILS_gt05=(v.ILS > 0.5).mean(),
                      ILS_min=v.ILS.min(), ILS_q1=v.ILS.quantile(0.25), ILS_med=v.ILS.median(),
                      ILS_q3=v.ILS.quantile(0.75), D_pos=(g.D_bin_after_cme > g.D_cme_after_bin).mean()))
os.makedirs(OUT, exist_ok=True)
pd.DataFrame(shares).to_csv(f"{OUT}/q7_native_shares_by_asset.csv", index=False)
pd.DataFrame(level).to_csv(f"{OUT}/q13_day_level.csv", index=False)

trad = d[d.asset != "BTC"]
v = trad[trad.IS_lo.notna()]
print(f"traditional assets: {len(trad)} market-days on {trad.day.nunique()} dates; IS on {len(v)} market-days; "
      f"IS_lo > 0.5 on {(v.IS_lo > 0.5).sum()}, ILS > 0.5 on {(v.ILS > 0.5).sum()}; "
      f"LLR median {trad.LLR.median():.0f}, IQR {trad.LLR.quantile(.25):.0f}-{trad.LLR.quantile(.75):.0f}, "
      f"range {trad.LLR.min():.0f}-{trad.LLR.max():.0f}; HY peak 50-100 ms on {trad.HY_peak_ms.between(50, 100).mean():.0%}")

if os.path.exists("results/q9_shift.csv"):
    s = pd.read_csv("results/q9_shift.csv")
    s["Dnet"] = s.D_bin_after_cme - s.D_cme_after_bin
    t = s.groupby("shift_ms").agg(LLR_gt1=("LLR", lambda x: (x > 1).mean()), LLR_med=("LLR", "median"),
                                  gridpeak_pos=("grid_peak_ms", lambda x: (x > 0).mean()),
                                  HY_pos=("HY_peak_ms", lambda x: (x > 0).mean()), HY_med=("HY_peak_ms", "median"),
                                  HY_ge0=("HY_peak_ms", lambda x: (x >= 0).mean()),
                                  Dnet_pos=("Dnet", lambda x: (x > 0).mean()), Dnet_med=("Dnet", "median"),
                                  DBC_med=("D_bin_after_cme", "median"), DCB_med=("D_cme_after_bin", "median"))
    t.reset_index().to_csv(f"{OUT}/q9_shift_summary.csv", index=False)
    print(t.round(2).to_string())
