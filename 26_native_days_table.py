"""Numbers for the paper from the multi-day native-quote run (after 24_native_multiday.py).
RQ3 across days uses FULL days only (>= 12 h of CME-open time with native Binance quotes); partial days are reported
separately.  The VECM information share is not used on days when the two venues referenced different delivery months
near expiry (ROLL: the energy basis drifted 50-90 bps within 19 Aug 2026, VECM half-life 6-15 min instead of 10-30 s);
LLR, HY and D are kept on those days.  Writes results/tab_native_days.tex (columns added to Table 4) and prints the
numbers for the text."""
import os
import numpy as np, pandas as pd

NAME = {"BTC": "Bitcoin", "ES": "S\\&P 500", "GC": "Gold", "SI": "Silver", "PL": "Platinum", "PA": "Palladium",
        "HG": "Copper", "CL": "WTI crude", "BZ": "Brent crude", "NG": "Natural gas"}
ORDER = list(NAME)
FULL_HOURS = 12
ROLL = {("CL", "2026-08-19"), ("BZ", "2026-08-19"), ("NG", "2026-08-19")}


def f(x, d=0):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


d = pd.read_csv("results/q7_rq3_days.csv")
d = d[d.cme_ticks.astype(str).str.lower() == "true"].copy()
roll = d.apply(lambda r: (r.asset, r.day) in ROLL, axis=1)
d.loc[roll, ["IS_lo", "IS_hi", "CS"]] = np.nan
full, part = d[d.open_hours >= FULL_HOURS], d[d.open_hours < FULL_HOURS]

rows, tex = [], []
for a in sorted(d.asset.unique(), key=ORDER.index):
    g, p = full[full.asset == a], part[part.asset == a]
    rows.append(dict(asset=a, full_days=len(g), first=g.day.min(), last=g.day.max(),
                     LLR_med=g.LLR.median(), LLR_min=g.LLR.min(), LLR_max=g.LLR.max(),
                     IS_days=int(g.IS_lo.notna().sum()), IS_lo_med=g.IS_lo.median(), IS_hi_med=g.IS_hi.median(),
                     IS_lo_min=g.IS_lo.min(), HY_med=g.HY_peak_ms.median(),
                     HY_50_100=((g.HY_peak_ms >= 50) & (g.HY_peak_ms <= 100)).mean(),
                     HY_pos=(g.HY_peak_ms > 0).mean(), D_BC=g.D_bin_after_cme.median(), D_CB=g.D_cme_after_bin.median(),
                     spread=g.native_spread_bps.median(), age=g.native_age_med_s.median(),
                     part_days=len(p), part_LLR_min=p.LLR.min(), part_LLR_gt1=(p.LLR > 1).mean() if len(p) else np.nan))
    tex.append(f"{NAME[a]} & {f(g.LLR.median())} [{f(g.LLR.min())}] & {f(g.HY_peak_ms.median())}")
s = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(f"RQ3 on full days (>= {FULL_HOURS} h of CME-open time with native quotes)")
print(s.drop(columns=["part_days", "part_LLR_min", "part_LLR_gt1"]).round(2).to_string(index=False))
print(f"\nAll full market-days: {len(full)} on {full.day.nunique()} days ({full.day.min()} .. {full.day.max()}); "
      f"LLR > 1 on {(full.LLR > 1).mean():.0%}; LLR median {full.LLR.median():.0f}, range {full.LLR.min():.0f}-{full.LLR.max():.0f}; "
      f"HY peak > 0 on {(full.HY_peak_ms > 0).mean():.0%}; HY peak 50-100 ms on {full.HY_peak_ms.between(50, 100).mean():.0%}")
print(f"IS^C lower bound: median over market medians {s.IS_lo_med.median():.2f}, range of market medians "
      f"{s.IS_lo_med.min():.2f}-{s.IS_lo_med.max():.2f}; upper {s.IS_hi_med.min():.2f}-{s.IS_hi_med.max():.2f}")
print(f"D(100 ms) medians: Binance after CME {s.D_BC.min():.2f}-{s.D_BC.max():.2f}; CME after Binance {s.D_CB.min():.3f}-{s.D_CB.max():.3f}")
print(f"Partial days (< {FULL_HOURS} h): {len(part)} market-days on {part.day.nunique()} days, LLR > 1 on "
      f"{(part.LLR > 1).mean():.0%}, min LLR {part.LLR.min():.1f}")
print(f"Native sources on full days: {full.groupby('native_source').day.nunique().to_dict()} days")
s.to_csv("results/q7_rq3_fulldays.csv", index=False)
os.makedirs("results", exist_ok=True)
open("results/tab_native_days.tex", "w").write("\n".join(tex) + "\n")
print("\nColumns for Table 4 (LLR median [min], HY peak median), full days:\n" + "\n".join(tex))

if os.path.exists("results/q7_reopen_native.csv"):
    r = pd.read_csv("results/q7_reopen_native.csv")
    r["d60"] = (r.g60 - r.g60_last).abs()
    print(f"\nReopens with native quotes: {len(r)} on {r.day.nunique()} days ({r.day.min()} .. {r.day.max()}; "
          f"weekend {int((r.kind == 'weekend').sum())}, daily break {int((r.kind == 'daily_break').sum())}); "
          f"CME mbp-1 at {int((r.cme_quotes == 'mbp-1').sum())}")
    print(f"  |native - last-trade gap| at 1 min: median {r.d60.median():.1f} bps, 90th pct {r.d60.quantile(.9):.1f}, "
          f"max {r.d60.max():.1f} ({r.loc[r.d60.idxmax(), 'asset']} {r.loc[r.d60.idxmax(), 'day']}); "
          f"sign agreement {(np.sign(r.g60) == np.sign(r.g60_last)).mean():.0%}")
    sn, sl = r.signal_native != 0, r.signal_last != 0
    print(f"  signals (|gap| > 20 bps at 1 min): last trade {int(sl.sum())}, native {int(sn.sum())}, both (same side) "
          f"{int((sn & (r.signal_native == r.signal_last)).sum())}; last-trade only: "
          f"{r[sl & ~sn][['day', 'asset', 'g60_last', 'g60']].round(1).values.tolist()}")
    t = r[sn]
    print(f"  native signals: gross at mid {t.gross_mid_bps.mean():.1f} bps (median {t.gross_mid_bps.median():.1f}), "
          f"{int((t.gross_mid_bps > 0).sum())} of {len(t)} positive")
    e = t[t.exec_net_bps.notna()]
    if len(e):
        print(f"  executed at the real bid/ask on BOTH venues (CME mbp-1): {len(e)} trades, gross at mid {e.gross_mid_bps.mean():.1f}, "
              f"after crossing spreads {e.exec_gross_bps.mean():.1f}, fees {e.fees_bps.mean():.1f}, net {e.exec_net_bps.mean():.1f} bps; "
              f"{int((e.exec_net_bps > 0).sum())} of {len(e)} profitable; Binance depth at entry "
              f"{e.bin_depth_entry_usd.min():,.0f}-{e.bin_depth_entry_usd.max():,.0f} USD")
    cols = ["day", "asset", "kind", "cme_quotes", "g60", "g60_last", "g960", "gross_mid_bps", "exec_gross_bps", "exec_net_bps",
            "bin_spread_entry_bps", "bin_depth_entry_usd"]
    print(t[cols].round(1).to_string(index=False))
