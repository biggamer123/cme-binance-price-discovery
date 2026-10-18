"""Checks asked for by the first review (all from existing data/results; prints numbers for the paper).
A  RQ1 weekend regressions: HC1 vs HC3 s.e., reverse-regression upper bound (beta/R2), leave-one-out range,
   binomial test of the sign hit rate, slope-difference z-tests; daily-break HC3 s.e.
B  Funding during CME weekend closures: does Binance's funding track the perp's move away from the Friday close
   (what a frozen index implies), how large is it, and how does it relate to the RQ1 shortfall?
C  Event counts behind the tick-level reopen results (Fig. 3a, Table 5 half-lives).
D  Information leadership share (Putnins 2013) from the session VECMs (tbbo, mbp-1) and the native full days.
E  Bounded lead-lag share LLR/(1+LLR).   F  CME gold spread.   G  native Binance quote updates per day.
Writes results/review_*.csv."""
import glob, os
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOTS = {"BTC": "BTCUSDT", "ES": "SPYUSDT", "GC": "XAUUSDT", "SI": "XAGUSDT", "PL": "XPTUSDT", "PA": "XPDUSDT",
         "HG": "COPPERUSDT", "CL": "CLUSDT", "BZ": "BZUSDT", "NG": "NATGASUSDT"}
pd.set_option("display.width", 250)


def ols(y, x, cov):
    m = sm.OLS(y, sm.add_constant(x)).fit(cov_type=cov)
    return m.params[1], m.bse[1], m.rsquared


# ---------------------------------------------------------------- A
rows, ev_all = [], {}
for r in ROOTS:
    f = f"results/q1_events_{r}.csv"
    if not os.path.exists(f):
        continue
    e = pd.read_csv(f, parse_dates=["close_ts", "reopen_ts"]); ev_all[r] = e
    for kind in ["weekend", "daily_break"]:
        w = e[e.kind == kind].dropna(subset=["cme_gap", "bin_ret"])
        if len(w) < 5:
            continue
        y, x = w.cme_gap.values * 1e4, w.bin_ret.values * 1e4
        b1, s1, r2 = ols(y, x, "HC1"); b3, s3, _ = ols(y, x, "HC3")
        loo = [ols(np.delete(y, i), np.delete(x, i), "nonrobust") for i in range(len(y))]
        hits = int((np.sign(y) == np.sign(x)).sum())
        rows.append(dict(asset=r, kind=kind, n=len(w), beta=b1, se_hc1=s1, se_hc3=s3, r2=r2,
                         beta_upper=b1 / r2 if r2 > 0 else np.nan,
                         loo_beta_min=min(l[0] for l in loo), loo_beta_max=max(l[0] for l in loo),
                         loo_r2_min=min(l[2] for l in loo), hit=hits / len(w),
                         hit_p=stats.binomtest(hits, len(w), 0.5).pvalue))
A = pd.DataFrame(rows); A.to_csv("results/review_rq1.csv", index=False)
print("A. RQ1 robustness\n" + A.round(3).to_string(index=False))
wk = A[A.kind == "weekend"].set_index("asset")
for a, b in [("ES", "SI"), ("GC", "SI"), ("CL", "GC"), ("GC", "PL"), ("SI", "PA"), ("PA", "HG")]:
    if a in wk.index and b in wk.index:
        z = (wk.beta[a] - wk.beta[b]) / np.hypot(wk.se_hc3[a], wk.se_hc3[b])
        print(f"   slope {a} - {b}: z = {z:.2f}, p = {2 * stats.norm.sf(abs(z)):.3f}")


# ---------------------------------------------------------------- B
def klines(sym):
    parts = []
    for f in sorted(glob.glob(f"data/binance/{sym}/klines1m/*.csv")):
        d = pd.read_csv(f, header=None)
        if not str(d.iloc[0, 0]).isdigit():
            d = d.iloc[1:]
        d = d.iloc[:, [0, 4]].astype(float); d.columns = ["t", "close"]; parts.append(d)
    b = pd.concat(parts).drop_duplicates("t")
    unit = "us" if b.t.iloc[0] > 1e14 else "ms"
    s = pd.Series(b.close.values, index=pd.to_datetime(b.t.astype("int64"), unit=unit, utc=True)).sort_index()
    return s


def funding(sym):
    fs = sorted(glob.glob(f"data/binance/{sym}/funding/*.csv"))
    if not fs:
        return None
    d = pd.concat(pd.read_csv(f) for f in fs)
    return pd.Series(d.last_funding_rate.astype(float).values * 1e4,
                     index=pd.to_datetime(d.calc_time, unit="ms", utc=True).dt.floor("min")).sort_index()


pay_rows, wk_rows = [], []
for r, sym in ROOTS.items():
    if r in ("BTC", "ES") or r not in ev_all:
        continue
    fu = funding(sym)
    if fu is None:
        continue
    px = klines(sym); lp = np.log(px)
    e = ev_all[r]; e = e[e.kind == "weekend"].dropna(subset=["cme_gap", "bin_ret"])
    for _, ev in e.iterrows():
        c0 = lp.asof(ev.close_ts)                                   # Binance log price at the CME close
        f_in = fu[(fu.index > ev.close_ts) & (fu.index <= ev.reopen_ts)]
        for ft, rate in f_in.items():
            win = lp[(lp.index > ft - pd.Timedelta(hours=4)) & (lp.index <= ft)]
            dev_avg = (win.mean() - c0) * 1e4 if len(win) else np.nan      # avg distance from Friday close
            dev_end = (lp.asof(ft) - c0) * 1e4
            pay_rows.append(dict(asset=r, close=ev.close_ts, t=ft, rate_bps=rate, dev_avg_bps=dev_avg, dev_end_bps=dev_end))
        wk_rows.append(dict(asset=r, close=ev.close_ts, G=ev.cme_gap * 1e4, R=ev.bin_ret * 1e4,
                            F=f_in.sum(), F_abs=f_in.abs().sum(), n_pay=len(f_in)))
P = pd.DataFrame(pay_rows); W = pd.DataFrame(wk_rows)
P.to_csv("results/review_funding_payments.csv", index=False); W.to_csv("results/review_funding_weekends.csv", index=False)
# what a frozen index + Binance's formula would give: premium beyond a +-5 bps dead zone, capped at +-50 bps
P["pred"] = np.clip(np.sign(P.dev_avg_bps) * np.maximum(P.dev_avg_bps.abs() - 5, 0), -50, 50)
print("\nB. Funding payments inside CME weekend closures (bps per 4 h payment; dev = Binance price vs its level at the CME close)")
for a, g in P.groupby("asset"):
    nz = g[g.rate_bps.abs() > 0.01]
    big = g[g.dev_avg_bps.abs() > 20]
    print(f"  {a}: payments {len(g)}, nonzero {len(nz) / len(g):.0%}; corr(rate, dev_avg) {g.rate_bps.corr(g.dev_avg_bps):.2f}; "
          f"corr(rate, frozen-index prediction) {g.rate_bps.corr(g.pred):.2f}; sign agree (nonzero) "
          f"{(np.sign(nz.rate_bps) == np.sign(nz.dev_avg_bps)).mean():.0%}; when |dev|>20 bps: n={len(big)}, "
          f"median rate {big.rate_bps.abs().median():.1f} vs median |dev| {big.dev_avg_bps.abs().median():.1f}, "
          f"zero in {(big.rate_bps.abs() < 0.01).mean():.0%}")
allc = P.rate_bps.corr(P.pred); print(f"  pooled corr(actual, frozen-index prediction) = {allc:.2f}; "
                                       f"slope actual on prediction = {np.polyfit(P.pred.fillna(0), P.rate_bps, 1)[0]:.2f}")
W["S"] = W.G - W.R
W["sameside"] = np.sign(W.F) == np.sign(W.R)
print("\n  Per weekend: |F| summed over the closure vs |Binance weekend move|")
for a, g in W.groupby("asset"):
    act = g[g.F_abs > 5]
    print(f"  {a}: weekends {len(g)}, funding > 5 bps on {len(act)}; corr(F, R) {g.F.corr(g.R):.2f}; median |F|/|R| where "
          f"|R|>30 bps: {(g.F.abs() / g.R.abs())[g.R.abs() > 30].median():.2f}; F same sign as R {g.sameside[g.F_abs > 0.5].mean():.0%}")
# pooled, asset fixed effects: does the shortfall S = G - R load on the funding paid, given R?
X = pd.get_dummies(W.asset, drop_first=True, dtype=float)
for name, cols in [("S on R", ["R"]), ("S on R and F", ["R", "F"])]:
    m = sm.OLS(W.S, sm.add_constant(pd.concat([W[cols], X], axis=1))).fit(cov_type="HC3")
    print(f"  pooled {name}: " + ", ".join(f"{c} {m.params[c]:.3f} (se {m.bse[c]:.3f})" for c in cols) + f", n={len(W)}")
for lab, g in [("funding active (|F|>5 bps)", W[W.F_abs > 5]), ("funding inactive", W[W.F_abs <= 5])]:
    Xg = pd.get_dummies(g.asset, drop_first=True, dtype=float)
    m = sm.OLS(g.G, sm.add_constant(pd.concat([g[["R"]], Xg], axis=1))).fit(cov_type="HC3")
    print(f"  pooled RQ1 slope, {lab}: {m.params['R']:.2f} (se {m.bse['R']:.2f}), n={len(g)}, mean |R| {g.R.abs().mean():.0f} bps")

# ---------------------------------------------------------------- C
te = pd.read_csv("results/q3b_tick_events.csv")
kc = [c for c in te.columns if c in ("kind", "type")][0]; ac = [c for c in te.columns if c in ("asset", "root")][0]
print("\nC. Tick-window reopens per market:\n" + te.groupby([ac, kc]).size().unstack(fill_value=0).to_string())


# ---------------------------------------------------------------- D, E
def ils(cs, lo, hi):
    """Putnins (2013): ILS_1 = IS1/IS2*CS2/CS1 squared over (that + 1), IS = mid of the Hasbrouck bounds"""
    is1 = (lo + hi) / 2; is2 = 1 - is1; cs2 = 1 - cs
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.abs(is1 / is2 * cs2 / cs)
    return x ** 2 / (x ** 2 + 1)


rows = []
summ = pd.read_csv("results/q2_summary_mbp1.csv").set_index("asset") if os.path.exists("results/q2_summary_mbp1.csv") else None
for r in ROOTS:
    row = dict(asset=r)
    for tag, f in [("tbbo", f"results/q2_sessions_{r}.csv"), ("mbp1", f"results/q2_sessions_{r}_mbp1.csv")]:
        if os.path.exists(f):
            s = pd.read_csv(f)
            row[f"ILS_{tag}"] = np.nanmedian(ils(s.cs_cme.values, s.is_cme_lo.values, s.is_cme_hi.values))
            row[f"sessions_{tag}"] = len(s)
    if summ is not None and r in summ.index:
        L = summ.loc[r, "LLR"]; row["LLR_mbp1"] = L; row["LLR_share"] = L / (1 + L)
    rows.append(row)
d = pd.read_csv("results/q7_rq3_days.csv"); d = d[(d.cme_ticks.astype(str).str.lower() == "true") & (d.open_hours >= 12)]
d = d[~((d.day == "2026-08-19") & d.asset.isin(["CL", "BZ", "NG"]))]
nat = d.assign(ILS=ils(d.CS.values, d.IS_lo.values, d.IS_hi.values)).groupby("asset").ILS.median()
D = pd.DataFrame(rows).set_index("asset"); D["ILS_native_days"] = nat
D.to_csv("results/review_ils.csv")
print("\nD/E. Information leadership share of CME (median) and bounded lead-lag share\n" + D.round(3).to_string())

# ---------------------------------------------------------------- F, G
g = pd.read_parquet("data/cme/days/GC_mbp1_2026-06-01.parquet", columns=["ts_event", "bid_px_00", "ask_px_00"])
g = g[(g.bid_px_00 > 0) & (g.ask_px_00 > g.bid_px_00)]
print(f"\nF. CME gold (GC) median quoted spread on 1 Jun 2026: {((g.ask_px_00 - g.bid_px_00) / ((g.ask_px_00 + g.bid_px_00) / 2) * 1e4).median():.3f} bps "
      f"(tick 0.10 -> {0.1 / ((g.ask_px_00 + g.bid_px_00).median() / 2) * 1e4:.3f} bps)")
t = pd.read_csv("results/q7_rq3_days.csv"); t = t[(t.native_source == "tardis") & (t.open_hours >= 22)]
print("G. Native Binance quote updates per day (thousands, median over full Tardis days):",
      (t.groupby("asset").native_updates.median() / 1e3).round(0).to_dict())
