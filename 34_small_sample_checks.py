"""Small-sample checks (fourth review).

A) Weekend reopen-gap regressions pooled over the nine traditional markets, as in 30_rq1_inference.py: the number of
   unique weekend dates behind the pooled inference; the energy-minus-metal and gold-minus-silver slope contrasts with
   standard errors clustered by weekend date (CR1), p-values from the normal and from t(G-1), and wild-cluster
   restricted (WCR) bootstrap p-values (null imposed, 9,999 draws, Rademacher and Webb weights; Cameron, Gelbach and
   Miller 2008), which stay reliable with few clusters.
B) Native-quote dates: inventory of the full native-quote market-days (>= 12 h of CME-open time with native quotes, the
   rule of 26_native_days_table.py) by date: source, CME-open hours, markets; and a date-level summary of the lead-lag
   measures (median and minimum LLR, median HY peak, share of markets with D(B|C) > D(C|B)).

Outputs: results/q14_rq1_small_sample.txt, results/q14_native_dates.csv
"""
import numpy as np, pandas as pd
from scipy import stats

ROOTS = ["BTC", "ES", "GC", "SI", "PL", "PA", "HG", "CL", "BZ", "NG"]
ENERGY, METALS = ["CL", "BZ", "NG"], ["GC", "SI", "PL", "PA", "HG"]
B, SEED, FULL_HOURS = 9999, 2026, 12

# ---------------------------------------------------------------- A) pooled weekend regressions
ev = pd.concat([pd.read_csv(f"results/q1_events_{r}.csv") for r in ROOTS], ignore_index=True)
ev = ev[ev.kind == "weekend"].dropna(subset=["cme_gap", "bin_ret"]).copy()
ev["G"], ev["R"] = ev.cme_gap * 1e4, ev.bin_ret * 1e4
ev["wk"] = pd.to_datetime(ev.reopen_ts, utc=True).dt.strftime("%Y-%m-%d")
P = ev[ev.asset != "BTC"].reset_index(drop=True)
D = pd.get_dummies(P.asset, dtype=float)
X = pd.concat([D.add_prefix("a_"), D.mul(P.R, axis=0).add_prefix("b_")], axis=1)
cols, Xv, y = list(X.columns), X.values, P.G.values
g = pd.factorize(P.wk)[0]
G, (N, K) = g.max() + 1, Xv.shape
M = np.zeros((G, N)); M[g, np.arange(N)] = 1.0                         # cluster indicator
XtXi = np.linalg.inv(Xv.T @ Xv)
cr1 = G / (G - 1) * (N - 1) / (N - K)


def contrast(weights):
    r = np.zeros(K)
    for name, w in weights.items():
        r[cols.index(f"b_{name}")] = w
    return r


def t_stat(r, Y):
    """cluster-robust (CR1) t statistic of r'b = 0 for each column of Y"""
    a = XtXi @ r; h = Xv @ a
    Bh = XtXi @ (Xv.T @ Y); U = Y - Xv @ Bh
    est = h @ Y
    Z = M @ (h[:, None] * U)
    return est / np.sqrt(cr1 * (Z ** 2).sum(0)), est


def wcr(r, kind, rng):
    b = XtXi @ (Xv.T @ y)
    a = XtXi @ r
    br = b - a * (r @ b) / (r @ a)                                     # restricted estimate, r'b = 0
    ur = y - Xv @ br
    if kind == "rademacher":
        W = rng.choice([-1.0, 1.0], size=(G, B))
    else:                                                              # Webb six-point
        W = rng.choice(np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)]), size=(G, B))
    Ystar = (Xv @ br)[:, None] + ur[:, None] * W[g, :]
    tb, _ = t_stat(r, Ystar)
    return tb


rng = np.random.default_rng(SEED)
out = [f"pooled weekend regressions, nine traditional markets: {N} market-weekend observations on {G} weekend dates",
       f"  weekend dates with energy data: {P[P.asset.isin(ENERGY)].wk.nunique()}, "
       f"with metal data: {P[P.asset.isin(METALS)].wk.nunique()}, with both: "
       f"{len(set(P[P.asset.isin(ENERGY)].wk) & set(P[P.asset.isin(METALS)].wk))}"]
for label, w in [("energy - metals", {**{a: 1 / 3 for a in ENERGY}, **{a: -1 / 5 for a in METALS}}),
                 ("gold - silver", {"GC": 1, "SI": -1})]:
    r = contrast(w)
    t0, est = t_stat(r, y[:, None]); t0, est = float(t0[0]), float(est[0])
    se = abs(est / t0)
    p_norm, p_t = 2 * stats.norm.sf(abs(t0)), 2 * stats.t.sf(abs(t0), G - 1)
    p_rad = float(np.mean(np.abs(wcr(r, "rademacher", rng)) >= abs(t0)))
    p_webb = float(np.mean(np.abs(wcr(r, "webb", rng)) >= abs(t0)))
    tq = stats.t.ppf(0.975, G - 1)
    out.append(f"{label}: {est:.3f} (CR1 s.e. {se:.3f}, t(G-1) 95% CI [{est - tq * se:.3f}, {est + tq * se:.3f}]); "
               f"p normal {p_norm:.4f}, p t({G - 1}) {p_t:.4f}, WCR bootstrap p Rademacher {p_rad:.4f}, Webb {p_webb:.4f} "
               f"({B} draws)")
open("results/q14_rq1_small_sample.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))

# ---------------------------------------------------------------- B) native-quote dates
d = pd.read_csv("results/q7_rq3_days.csv")
d = d[(d.cme_ticks.astype(str).str.lower() == "true") & (d.open_hours >= FULL_HOURS)].copy()
d["lead_share"] = d.LLR / (1 + d.LLR)
rows = []
for day, x in d.groupby("day"):
    rows.append(dict(day=day, source=x.native_source.iloc[0], cme_open_hours=x.open_hours.median(), markets=len(x),
                     assets=" ".join(sorted(x.asset)), LLR_med=x.LLR.median(), LLR_min=x.LLR.min(),
                     LLR_max=x.LLR.max(), lead_share_min=x.lead_share.min(), HY_med=x.HY_peak_ms.median(),
                     HY_pos=(x.HY_peak_ms > 0).mean(), D_pos=(x.D_bin_after_cme > x.D_cme_after_bin).mean()))
T = pd.DataFrame(rows)
T.to_csv("results/q14_native_dates.csv", index=False)
pd.set_option("display.width", 250)
print(f"\nfull native-quote market-days: {len(d)} on {d.day.nunique()} dates")
print(T.round(2).to_string(index=False))
print(f"date medians of LLR: median {T.LLR_med.median():.0f}, range {T.LLR_med.min():.0f}-{T.LLR_med.max():.0f}")
