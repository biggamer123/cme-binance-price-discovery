"""Inference for the weekend reopen-gap regressions (third review, point 1).
Per market (G = a + b R): b, HC3 s.e., 95% interval (t with n-2 df), p for b = 0 and for b = 1, Holm-adjusted
p-values over the ten markets, leave-one-weekend-out range of b and R2, HAC (Newey-West, 1 lag) s.e. and a lag-1
residual autocorrelation test across consecutive weekends.  Pooled over the nine traditional markets: market-specific
intercepts and slopes with s.e. clustered by weekend date (the same news hits every market on the same weekend);
Wald tests that all slopes are equal and that the energy and metal slopes are equal on average.
Weekends before / after Binance's 8 May 2026 index change are counted per market.
Outputs: results/q10_rq1_inference.csv, results/q10_rq1_pooled.txt"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox
ROOTS = ["BTC", "ES", "GC", "SI", "PL", "PA", "HG", "CL", "BZ", "NG"]
SW = pd.Timestamp("2026-05-09", tz="UTC")

ev = pd.concat([pd.read_csv(f"results/q1_events_{r}.csv") for r in ROOTS], ignore_index=True)
ev = ev[ev.kind == "weekend"].dropna(subset=["cme_gap", "bin_ret"]).copy()
ev["G"], ev["R"] = ev.cme_gap * 1e4, ev.bin_ret * 1e4
ev["reopen"] = pd.to_datetime(ev.reopen_ts, utc=True); ev["wk"] = ev.reopen.dt.strftime("%Y-%m-%d")
rows = []
for r in ROOTS:
    w = ev[ev.asset == r].sort_values("reopen"); y, x = w.G.values, w.R.values; n = len(w)
    X = sm.add_constant(x); m = sm.OLS(y, X)
    f3 = m.fit(cov_type="HC3"); fh = m.fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    b, s = f3.params[1], f3.bse[1]; tq = stats.t.ppf(0.975, n - 2)
    loo = [sm.OLS(np.delete(y, i), sm.add_constant(np.delete(x, i))).fit() for i in range(n)]
    lb = acorr_ljungbox(f3.resid, lags=[1], return_df=True)
    rows.append(dict(asset=r, n=n, n_before=int((w.reopen < SW).sum()), n_after=int((w.reopen >= SW).sum()),
                     beta=b, se_hc3=s, ci_lo=b - tq * s, ci_hi=b + tq * s, r2=f3.rsquared,
                     p_beta0=2 * stats.t.sf(abs(b / s), n - 2), p_beta1=2 * stats.t.sf(abs((b - 1) / s), n - 2),
                     se_hac=fh.bse[1], loo_beta_min=min(l.params[1] for l in loo), loo_beta_max=max(l.params[1] for l in loo),
                     loo_r2_min=min(l.rsquared for l in loo), resid_ac1=pd.Series(f3.resid).autocorr(1),
                     ljungbox_p=float(lb.lb_pvalue.iloc[0])))
T = pd.DataFrame(rows)
T["p_beta0_holm"] = multipletests(T.p_beta0, method="holm")[1]
T["p_beta1_holm"] = multipletests(T.p_beta1, method="holm")[1]
T.to_csv("results/q10_rq1_inference.csv", index=False)
pd.set_option("display.width", 250)
print(T.round(3).to_string(index=False))

# pooled, traditional markets: market intercepts and slopes, clustered by weekend date
P = ev[ev.asset != "BTC"].reset_index(drop=True)
A = sorted(P.asset.unique())
X = pd.concat([pd.get_dummies(P.asset, dtype=float).add_prefix("a_"),
               pd.get_dummies(P.asset, dtype=float).mul(P.R, axis=0).add_prefix("b_")], axis=1)
g = pd.factorize(P.wk)[0]
fit = sm.OLS(P.G, X).fit(cov_type="cluster", cov_kwds={"groups": g})
out = [f"weekends (clusters): {len(np.unique(g))}, observations: {len(P)}"]
bn = [f"b_{a}" for a in A]
R = np.zeros((len(A) - 1, X.shape[1]))
for i in range(len(A) - 1):
    R[i, list(X.columns).index(bn[i])] = 1; R[i, list(X.columns).index(bn[i + 1])] = -1
w1 = fit.wald_test(R, scalar=True); out.append(f"all nine slopes equal: chi2={float(w1.statistic):.2f}, df={len(A)-1}, p={float(w1.pvalue):.3f}")
energy, metals = ["CL", "BZ", "NG"], ["GC", "SI", "PL", "PA", "HG"]
r2 = np.zeros((1, X.shape[1]))
for a in energy: r2[0, list(X.columns).index(f"b_{a}")] = 1 / len(energy)
for a in metals: r2[0, list(X.columns).index(f"b_{a}")] = -1 / len(metals)
w2 = fit.wald_test(r2, scalar=True)
out.append(f"mean energy slope - mean metal slope = {float(r2 @ fit.params.values):.3f}, p={float(w2.pvalue):.3f}")
r3 = np.zeros((1, X.shape[1])); r3[0, list(X.columns).index("b_GC")] = 1; r3[0, list(X.columns).index("b_SI")] = -1
w3 = fit.wald_test(r3, scalar=True); out.append(f"gold - silver = {float(r3 @ fit.params.values):.3f}, p={float(w3.pvalue):.3f}")
for a in A:
    out.append(f"  {a}: b={fit.params[f'b_{a}']:.3f} clustered s.e.={fit.bse[f'b_{a}']:.3f}")
open("results/q10_rq1_pooled.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
