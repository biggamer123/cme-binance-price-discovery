"""Q1b: after CME reopens, how fast does Binance absorb the part of the CME gap it had NOT priced?
surprise = cme_gap - bin_ret (what CME priced at the reopen beyond Binance's closure move)
catch-up slope at h = OLS slope of Binance return over [reopen, reopen+h] on surprise. 1 = full catch-up."""
import glob, numpy as np, pandas as pd, statsmodels.api as sm
rows = []
for f in sorted(glob.glob("results/q1_events_*.csv")):
    e = pd.read_csv(f)
    for kind in ["weekend", "daily_break"]:
        d = e[e.kind == kind].copy()
        if len(d) < 8: continue
        d["surprise"] = d.cme_gap - d.bin_ret
        r = dict(asset=d.asset.iloc[0], kind=kind, n=len(d), mean_abs_surprise_bps=d.surprise.abs().mean() * 1e4)
        for h in ["bin_post5", "bin_post30"]:
            m = sm.OLS(d[h], sm.add_constant(d.surprise)).fit(cov_type="HC1")
            r[f"catchup_{h[8:]}"] = m.params["surprise"]; r[f"se_{h[8:]}"] = m.bse["surprise"]
        rows.append(r)
out = pd.DataFrame(rows); out.to_csv("results/q1b_catchup.csv", index=False)
print(out.round(3).to_string(index=False))
