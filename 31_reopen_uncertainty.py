"""Uncertainty of the reopen-convergence numbers (third review, point 7).
A) 1-minute data, weekend reopens Jan-Sep 2026 (events of 11_q3_reopen_halflife.py, cached by 24_native_multiday.py):
   share of the basis-adjusted gap still open 1 and 5 min after the reopen (slope of g_k on g_0) and the half-life,
   with 95% bootstrap intervals (2,000 resamples of weekends).
B) Tick window (results/q3b_tick_events.csv, 12_q3b_reopen_ticks.py): share open after 60 s measured from the reopen
   and from 5 s and 10 s after it (drops the first seconds, when the CME book is still rebuilding).
Outputs: results/q12_reopen_boot.csv, results/q12_reopen_tick_start.csv"""
import numpy as np, pandas as pd, statsmodels.api as sm
rng = np.random.default_rng(7)
ROOTS = ["GC", "SI", "PL", "PA", "HG", "CL", "BZ", "NG", "ES"]
KS = list(range(0, 31))


def slope(y, x):
    X = np.column_stack([np.ones(len(x)), x]); return np.linalg.lstsq(X, y, rcond=None)[0][1]


def half_life(betas):
    for j in range(1, len(KS)):
        if betas[j] <= 0.5:
            b0, b1 = betas[j - 1], betas[j]
            return KS[j - 1] + (b0 - 0.5) / (b0 - b1) * (KS[j] - KS[j - 1]) if b0 > b1 else KS[j]
    return np.nan


rows = []
for r in ROOTS:
    e = pd.read_pickle(f"results/q7_cache/events_{r}.pkl")
    if "kind" not in e:
        continue
    e = e[e.kind == "weekend"]
    G = np.column_stack([(e[f"b{k}"] - e[f"c{k}"] - e.base_post).values for k in KS])
    G = G[np.isfinite(G).all(1)]; n = len(G)
    if n < 8:
        continue
    est = [slope(G[:, j], G[:, 0]) for j in range(len(KS))]
    boot = []
    for _ in range(2000):
        s = G[rng.integers(0, n, n)]
        b = [slope(s[:, j], s[:, 0]) for j in range(len(KS))]
        boot.append((b[1], b[5], half_life(b)))
    B = np.array(boot)
    q = lambda v: np.nanpercentile(v, [2.5, 97.5])
    rows.append(dict(asset=r, weekends=n, open_1m=est[1], open_1m_lo=q(B[:, 0])[0], open_1m_hi=q(B[:, 0])[1],
                     open_5m=est[5], open_5m_lo=q(B[:, 1])[0], open_5m_hi=q(B[:, 1])[1],
                     half_life_min=half_life(est), hl_lo=q(B[:, 2])[0], hl_hi=q(B[:, 2])[1],
                     hl_undefined_share=np.isnan(B[:, 2]).mean()))
T = pd.DataFrame(rows); T.to_csv("results/q12_reopen_boot.csv", index=False)
pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))

d = pd.read_csv("results/q3b_tick_events.csv")
g = lambda s: d[f"b{s:g}"] - d[f"c{s:g}"] - d.base_post
out = []
for (grp, kind), x in list(d.groupby(["group", "kind"])) + [(("all", k), x) for k, x in d.groupby("kind")]:
    for s0 in (0, 5, 10):
        y, x0 = g(60).loc[x.index], g(s0).loc[x.index]; ok = np.isfinite(y) & np.isfinite(x0)
        if ok.sum() < 4:
            continue
        m = sm.OLS(y[ok], sm.add_constant(x0[ok])).fit(cov_type="HC1")
        out.append(dict(group=grp, kind=kind, n=int(ok.sum()), start_s=s0, open_60s=m.params.iloc[1], se=m.bse.iloc[1]))
U = pd.DataFrame(out); U.to_csv("results/q12_reopen_tick_start.csv", index=False); print(U.round(3).to_string(index=False))
