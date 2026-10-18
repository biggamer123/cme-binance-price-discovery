"""Q3: how fast does the Binance-CME gap close after CME reopens, and could you trade it?

Sample: every CME reopen (daily break + weekend) from each Binance perp's listing date to end-Sep 2026.
Resolution: 1 minute (Binance 1m klines; CME 1s bars as-of).

Gap at minute k after the reopen t1:
    g_k = log b(t1+k) - log c(t1+k) - basis
    b(t) = last Binance trade before t (close of the 1m bar ending at t)
    c(t) = last CME 1s-bar close before t; c(t1) = CME opening print (open of first bar, same as q1)
    basis = normal Binance-CME log spread
        'post' = median over [t1+60, t1+120] min  -> convergence target (uses the future; descriptive only)
        'pre'  = median over the last 30 min before the CME close  -> known in real time (used for trading)
For k < 0 (CME still closed) c is fixed at the opening print, so g_k shows Binance drifting toward
the price CME will open at (pre-open leakage).

Fraction of the reopen gap still open at k:  beta_k = OLS slope of g_k on g_0 across events.
Half-life = first k where beta_k <= 0.5 (linear interpolation).
Who closes it:  Binance share = slope of -(log b_k - log b_0) on g_0 ; CME share = slope of (log c_k - log c_0) on g_0.

Trade test (no look-ahead): at t1+L, if |g_L(pre basis)| > theta, go -sign(g_L) in the Binance perp,
exit at t1+L+H. One-legged P&L = -sign * Binance log return. Hedged P&L = -sign * (g_{L+H} - g_L).
Run: python 11_q3_reopen_halflife.py [DATA_DIR] [OUT_DIR]
"""
import glob, os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(OUT, exist_ok=True)

PAIRS = {"CL": "CLUSDT", "BZ": "BZUSDT", "NG": "NATGASUSDT",
         "GC": "XAUUSDT", "SI": "XAGUSDT", "PL": "XPTUSDT", "PA": "XPDUSDT", "HG": "COPPERUSDT"}
GROUP = {"CL": "energy", "BZ": "energy", "NG": "energy",
         "GC": "metals", "SI": "metals", "PL": "metals", "PA": "metals", "HG": "metals"}
DEGRADED = {"2026-01-31", "2026-03-15", "2026-03-16", "2026-03-21", "2026-04-10", "2026-05-24", "2026-08-29"}
K = np.r_[-60, -30, -15, -10, -5, -2, -1, 0, np.arange(1, 31), 45, 60]          # minutes from reopen
MIN = np.int64(60_000_000_000); SEC = np.int64(1_000_000_000)


def load_cme(root):
    df = pd.read_parquet(f"{DATA}/cme/{root}_ohlcv1s.parquet", columns=["open", "close", "instrument_id"])
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    return df.index.asi8, df["open"].values, df["close"].values, df["instrument_id"].values


def load_binance(sym):
    parts = []
    for f in sorted(glob.glob(f"{DATA}/binance/{sym}/klines1m/*.csv")):
        d = pd.read_csv(f, header=None)
        if not str(d.iloc[0, 0]).isdigit(): d = d.iloc[1:]
        d = d.iloc[:, [0, 4]].astype(float); d.columns = ["t", "close"]
        parts.append(d)
    b = pd.concat(parts).drop_duplicates("t").sort_values("t")
    t = b["t"].values.astype("int64")
    t = t * 1000 if t[0] < 1e14 else t                                        # ms -> us
    t_close = t * 1000 + MIN                                                  # ns, time the bar ends
    return t_close, b["close"].values


def asof(t, v, q):
    """value of the last observation with timestamp <= q (vectorised)."""
    i = np.searchsorted(t, q, side="right") - 1
    out = np.where(i >= 0, v[np.clip(i, 0, None)], np.nan)
    return out, i


def events(root, sym):
    ct, co, cc, ci = load_cme(root)
    bt, bc = load_binance(sym)
    gap = np.diff(ct)
    ts = pd.DatetimeIndex(ct, tz="UTC").tz_convert("America/Chicago")
    idx = np.where((gap > 45 * MIN) & (ts.hour[1:] == 17) & (ts.minute[1:] < 15))[0] + 1
    rows, paths = [], []
    for i in idx:
        t0, t1 = ct[i - 1], ct[i]
        d0 = pd.Timestamp(t0, tz="UTC"); d1 = pd.Timestamp(t1, tz="UTC")
        if {d0.strftime("%Y-%m-%d"), d1.strftime("%Y-%m-%d")} & DEGRADED: continue
        if t1 < bt[0] + 3 * MIN * 60 or t1 + 125 * MIN > bt[-1]: continue          # Binance not listed yet / end of data
        hrs = (t1 - t0) / 3.6e12
        dow = d1.tz_convert("America/Chicago").dayofweek
        kind = "daily_break" if hrs < 3 else ("weekend" if dow == 6 and 40 < hrs < 60 else "holiday")
        if kind == "holiday": continue
        # skip contract rolls anywhere from close to reopen+2h
        j_end = np.searchsorted(ct, t1 + 120 * MIN, side="right") - 1
        if ci[i - 1] != ci[i] or ci[j_end] != ci[i]: continue
        # baselines
        pre_grid = t0 + SEC - np.arange(0, 31) * MIN
        post_grid = t1 + np.arange(60, 121) * MIN
        b_pre, _ = asof(bt, bc, pre_grid); c_pre, _ = asof(ct, cc, pre_grid - SEC)
        b_post, _ = asof(bt, bc, post_grid); c_post, _ = asof(ct, cc, post_grid - SEC)
        base_pre = np.nanmedian(np.log(b_pre) - np.log(c_pre))
        base_post = np.nanmedian(np.log(b_post) - np.log(c_post))
        # paths
        q = t1 + K.astype(np.int64) * MIN
        b, _ = asof(bt, bc, q)
        c, _ = asof(ct, cc, q - SEC)
        c = np.where(K <= 0, co[i], c)                                          # CME opening print at/before reopen
        # stale Binance check: bar ending at t1 must exist (perp traded in the minute before reopen)
        if not np.isfinite(b).all() or not np.isfinite([base_pre, base_post]).all(): continue
        lb, lc = np.log(b), np.log(c)
        p = dict(asset=root, group=GROUP[root], kind=kind, reopen=d1, close=d0,
                 base_pre=base_pre, base_post=base_post)
        for k, x, y in zip(K, lb, lc):
            p[f"b{k}"] = x; p[f"c{k}"] = y
        paths.append(p)
    return pd.DataFrame(paths)


def gap(df, k, base="post"):
    return df[f"b{k}"] - df[f"c{k}"] - df[f"base_{base}"]


def slope(y, x):
    m = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC1")
    return m.params.iloc[1], m.bse.iloc[1]


def half_life(ks, betas):
    """first k (min) where beta <= 0.5 among k >= 0, linear interpolation; nan if never."""
    ks, betas = np.asarray(ks, float), np.asarray(betas, float)
    for j in range(1, len(ks)):
        if betas[j] <= 0.5:
            b0, b1 = betas[j - 1], betas[j]
            return ks[j - 1] + (b0 - 0.5) / (b0 - b1) * (ks[j] - ks[j - 1]) if b0 > b1 else ks[j]
    return np.nan


def convergence(ev, base="post"):
    rows, curves = [], []
    for (asset, kind), d in list(ev.groupby(["asset", "kind"])) + \
                            [(("METALS", kind), d) for kind, d in ev[ev.group == "metals"].groupby("kind")] + \
                            [(("ENERGY", kind), d) for kind, d in ev[ev.group == "energy"].groupby("kind")]:
        if len(d) < 8: continue
        g0 = gap(d, 0, base)
        r = dict(asset=asset, kind=kind, n=len(d), mean_abs_gap0_bps=g0.abs().mean() * 1e4,
                 median_abs_gap0_bps=g0.abs().median() * 1e4,
                 basis_shift_bps=((d.base_post - d.base_pre).abs().median()) * 1e4)
        ks, betas = [], []
        for k in K:
            gk = gap(d, k, base)
            bk, se = slope(gk, g0)
            bin_share, _ = slope(-(d[f"b{k}"] - d["b0"]), g0)
            cme_share, _ = slope(d[f"c{k}"] - d["c0"], g0)
            curves.append(dict(asset=asset, kind=kind, k_min=k, frac_open=bk, se=se,
                               binance_share=bin_share, cme_share=cme_share,
                               mean_abs_gap_bps=gk.abs().mean() * 1e4))
            if k >= 0: ks.append(k); betas.append(bk)
        r["half_life_min"] = half_life(ks, betas)
        for k in [-15, -5, -1, 1, 2, 5, 10, 30, 60]:
            c = [x for x in curves if x["asset"] == asset and x["kind"] == kind and x["k_min"] == k][0]
            r[f"open_{k}m"] = c["frac_open"]
        for k in [1, 5, 30]:
            c = [x for x in curves if x["asset"] == asset and x["kind"] == kind and x["k_min"] == k][0]
            r[f"bin_share_{k}m"] = c["binance_share"]; r[f"cme_share_{k}m"] = c["cme_share"]
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(curves)


def trade_test(ev, Ls=(1, 2, 5), Hs=(5, 15, 30), thetas=(0, 10, 20, 40)):
    rows = []
    groups = list(ev.groupby(["asset", "kind"])) + \
             [(("METALS", kind), d) for kind, d in ev[ev.group == "metals"].groupby("kind")] + \
             [(("ENERGY", kind), d) for kind, d in ev[ev.group == "energy"].groupby("kind")]
    for (asset, kind), d in groups:
        for L in Ls:
            gL = gap(d, L, "pre")
            for H in Hs:
                if L + H not in K: continue
                gLH = gap(d, L + H, "pre")
                one = -np.sign(gL) * (d[f"b{L + H}"] - d[f"b{L}"]) * 1e4
                hed = -np.sign(gL) * (gLH - gL) * 1e4
                for th in thetas:
                    m = gL.abs() * 1e4 > th
                    n = int(m.sum())
                    if n < 5: continue
                    x, h = one[m], hed[m]
                    rows.append(dict(asset=asset, kind=kind, entry_min=L, hold_min=H, min_gap_bps=th, trades=n,
                                     events=len(d), mean_abs_signal_bps=(gL.abs() * 1e4)[m].mean(),
                                     onelegged_mean_bps=x.mean(), onelegged_median_bps=x.median(),
                                     onelegged_t=x.mean() / (x.std(ddof=1) / np.sqrt(n)),
                                     onelegged_hit=(x > 0).mean(),
                                     hedged_mean_bps=h.mean(), hedged_t=h.mean() / (h.std(ddof=1) / np.sqrt(n))))
    return pd.DataFrame(rows)


def trade_test_skipbar(ev, plans=((1, 1), (1, 2), (2, 3), (5, 6)), Hs=(5, 15), thetas=(0, 20)):
    """Signal at minute L, entry at minute E. E = L+1 skips one bar so the entry price is not the same
    1m close that produced the signal (removes bid-ask bounce in last-trade prices)."""
    rows = []
    groups = list(ev.groupby(["asset", "kind"])) + \
             [(("METALS", k), d) for k, d in ev[ev.group == "metals"].groupby("kind")] + \
             [(("METALS ex PA", k), d) for k, d in ev[(ev.group == "metals") & (ev.asset != "PA")].groupby("kind")] + \
             [(("ENERGY", k), d) for k, d in ev[ev.group == "energy"].groupby("kind")]
    for (asset, kind), d in groups:
        for L, E in plans:
            s = gap(d, L, "pre")
            for H in Hs:
                if E + H not in K: continue
                pnl = -np.sign(s) * (d[f"b{E + H}"] - d[f"b{E}"]) * 1e4
                for th in thetas:
                    m = s.abs() * 1e4 > th; n = int(m.sum())
                    if n < 5: continue
                    x = pnl[m]
                    rows.append(dict(asset=asset, kind=kind, signal_min=L, entry_min=E, hold_min=H, min_gap_bps=th,
                                     trades=n, mean_bps=x.mean(), median_bps=x.median(),
                                     t=x.mean() / (x.std(ddof=1) / np.sqrt(n)), hit=(x > 0).mean(),
                                     net_after_10bps_rt=x.mean() - 10))
    return pd.DataFrame(rows)


def main():
    ev = pd.concat([events(r, s) for r, s in PAIRS.items()], ignore_index=True)
    ev.to_parquet(f"{OUT}/q3_paths.parquet")
    print(ev.groupby(["asset", "kind"]).size().unstack())
    summ_post, curves_post = convergence(ev, "post")
    summ_pre, curves_pre = convergence(ev, "pre")
    summ_post.to_csv(f"{OUT}/q3_halflife_summary.csv", index=False)
    curves_post.to_csv(f"{OUT}/q3_halflife_curves.csv", index=False)
    summ_pre.to_csv(f"{OUT}/q3_halflife_summary_prebasis.csv", index=False)
    curves_pre.to_csv(f"{OUT}/q3_halflife_curves_prebasis.csv", index=False)
    tt = trade_test(ev); tt.to_csv(f"{OUT}/q3_trade_test.csv", index=False)
    trade_test_skipbar(ev).to_csv(f"{OUT}/q3_trade_test_skipbar.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("\n== convergence (post-reopen basis)\n", summ_post.round(2).to_string(index=False))
    print("\n== convergence (pre-close basis)\n", summ_pre.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
