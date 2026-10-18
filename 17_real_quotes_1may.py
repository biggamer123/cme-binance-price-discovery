"""Validation with NATIVE Binance quotes (Tardis.dev book_ticker = Binance bookTicker feed, exchange
timestamps) on 1 May 2026, 00:00-21:00 UTC (CME open the whole time; Friday close at 21:00 UTC).
 (1) How good is the trade-reconstructed quote?  coverage, error vs true mid (bps), share inside the
     true spread, how much the reconstruction lags the true mid.
 (2) RQ3 re-run with the true Binance mid vs the reconstructed mid on the SAME window:
     100 ms cross-correlation (LLR, peak), VECM information shares, Hayashi-Yoshida peak/LLR,
     directional response D(h).  CME = mbp-1 mid, dominant contract.
Output: results/q5_real_1may_{ROOT}.csv"""
import glob, sys, warnings, numpy as np, pandas as pd, pyarrow.parquet as pq
from statsmodels.tsa.vector_ar.vecm import VECM
warnings.filterwarnings("ignore")
PAIRS = {"GC": "XAUUSDT", "SI": "XAGUSDT", "PL": "XPTUSDT", "PA": "XPDUSDT", "HG": "COPPERUSDT",
         "CL": "CLUSDT", "BZ": "BZUSDT", "NG": "NATGASUSDT", "ES": "SPYUSDT", "BTC": "BTCUSDT"}
W0 = pd.Timestamp("2026-05-01 00:00", tz="UTC").value // 10**6
W1 = pd.Timestamp("2026-05-01 21:00", tz="UTC").value // 10**6
STEP, MAXLAG, STALE = 100, 50, 2000
LAGS = np.array([-5000, -2000, -1000, -500, -300, -200, -150, -100, -75, -50, -25, -10, -5, 0,
                 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 5000])
HORIZ = np.array([100, 1000])
GRID = np.arange(W0, W1, STEP)
CURVES = []

def last_per_ms(t, *cols):
    keep = np.r_[t[1:] != t[:-1], True]
    return (t[keep],) + tuple(c[keep] for c in cols)

def true_bbo(sym):
    ts, b, a = [], [], []
    for x in pd.read_csv(f"data/tardis/book_ticker/2026-05-01/{sym}.csv.gz", chunksize=4_000_000,
                         usecols=["timestamp", "bid_price", "ask_price"]):
        t = x.timestamp.values // 1000
        t, bb, aa = last_per_ms(t, x.bid_price.values, x.ask_price.values)
        ts.append(t); b.append(bb); a.append(aa)
    t, b, a = np.concatenate(ts), np.concatenate(b), np.concatenate(a)
    o = np.argsort(t, kind="stable"); t, b, a = last_per_ms(t[o], b[o], a[o])
    ok = (b > 0) & (a > b); return t[ok], b[ok], a[ok]

def agg(sym):
    fs = [f"data/binance/{sym}/aggTrades/{sym}-aggTrades-{d}.csv" for d in ("2026-04-30", "2026-05-01")]
    x = pd.concat(pd.read_csv(f, usecols=["agg_trade_id", "price", "transact_time", "is_buyer_maker"]) for f in fs)
    x = x.sort_values("agg_trade_id")
    t = x.transact_time.values.astype("int64"); t = t // 1000 if t[0] > 1e14 else t
    sell = x.is_buyer_maker.astype(str).str.lower().eq("true").values
    return t, x.price.values, sell

def sample(t, v, grid):
    i = np.searchsorted(t, grid, side="right") - 1; ok = i >= 0
    out, age = np.full(len(grid), np.nan), np.full(len(grid), np.inf)
    out[ok], age[ok] = v[i[ok]], grid[ok] - t[i[ok]]; return out, age

def cme(root):
    ts, mid, ins = [], [], []
    for b in pq.ParquetFile(f"data/cme/{root}_mbp1.parquet").iter_batches(
            batch_size=2_000_000, columns=["ts_event", "instrument_id", "bid_px_00", "ask_px_00"]):
        d = b.to_pandas()
        t = pd.to_datetime(d.ts_event, utc=True).astype("int64").values // 1_000_000
        m = (t >= W0 - 60_000) & (t < W1) & (d.bid_px_00.values > 0) & (d.ask_px_00.values > d.bid_px_00.values)
        if not m.any(): continue
        ts.append(t[m]); mid.append(((d.bid_px_00 + d.ask_px_00) / 2).values[m]); ins.append(d.instrument_id.values[m])
    t, m, i = map(np.concatenate, (ts, mid, ins)); o = np.argsort(t, kind="stable"); t, m, i = t[o], m[o], i[o]
    dom = pd.Series(i).value_counts().index[0]; k = i == dom
    return last_per_ms(t[k], m[k])

def events(t, m):
    k = (t >= W0) & (t < W1); t, m = t[k], np.log(m[k]); ch = np.r_[True, np.diff(m) != 0]; return t[ch], m[ch]

def xcorr(x, y):
    rx, ry = np.diff(x), np.diff(y); out = []
    for k in range(-MAXLAG, MAXLAG + 1):
        a, b = (rx[:len(rx) - k], ry[k:]) if k >= 0 else (rx[-k:], ry[:len(ry) + k])
        ok = np.isfinite(a) & np.isfinite(b)
        out.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 100 else np.nan)
    r = np.array(out); lags = np.arange(-MAXLAG, MAXLAG + 1) * STEP
    return r, lags

def vecm(lc, lb):
    y = np.column_stack([lc, lb]) * 1e4; y = y[np.isfinite(y).all(1)]
    m = VECM(y, k_ar_diff=10, coint_rank=1, deterministic="ci").fit()
    a, S = m.alpha.ravel(), m.sigma_u; ap = np.array([a[1], -a[0]]); psi = ap / ap.sum()
    def hasb(order):
        P = np.eye(2)[order]; F = np.linalg.cholesky(P @ S @ P.T); v = (psi[order] @ F) ** 2
        res = np.empty(2); res[order[0]] = v[0] / (psi @ S @ psi); res[order[1]] = v[1] / (psi @ S @ psi); return res
    h1, h2 = hasb([0, 1]), hasb([1, 0]); beta = m.beta.ravel(); sp = a @ (beta / beta[0])
    hl = np.log(0.5) / np.log(1 + sp) * STEP / 1000 if -1 < sp < 0 else np.nan
    return psi[0], min(h1[0], h2[0]), max(h1[0], h2[0]), hl

def hy(tx, x, ty, y):
    rho = []
    dx, dy = np.diff(x), np.diff(y); cs = np.r_[0.0, np.cumsum(dy)]
    for L in LAGS:
        s = ty - L; j0 = np.searchsorted(s[1:], tx[:-1], side="right"); j1 = np.searchsorted(s[:-1], tx[1:], side="left")
        v = cs[np.clip(j1, 0, len(dy))] - cs[np.clip(j0, 0, len(dy))]
        rho.append(np.sum(dx * v) / np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2)))
    rho = np.array(rho); CURVES.append(rho); return LAGS[np.argmax(rho)], rho.max(), np.sum(rho[LAGS > 0] ** 2) / np.sum(rho[LAGS < 0] ** 2)

def resp(ts, xs, td, xd):
    d = np.sign(np.diff(xs)); te = ts[1:]; i0 = np.searchsorted(td, te, side="right") - 1; ok = i0 >= 0; out = []
    for h in HORIZ:
        i1 = np.searchsorted(td, te + h, side="right") - 1
        out.append(np.nanmean(np.where(ok, d * np.sign(xd[np.clip(i1, 0, None)] - xd[np.clip(i0, 0, None)]), np.nan)))
    return out

def run(root):
    sym = PAIRS[root]; R = dict(asset=root)
    tt, tb, ta = true_bbo(sym); tm = (tb + ta) / 2
    at, ap, sell = agg(sym)
    # reconstructed quote, grid version (2 s rule), as in the main analysis
    bid, bage = sample(at[sell], ap[sell], GRID); ask, aage = sample(at[~sell], ap[~sell], GRID)
    rec = np.where((bage < STALE) & (aage < STALE), (bid + ask) / 2, np.nan)
    rec60 = np.where((bage < 60_000) & (aage < 60_000), (bid + ask) / 2, np.nan)
    TB, _ = sample(tt, tb, GRID); TA, _ = sample(tt, ta, GRID); TM = (TB + TA) / 2
    _, tage = sample(tt, tm, GRID)
    v = np.isfinite(rec) & np.isfinite(TM)
    err = np.abs(rec[v] / TM[v] - 1) * 1e4
    R.update(true_updates_per_min=len(tt) / ((W1 - W0) / 60000), true_spread_bps_med=np.nanmedian((TA - TB) / TM * 1e4),
             true_quote_age_med_s=np.median(tage) / 1000, rec_coverage=np.isfinite(rec).mean(),
             rec_age_med_s=np.median(np.maximum(bage, aage)) / 1000,
             rec_err_bps_med=np.median(err), rec_err_bps_p90=np.percentile(err, 90),
             rec_inside_spread=np.mean((rec[v] >= TB[v] - 1e-12) & (rec[v] <= TA[v] + 1e-12)),
             rec_err_in_halfspreads_med=np.median(np.abs(rec[v] - TM[v]) / ((TA[v] - TB[v]) / 2)))
    r, lg = xcorr(np.log(TM), np.log(rec60)); R["rec_lag_vs_true_ms"] = int(lg[np.nanargmax(r)]); R["rec_true_corr0"] = r[MAXLAG]
    # RQ3 with CME
    ct, cm = cme(root); C, _ = sample(ct, cm, GRID); lc = np.log(C)
    for name, lb in (("true", np.log(TM)), ("rec", np.log(rec))):
        r, lg = xcorr(lc, lb); pos, neg = np.nansum(r[lg > 0] ** 2), np.nansum(r[lg < 0] ** 2)
        R[f"{name}_LLR"] = pos / neg; R[f"{name}_peak_ms"] = int(lg[np.nanargmax(r)]); R[f"{name}_peak_rho"] = np.nanmax(r)
    for name, lb in (("true", np.log(TM)), ("rec", np.log(rec60))):
        cs, lo, hi, hl = vecm(lc, lb); R.update({f"{name}_CS": cs, f"{name}_IS_lo": lo, f"{name}_IS_hi": hi, f"{name}_halflife_s": hl})
    ex, x = events(ct, cm)
    # reconstructed tick events as in the tick-time script (no staleness rule, mid changes)
    b_ = pd.Series(np.where(sell, ap, np.nan)).ffill().values; a_ = pd.Series(np.where(~sell, ap, np.nan)).ffill().values
    m_ = (b_ + a_) / 2; ok = np.isfinite(m_); rt, rm = last_per_ms(at[ok], m_[ok])
    for name, (ty, y) in (("true", events(tt, tm)), ("rec", events(rt, rm))):
        pk, pr, llr = hy(ex, x, ty, y); R.update({f"{name}_hy_peak_ms": int(pk), f"{name}_hy_rho": pr, f"{name}_hy_LLR": llr,
                                                  f"{name}_moves": len(ty)})
        d1 = resp(ex, x, ty, y); d2 = resp(ty, y, ex, x)
        R.update({f"{name}_D_bin_after_cme_100ms": d1[0], f"{name}_D_bin_after_cme_1s": d1[1],
                  f"{name}_D_cme_after_bin_100ms": d2[0], f"{name}_D_cme_after_bin_1s": d2[1]})
    R["cme_moves"] = len(ex)
    pd.DataFrame({"lag_ms": LAGS, "true": CURVES[-2], "rec": CURVES[-1]}).to_csv(f"results/q5_hy_1may_{root}.csv", index=False)
    pd.DataFrame([R]).to_csv(f"results/q5_real_1may_{root}.csv", index=False)
    print(pd.Series(R).to_string(), flush=True)

if __name__ == "__main__":
    for r in sys.argv[1:] or PAIRS: run(r)
