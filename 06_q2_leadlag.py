"""Q2: when both markets are open, who leads and how fast does the other adjust?
Tick window 24 Apr - 5 May 2026.
  CME     : Databento tbbo -> mid quote before every trade (ts_event, ns)
  Binance : aggTrades -> trade-reconstructed quotes. Seller-initiated trade (is_buyer_maker) ~ bid,
            buyer-initiated ~ ask. Mid = (last bid + last ask)/2, valid only if both sides are < 2 s old.
Analyses (CME open hours only: no daily break 16-17 CT, no weekend)
  1. Cross-correlation of 100 ms returns, lags -5 s..+5 s. corr(r_cme[t], r_bin[t+k]) peaking at k>0 => CME leads.
     Lead-lag ratio LLR = sum_{k>0} rho^2 / sum_{k<0} rho^2 (Huth & Abergel 2014). LLR>1 => CME leads.
  2. Per-session VECM on 1 s log mids -> Hasbrouck information share (bounds over both orderings),
     Gonzalo-Granger component share, and half-life of the CME-Binance price gap.
Outputs: results/q2_xcorr_{ROOT}.csv, results/q2_xcorr.png, results/q2_summary.csv, results/q2_sessions.csv"""
import glob, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import VECM
warnings.filterwarnings("ignore")

PAIRS = {"GC": "XAUUSDT", "SI": "XAGUSDT", "PL": "XPTUSDT", "PA": "XPDUSDT", "HG": "COPPERUSDT",
         "CL": "CLUSDT", "BZ": "BZUSDT", "NG": "NATGASUSDT", "ES": "SPYUSDT", "BTC": "BTCUSDT"}
START, END = pd.Timestamp("2026-04-24", tz="UTC"), pd.Timestamp("2026-05-06", tz="UTC")
STEP_MS, MAXLAG, STALE_MS = 100, 50, 2000          # 100 ms grid, +-5 s, 2 s staleness
os.makedirs("results", exist_ok=True)
# CME_SRC=mbp1 -> use full top-of-book (every quote change) instead of tbbo (quote at trades only)
SRC = os.environ.get("CME_SRC", "tbbo"); SUF = "" if SRC == "tbbo" else f"_{SRC}"

def cme_mid(root):
    """CME mid quotes. Read in batches and keep only the last quote in each 10 ms bucket
    (the analysis grid is 100 ms, so nothing is lost) - keeps ES mbp-1 (65M rows) within memory."""
    import pyarrow.parquet as pq
    ts_all, mid_all, inst_all = [], [], []
    pf = pq.ParquetFile(f"data/cme/{root}_{SRC}.parquet")
    for b in pf.iter_batches(batch_size=2_000_000, columns=["ts_event", "instrument_id", "bid_px_00", "ask_px_00"]):
        d = b.to_pandas()
        d = d[(d.bid_px_00 > 0) & (d.ask_px_00 > d.bid_px_00)]
        t = pd.to_datetime(d.ts_event, utc=True).astype("int64").values // 1_000_000      # ms
        keep = np.r_[t[1:] // 10 != t[:-1] // 10, True]                                    # last per 10 ms
        ts_all.append(t[keep]); mid_all.append(((d.bid_px_00 + d.ask_px_00) / 2).values[keep])
        inst_all.append(d.instrument_id.values[keep])
    t, m, i = np.concatenate(ts_all), np.concatenate(mid_all), np.concatenate(inst_all)
    o = np.argsort(t, kind="stable")
    return t[o], m[o], i[o]

def binance_quotes(sym):
    parts = []
    for f in sorted(glob.glob(f"data/binance/{sym}/aggTrades/*.csv")):
        x = pd.read_csv(f, usecols=["price", "transact_time", "is_buyer_maker"])
        parts.append(x)
    x = pd.concat(parts)
    t = x.transact_time.values.astype("int64")
    if t[0] > 1e14: t = t // 1000                                                         # us -> ms
    sell = x.is_buyer_maker.astype(str).str.lower().eq("true").values
    o = np.argsort(t, kind="stable")
    t, p, sell = t[o], x.price.values[o], sell[o]
    return (t[sell], p[sell]), (t[~sell], p[~sell])

def sample(t, v, grid):
    """previous-tick value on grid, plus age (ms) of that tick"""
    i = np.searchsorted(t, grid, side="right") - 1
    ok = i >= 0
    out, age = np.full(len(grid), np.nan), np.full(len(grid), np.inf)
    out[ok], age[ok] = v[i[ok]], grid[ok] - t[i[ok]]
    return out, age

def build(root, sym, step_ms, stale_ms=STALE_MS):
    ct, cm, cinst = cme_mid(root)
    (bt, bp), (at, ap) = binance_quotes(sym)
    grid = np.arange(START.value // 1_000_000, END.value // 1_000_000, step_ms)
    c, _ = sample(ct, cm, grid)
    inst, _ = sample(ct, cinst.astype(float), grid)
    bid, bage = sample(bt, bp, grid)
    ask, aage = sample(at, ap, grid)
    b = np.where((bage < stale_ms) & (aage < stale_ms), (bid + ask) / 2, np.nan)
    ts = pd.to_datetime(grid, unit="ms", utc=True).tz_convert("America/Chicago")
    closed = (ts.hour == 16) | ((ts.dayofweek == 4) & (ts.hour >= 16)) | (ts.dayofweek == 5) | \
             ((ts.dayofweek == 6) & (ts.hour < 17))
    c[closed] = np.nan
    session = (ts + pd.Timedelta(hours=7)).date                                         # 17:00 CT starts next session
    return grid, np.log(c), np.log(b), inst, np.asarray(session)

def xcorr(lc, lb, inst):
    rc, rb = np.diff(lc), np.diff(lb)
    rc[np.diff(inst) != 0] = np.nan
    out = []
    for k in range(-MAXLAG, MAXLAG + 1):
        if k >= 0: x, y = rc[:len(rc) - k], rb[k:]
        else:      x, y = rc[-k:], rb[:len(rb) + k]
        m = np.isfinite(x) & np.isfinite(y)
        out.append((k * STEP_MS / 1000, np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 100 else np.nan))
    return pd.DataFrame(out, columns=["lag_s", "rho"])

def info_shares(lc, lb, lags=10):
    y = np.column_stack([lc, lb]) * 1e4                     # bps scale for numerical stability
    y = y[np.isfinite(y).all(1)]
    if len(y) < 300_000: return None          # skip partial sessions (< ~8 h)
    m = VECM(y, k_ar_diff=lags, coint_rank=1, deterministic="ci").fit()
    a, S = m.alpha.ravel(), m.sigma_u
    ap = np.array([a[1], -a[0]]); psi = ap / ap.sum()       # Gonzalo-Granger component shares
    def hasb(order):
        P = np.eye(2)[order]; F = np.linalg.cholesky(P @ S @ P.T)
        v = (psi[order] @ F) ** 2; tot = psi @ S @ psi
        res = np.empty(2); res[order[0]] = v[0] / tot; res[order[1]] = v[1] / tot; return res
    h1, h2 = hasb([0, 1]), hasb([1, 0])
    beta = m.beta.ravel(); speed = a @ (beta / beta[0])     # gap z = p_c - b*p_b ; dz ~ speed*z
    hl = np.log(0.5) / np.log(1 + speed) * STEP_MS / 1000 if -1 < speed < 0 else np.nan   # seconds
    return dict(n=len(y), cs_cme=psi[0], is_cme_lo=min(h1[0], h2[0]), is_cme_hi=max(h1[0], h2[0]),
                alpha_cme=a[0], alpha_bin=a[1], gap_halflife_s=hl)

def main():
    summ, sess_rows, curves = [], [], {}
    for root, sym in PAIRS.items():
        if not os.path.exists(f"data/cme/{root}_{SRC}.parquet"): print("skip", root); continue
        _, lc, lb, inst, _ = build(root, sym, STEP_MS)
        xc = xcorr(lc, lb, inst); xc.to_csv(f"results/q2_xcorr_{root}{SUF}.csv", index=False); curves[root] = xc
        pos, neg = xc[xc.lag_s > 0].rho ** 2, xc[xc.lag_s < 0].rho ** 2
        peak = xc.loc[xc.rho.idxmax()]
        # VECM on the same 100 ms grid with continuous (previous-tick) prices, one fit per CME session.
        # 10 lags = 1 s of dynamics. Coarser grids make both markets move in the same bucket and the
        # Hasbrouck bounds uninformative.
        _, lc1, lb1, _, sess = build(root, sym, STEP_MS, stale_ms=60_000)
        rows = []
        for s in np.unique(sess):
            r = info_shares(lc1[sess == s], lb1[sess == s])
            if r: r.update(asset=root, session=str(s)); rows.append(r)
        sd = pd.DataFrame(rows); sess_rows.append(sd)
        valid = np.isfinite(lc) & np.isfinite(lb)
        summ.append(dict(asset=root, binance=sym, coverage=valid.mean() / np.isfinite(lc).mean(),
                         rho0=xc.loc[xc.lag_s == 0, "rho"].iloc[0], peak_lag_s=peak.lag_s, peak_rho=peak.rho,
                         LLR=pos.sum() / neg.sum(), sessions=len(sd),
                         IS_cme_lo=sd.is_cme_lo.median(), IS_cme_hi=sd.is_cme_hi.median(),
                         IS_cme_mid=((sd.is_cme_lo + sd.is_cme_hi) / 2).median(),
                         CS_cme=sd.cs_cme.median(), gap_halflife_s=sd.gap_halflife_s.median()))
        pd.DataFrame(summ[-1:]).to_csv(f"results/q2_summary_{root}{SUF}.csv", index=False)
        sd.to_csv(f"results/q2_sessions_{root}{SUF}.csv", index=False)
        print(pd.DataFrame(summ[-1:]).round(3).to_string(index=False), flush=True)
    out = pd.DataFrame(summ); out.to_csv(f"results/q2_summary{SUF}.csv", index=False)
    pd.concat(sess_rows).to_csv(f"results/q2_sessions{SUF}.csv", index=False)
    n = len(curves); fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6), sharex=True)
    for ax, (root, xc) in zip(axs.ravel(), curves.items()):
        ax.bar(xc.lag_s, xc.rho, width=0.09); ax.axvline(0, lw=0.6, color="k")
        ax.set_title(f"{root} vs {PAIRS[root]}", fontsize=9)
    for ax in axs.ravel(): ax.set_xlabel("lag (s); >0 = CME leads", fontsize=8)
    fig.tight_layout(); fig.savefig(f"results/q2_xcorr{SUF}.png", dpi=150)
    print(out.round(3).to_string(index=False))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1: PAIRS = {k: v for k, v in PAIRS.items() if k in sys.argv[1:]}
    main()
