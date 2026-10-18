"""Robustness for Q2 in TICK TIME (no grid):
 (a) Hayashi-Yoshida (HY) lead-lag: correlation of CME and Binance mid changes using each market's own
     tick intervals; Binance timestamps shifted by lag L. Peak at L>0 => CME leads.  HY-LLR = sum_{L>0} rho^2 / sum_{L<0} rho^2.
 (b) Event-time transmission: after each CME mid change (sign d), average signed Binance mid move over the
     next h ms, as a fraction of the average CME move size; and the reverse (CME response to Binance moves).
CME: mbp-1 mid (every quote change, exact ns->ms). Binance: trade-reconstructed mid (last seller-initiated
price = bid, last buyer-initiated = ask), evaluated at each trade, kept only when the mid changes.
Only CME open hours (sessions 17:00-16:00 CT); dominant CME contract per session."""
import glob, os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq
PAIRS = {"GC": "XAUUSDT", "SI": "XAGUSDT", "PL": "XPTUSDT", "PA": "XPDUSDT", "HG": "COPPERUSDT",
         "CL": "CLUSDT", "BZ": "BZUSDT", "NG": "NATGASUSDT", "ES": "SPYUSDT", "BTC": "BTCUSDT"}
LAGS = np.array([-5000, -2000, -1000, -500, -300, -200, -150, -100, -75, -50, -25, -10, -5, 0,
                 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 5000])          # ms; >0: Binance later
HORIZ = np.array([10, 50, 100, 250, 1000, 5000])                             # ms
START, END = pd.Timestamp("2026-04-24", tz="UTC"), pd.Timestamp("2026-05-06", tz="UTC")

def session_id(t_ms):
    ct = pd.to_datetime(t_ms, unit="ms", utc=True).tz_convert("America/Chicago")
    closed = (ct.hour == 16) | ((ct.dayofweek == 4) & (ct.hour >= 16)) | (ct.dayofweek == 5) | \
             ((ct.dayofweek == 6) & (ct.hour < 17))
    sid = np.asarray((ct + pd.Timedelta(hours=7)).strftime("%Y%m%d").astype(int))
    sid[np.asarray(closed)] = -1
    return sid

def cme_events(root):
    ts, mid, ins = [], [], []
    for b in pq.ParquetFile(f"data/cme/{root}_mbp1.parquet").iter_batches(
            batch_size=2_000_000, columns=["ts_event", "instrument_id", "bid_px_00", "ask_px_00"]):
        d = b.to_pandas(); d = d[(d.bid_px_00 > 0) & (d.ask_px_00 > d.bid_px_00)]
        t = pd.to_datetime(d.ts_event, utc=True).astype("int64").values // 1_000_000
        keep = np.r_[t[1:] != t[:-1], True]                                     # last update per ms
        ts.append(t[keep]); mid.append(((d.bid_px_00 + d.ask_px_00) / 2).values[keep]); ins.append(d.instrument_id.values[keep])
    t, m, i = map(np.concatenate, (ts, mid, ins)); o = np.argsort(t, kind="stable"); t, m, i = t[o], m[o], i[o]
    sid = session_id(t); keep = sid >= 0
    t, m, i, sid = t[keep], m[keep], i[keep], sid[keep]
    dom = pd.Series(i).groupby(sid).agg(lambda x: x.value_counts().index[0])      # dominant contract per session
    keep = i == dom.reindex(sid).values
    t, m, sid = t[keep], m[keep], sid[keep]
    ch = np.r_[True, np.diff(m) != 0]
    return t[ch], np.log(m[ch]), sid[ch]

def binance_events(sym):
    x = pd.concat(pd.read_csv(f, usecols=["agg_trade_id", "price", "transact_time", "is_buyer_maker"])
                  for f in sorted(glob.glob(f"data/binance/{sym}/aggTrades/*.csv")))
    x = x.sort_values("agg_trade_id")                      # exact execution order, also within the same ms
    t = x.transact_time.values.astype("int64"); t = t // 1000 if t[0] > 1e14 else t
    sell = x.is_buyer_maker.astype(str).str.lower().eq("true").values; p = x.price.values
    bid = pd.Series(np.where(sell, p, np.nan)).ffill().values
    ask = pd.Series(np.where(~sell, p, np.nan)).ffill().values
    m = (bid + ask) / 2; ok = np.isfinite(m); t, m = t[ok], m[ok]
    keep = np.r_[t[1:] != t[:-1], True]; t, m = t[keep], m[keep]                 # last mid per ms
    sid = session_id(t); keep = sid >= 0; t, m, sid = t[keep], m[keep], sid[keep]
    ch = np.r_[True, np.diff(m) != 0]
    return t[ch], np.log(m[ch]), sid[ch]

def hy_sums(tx, x, ty, y, lag):
    """HY covariance between X (CME) and Y shifted: Y observed at ty - lag. Intervals (t_{k-1}, t_k]."""
    dx, dy = np.diff(x), np.diff(y)
    ty = ty - lag
    a, b = tx[:-1], tx[1:]                     # X intervals
    c, d = ty[:-1], ty[1:]                     # Y intervals
    # Y intervals j overlapping (a_i, b_i]: d_j > a_i and c_j < b_i
    j0 = np.searchsorted(d, a, side="right")   # first j with d_j > a_i
    j1 = np.searchsorted(c, b, side="left")    # first j with c_j >= b_i  -> overlap j in [j0, j1)
    cs = np.r_[0.0, np.cumsum(dy)]
    s = cs[np.clip(j1, 0, len(dy))] - cs[np.clip(j0, 0, len(dy))]
    return float(np.sum(dx * s)), float(np.sum(dx * dx)), float(np.sum(dy * dy))

def transmission(t_src, x_src, t_dst, x_dst):
    """Directional response, scale-free: after each source move with sign d, E[d * sign(destination move
    over the next h ms)] = P(same direction) - P(opposite). 0 = no response, 1 = always follows."""
    dx = np.diff(x_src); te = t_src[1:]; d = np.sign(dx)
    i0 = np.searchsorted(t_dst, te, side="right") - 1                         # dst level at event time
    ok = i0 >= 0; out = []
    for h in HORIZ:
        i1 = np.searchsorted(t_dst, te + h, side="right") - 1
        mv = np.where(ok, x_dst[np.clip(i1, 0, None)] - x_dst[np.clip(i0, 0, None)], np.nan)
        out.append(np.nanmean(d * np.sign(mv)))
    return np.array(out), len(dx)

def run(root):
    tx, x, sx = cme_events(root); ty, y, sy = binance_events(PAIRS[root])
    cov = np.zeros(len(LAGS)); vx = np.zeros(len(LAGS)); vy = np.zeros(len(LAGS))
    tr_cb, tr_bc, ncb, nbc, w = np.zeros(len(HORIZ)), np.zeros(len(HORIZ)), 0, 0, 0
    for s in np.intersect1d(np.unique(sx), np.unique(sy)):
        a, b = sx == s, sy == s
        if a.sum() < 100 or b.sum() < 100: continue
        for k, L in enumerate(LAGS):
            c_, v1, v2 = hy_sums(tx[a], x[a], ty[b], y[b], L); cov[k] += c_; vx[k] += v1; vy[k] += v2
        r1, n1 = transmission(tx[a], x[a], ty[b], y[b]); r2, n2 = transmission(ty[b], y[b], tx[a], x[a])
        tr_cb += r1 * n1; ncb += n1; tr_bc += r2 * n2; nbc += n2
    rho = cov / np.sqrt(vx * vy)
    pd.DataFrame({"lag_ms": LAGS, "hy_rho": rho}).to_csv(f"results/q2_hy_{root}.csv", index=False)
    llr = np.sum(rho[LAGS > 0] ** 2) / np.sum(rho[LAGS < 0] ** 2)
    res = dict(asset=root, cme_moves=ncb, binance_moves=nbc, hy_peak_lag_ms=int(LAGS[np.argmax(rho)]),
               hy_peak_rho=rho.max(), hy_rho0=rho[LAGS == 0][0], hy_LLR=llr)
    for h, v in zip(HORIZ, tr_cb / ncb): res[f"bin_after_cme_{h}ms"] = v
    for h, v in zip(HORIZ, tr_bc / nbc): res[f"cme_after_bin_{h}ms"] = v
    pd.DataFrame([res]).to_csv(f"results/q2_tick_{root}.csv", index=False)
    print(pd.DataFrame([res]).round(3).to_string(index=False), flush=True)

if __name__ == "__main__":
    for r in sys.argv[1:] or PAIRS: run(r)
