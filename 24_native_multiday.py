"""Native Binance quotes on any UTC day: recorded best bid/offer files
(data/native_provider/<SYM>/<day>/*.parquet) or Tardis (data/tardis/book_ticker/<day>/<SYM>.csv.gz).
Per day and market:
  RQ3    with the NATIVE Binance mid on CME-open time: 100 ms cross-correlation (LLR, peak), VECM information-share
         bounds on the longest open stretch (>= 8.3 h), Hayashi-Yoshida peak, directional response D(100 ms) both ways.
         Needs CME mbp-1 for that day: data/cme/days/<ROOT>_mbp1_<day>.parquet (see README, Getting the data), or the
         23 Apr - 5 May tick window file.
  VALID  trade-rebuilt quote vs native (valid share, error, LLR, HY peak) if that day's aggTrades file exists.
  REOPEN every CME reopen of the main analysis (11_q3_reopen_halflife.events: same reopen time, same pre-close
         basis, same roll / bad-day filters) that falls in the day: gap path in SECONDS with the native mid, who
         moves, native spread and depth at entry, the gap 1 min after the reopen with the last trade (main analysis)
         vs the native mid, and the two-sided trade (enter at 1 min if |gap| > 20 bps, exit at 16 min) at mid and,
         when CME mbp-1 exists, EXECUTED at the real bid/ask on both venues.  Without mbp-1 the CME 1 s close is used.
Usage:  python 24_native_multiday.py 2026-07-14 2026-07-19 ...  [--assets GC,SI]
Outputs (re-running a day replaces its rows): results/q7_rq3_days.csv, results/q7_hy_days.csv,
results/q7_valid_days.csv, results/q7_reopen_native.csv"""
import glob, importlib.util, os, sys, warnings
import numpy as np, pandas as pd, pyarrow.parquet as pq
from statsmodels.tsa.vector_ar.vecm import VECM
warnings.filterwarnings("ignore")

PAIRS = {"GC": "XAUUSDT", "SI": "XAGUSDT", "PL": "XPTUSDT", "PA": "XPDUSDT", "HG": "COPPERUSDT",
         "CL": "CLUSDT", "BZ": "BZUSDT", "NG": "NATGASUSDT", "ES": "SPYUSDT", "BTC": "BTCUSDT"}
STEP, MAXLAG, STALE = 100, 50, 2000
LAGS = np.array([-5000, -2000, -1000, -500, -300, -200, -150, -100, -75, -50, -25, -10, -5, 0,
                 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 5000])
KS = [0, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 900, 960, 1800]     # seconds after the reopen
FEE_T = 5.0                     # Binance taker fee, bps per side (assumed VIP0)
THETA = 20.0                    # trade filter, bps
MIN_PTS = 300_000               # >= 8.3 h of 100 ms points for a VECM
TICK_WINDOW = ("2026-04-23", "2026-05-05")                            # dates in data/cme/<ROOT>_mbp1.parquet
LISTING_DAY = {("CL", "2026-04-01"), ("BZ", "2026-04-01"), ("NG", "2026-04-01")}


# ---------------------------------------------------------------- helpers
def last_per_ms(t, *cols):
    keep = np.r_[t[1:] != t[:-1], True]
    return (t[keep],) + tuple(c[keep] for c in cols)


def asof(t, v, q):
    """value and age (ms) of the last observation at or before each q"""
    i = np.searchsorted(t, q, side="right") - 1
    j = np.clip(i, 0, None)
    return np.where(i >= 0, v[j], np.nan), np.where(i >= 0, q - t[j], np.inf)


def closed(t_ms):
    ct = pd.to_datetime(t_ms, unit="ms", utc=True).tz_convert("America/Chicago")
    h, dw = ct.hour.values, ct.dayofweek.values
    return (h == 16) | ((dw == 4) & (h >= 16)) | (dw == 5) | ((dw == 6) & (h < 17))


def session(t_ms):
    """trading-session id (CT date of the session's end), -1 when CME is closed"""
    ct = pd.to_datetime(t_ms, unit="ms", utc=True).tz_convert("America/Chicago")
    sid = np.asarray((ct + pd.Timedelta(hours=7)).strftime("%Y%m%d").astype(int))
    sid[closed(t_ms)] = -1
    return sid


# ---------------------------------------------------------------- loaders
def load_native(sym, day):
    files = sorted(glob.glob(f"data/native_provider/{sym}/{day}/*.parquet"))
    parts, parts_e = [], []
    if files:
        for f in files:
            x = pd.read_parquet(f, columns=["recv_ns", "T_ms", "E_ms", "e", "u", "bid", "bid_qty", "ask", "ask_qty"])
            parts_e.append(set(x.e.dropna().unique()))
            t = x["T_ms"].astype("float64").fillna(x["E_ms"].astype("float64")).fillna(x["recv_ns"] // 1_000_000)
            parts.append(pd.DataFrame({"t": t.values.astype("int64"), "u": x["u"].astype("float64").values, "e": x.e.values,
                                       "bid": x.bid.values, "ask": x.ask.values,
                                       "bq": x.bid_qty.values, "aq": x.ask_qty.values}))
        d = pd.concat(parts, ignore_index=True); del parts
        if (d.e == "bbo").any():                             # best bid/offer stream; book snapshots only as a fallback
            d = d[d.e == "bbo"]
        src = "provider " + "/".join(sorted(d.e.dropna().unique()))
        d = d[~(d.u.notna() & d.u.duplicated())]             # same update recorded in two source files
        d = d.sort_values(["t", "u"], kind="stable")
    else:
        f = f"data/tardis/book_ticker/{day}/{sym}.csv.gz"
        if not os.path.exists(f):
            return None
        for x in pd.read_csv(f, chunksize=4_000_000, usecols=["timestamp", "bid_price", "ask_price", "bid_amount", "ask_amount"]):
            t = x.timestamp.values // 1000
            t, b, a, bq, aq = last_per_ms(t, x.bid_price.values, x.ask_price.values, x.bid_amount.values, x.ask_amount.values)
            parts.append(pd.DataFrame({"t": t, "bid": b, "ask": a, "bq": bq, "aq": aq}))
        d = pd.concat(parts, ignore_index=True).sort_values("t", kind="stable"); del parts
        src = "tardis"
    d = d[(d.bid > 0) & (d.ask > d.bid)]
    if len(d) < 1000:
        return None
    t, b, a, bq, aq = last_per_ms(d.t.values, d.bid.values, d.ask.values, d.bq.values, d.aq.values)
    return dict(t=t, bid=b, ask=a, bq=bq, aq=aq, src=src)


def load_cme(root, day, d0, d1):
    f = f"data/cme/days/{root}_mbp1_{day}.parquet"
    if not os.path.exists(f):
        f = f"data/cme/{root}_mbp1.parquet"
        if not (TICK_WINDOW[0] <= day <= TICK_WINDOW[1] and os.path.exists(f)):
            return None
    ts, bb, aa, ins = [], [], [], []
    for b in pq.ParquetFile(f).iter_batches(batch_size=2_000_000, columns=["ts_event", "instrument_id", "bid_px_00", "ask_px_00"]):
        x = b.to_pandas()
        t = pd.to_datetime(x.ts_event, utc=True).astype("int64").values // 1_000_000
        m = (t >= d0 - 3_600_000) & (t < d1) & (x.bid_px_00.values > 0) & (x.ask_px_00.values > x.bid_px_00.values)
        if m.any():
            ts.append(t[m]); bb.append(x.bid_px_00.values[m]); aa.append(x.ask_px_00.values[m]); ins.append(x.instrument_id.values[m])
    if not ts:
        return None
    t, b, a, i = map(np.concatenate, (ts, bb, aa, ins))
    o = np.argsort(t, kind="stable"); t, b, a, i = t[o], b[o], a[o], i[o]
    dom = pd.Series(i).value_counts().index[0]; k = i == dom
    t, b, a = last_per_ms(t[k], b[k], a[k])
    return dict(t=t, bid=b, ask=a, mid=(b + a) / 2, inst=int(dom))


_C1S = {}
def load_cme_1s(root):
    """CME 1 s bars: time the bar ends (ms), close, contract id"""
    if root not in _C1S:
        d = pd.read_parquet(f"data/cme/{root}_ohlcv1s.parquet", columns=["close", "instrument_id"]).sort_index()
        _C1S[root] = (pd.to_datetime(d.index, utc=True).asi8 // 1_000_000 + 1000, d.close.values, d.instrument_id.values)
    return _C1S[root]


def load_agg(sym, day):
    prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fs = [f"data/binance/{sym}/aggTrades/{sym}-aggTrades-{d}.csv" for d in (prev, day)]
    if not os.path.exists(fs[1]):
        return None
    x = pd.concat(pd.read_csv(f, usecols=["agg_trade_id", "price", "transact_time", "is_buyer_maker"]) for f in fs if os.path.exists(f))
    x = x.sort_values("agg_trade_id")
    t = x.transact_time.values.astype("int64"); t = t // 1000 if t[0] > 1e14 else t
    return t, x.price.values, x.is_buyer_maker.astype(str).str.lower().eq("true").values


_Q3, _EV = None, {}
def reopen_events(root):
    """the main analysis' reopen events (11_q3_reopen_halflife.py), cached per market"""
    global _Q3
    if _Q3 is None and not os.path.exists(f"results/q7_cache/events_{root}.pkl"):
        spec = importlib.util.spec_from_file_location("q3", "11_q3_reopen_halflife.py")
        _Q3 = importlib.util.module_from_spec(spec)
        argv, sys.argv = sys.argv, ["x"]
        spec.loader.exec_module(_Q3)
        sys.argv = argv
    if root not in _EV:
        cache = f"results/q7_cache/events_{root}.pkl"
        if os.path.exists(cache):
            _EV[root] = pd.read_pickle(cache)
        else:
            _EV[root] = _Q3.events(root, PAIRS[root]) if root in _Q3.PAIRS else pd.DataFrame()
            os.makedirs("results/q7_cache", exist_ok=True); _EV[root].to_pickle(cache)
    return _EV[root]


# ---------------------------------------------------------------- measures
def xcorr(x, y):
    rx, ry = np.diff(x), np.diff(y); out = []
    for k in range(-MAXLAG, MAXLAG + 1):
        a, b = (rx[:len(rx) - k], ry[k:]) if k >= 0 else (rx[-k:], ry[:len(ry) + k])
        ok = np.isfinite(a) & np.isfinite(b)
        out.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 100 else np.nan)
    r = np.array(out); lg = np.arange(-MAXLAG, MAXLAG + 1) * STEP
    if not np.isfinite(r).any():
        return np.nan, np.nan, np.nan
    return np.nansum(r[lg > 0] ** 2) / np.nansum(r[lg < 0] ** 2), int(lg[np.nanargmax(r)]), np.nanmax(r)


def vecm(lc, lb):
    y = np.column_stack([lc, lb]) * 1e4; y = y[np.isfinite(y).all(1)]
    if len(y) < MIN_PTS:
        return [np.nan] * 4
    m = VECM(y, k_ar_diff=10, coint_rank=1, deterministic="ci").fit()
    a, S = m.alpha.ravel(), m.sigma_u; ap = np.array([a[1], -a[0]]); psi = ap / ap.sum()
    def hasb(order):
        P = np.eye(2)[order]; F = np.linalg.cholesky(P @ S @ P.T); v = (psi[order] @ F) ** 2
        res = np.empty(2); res[order[0]] = v[0] / (psi @ S @ psi); res[order[1]] = v[1] / (psi @ S @ psi); return res
    h1, h2 = hasb([0, 1]), hasb([1, 0]); beta = m.beta.ravel(); sp = a @ (beta / beta[0])
    return psi[0], min(h1[0], h2[0]), max(h1[0], h2[0]), (np.log(0.5) / np.log(1 + sp) * STEP / 1000 if -1 < sp < 0 else np.nan)


def hy_sums(tx, x, ty, y, L):
    dx, dy = np.diff(x), np.diff(y); s = ty - L; cs = np.r_[0.0, np.cumsum(dy)]
    j0 = np.searchsorted(s[1:], tx[:-1], side="right"); j1 = np.searchsorted(s[:-1], tx[1:], side="left")
    v = cs[np.clip(j1, 0, len(dy))] - cs[np.clip(j0, 0, len(dy))]
    return np.sum(dx * v), np.sum(dx ** 2), np.sum(dy ** 2)


def resp(ts, xs, td, xd, h=100):
    """sum over moves of the leader of sign(move) * sign(follower's move over the next h ms), and the count"""
    d = np.sign(np.diff(xs)); te = ts[1:]
    i0 = np.searchsorted(td, te, side="right") - 1; i1 = np.searchsorted(td, te + h, side="right") - 1
    ok = i0 >= 0
    return np.sum(np.where(ok, d * np.sign(xd[np.clip(i1, 0, None)] - xd[np.clip(i0, 0, None)]), 0)), int(ok.sum())


def moves(t, mid, d0, d1):
    """mid changes inside [d0, d1) on CME-open time: times, log mids, session ids"""
    k = (t >= d0) & (t < d1); t, m = t[k], np.log(mid[k]); ch = np.r_[True, np.diff(m) != 0]; t, m = t[ch], m[ch]
    s = session(t); k = s >= 0
    return t[k], m[k], s[k]


def tick_measures(cm, bm):
    """HY curve, its peak, and D both ways, summed over sessions (cm, bm = (times, log mids, session ids))"""
    cov, vx, vy = np.zeros(len(LAGS)), np.zeros(len(LAGS)), np.zeros(len(LAGS)); d1 = d2 = n1 = n2 = 0
    for s in np.intersect1d(np.unique(cm[2]), np.unique(bm[2])):
        a, b = cm[2] == s, bm[2] == s
        if a.sum() < 100 or b.sum() < 100:
            continue
        for k, L in enumerate(LAGS):
            c_, v1, v2 = hy_sums(cm[0][a], cm[1][a], bm[0][b], bm[1][b], L); cov[k] += c_; vx[k] += v1; vy[k] += v2
        x1, m1 = resp(cm[0][a], cm[1][a], bm[0][b], bm[1][b]); x2, m2 = resp(bm[0][b], bm[1][b], cm[0][a], cm[1][a])
        d1 += x1; n1 += m1; d2 += x2; n2 += m2
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = cov / np.sqrt(vx * vy)
    if not np.isfinite(rho).any():
        return np.nan, np.nan, np.nan, np.nan, rho
    return int(LAGS[np.nanargmax(rho)]), np.nanmax(rho), d1 / max(n1, 1), d2 / max(n2, 1), rho


# ---------------------------------------------------------------- per day and market
def contract_info():
    f = "results/costs_two_sided.csv"
    return pd.read_csv(f).set_index("asset") if os.path.exists(f) else None


def run_day(day, root, cinfo):
    sym = PAIRS[root]; d0 = pd.Timestamp(day, tz="UTC").value // 1_000_000; d1 = d0 + 86_400_000
    if (root, day) in LISTING_DAY:
        return None, None, None, []
    nat = load_native(sym, day)
    if nat is None:
        return None, None, None, []
    grid = np.arange(d0, d1, STEP); cl = closed(grid)
    B, bage = asof(nat["t"], nat["bid"], grid); A, _ = asof(nat["t"], nat["ask"], grid); TM = (B + A) / 2
    have = (grid >= nat["t"][0]) & (grid <= nat["t"][-1])
    rq3 = dict(day=day, asset=root, native_source=nat["src"], native_updates=len(nat["t"]),
               native_first_utc=pd.Timestamp(nat["t"][0], unit="ms").strftime("%H:%M"),
               native_last_utc=pd.Timestamp(nat["t"][-1], unit="ms").strftime("%H:%M"),
               open_hours=(~cl & have).sum() * STEP / 3.6e6,
               native_spread_bps=np.nanmedian(((A - B) / TM * 1e4)[~cl & have]) if (~cl & have).any() else np.nan,
               native_age_med_s=np.median(bage[have]) / 1000 if have.any() else np.nan)
    hy, valid = None, None
    cme = load_cme(root, day, d0, d1) if rq3["open_hours"] >= 1 else None
    rq3["cme_ticks"] = cme is not None
    if cme is not None:
        C, _ = asof(cme["t"], cme["mid"], grid); C[cl] = np.nan; TM[cl | ~have] = np.nan
        lc, lb = np.log(C), np.log(TM)
        rq3["LLR"], rq3["peak_ms"], rq3["peak_rho"] = xcorr(lc, lb)
        o = np.isfinite(lc) & np.isfinite(lb)                        # longest stretch with both prices
        e = np.flatnonzero(np.diff(np.r_[0, o.astype(int), 0])).reshape(-1, 2)
        r0, r1 = e[np.argmax(e[:, 1] - e[:, 0])]
        rq3["vecm_hours"] = (r1 - r0) * STEP / 3.6e6
        rq3["CS"], rq3["IS_lo"], rq3["IS_hi"], rq3["halflife_s"] = vecm(lc[r0:r1], lb[r0:r1])
        ce = moves(cme["t"], cme["mid"], d0, d1); be = moves(nat["t"], (nat["bid"] + nat["ask"]) / 2, d0, d1)
        rq3["cme_moves"], rq3["bin_moves"] = len(ce[0]), len(be[0])
        rq3["HY_peak_ms"], rq3["HY_rho"], rq3["D_bin_after_cme"], rq3["D_cme_after_bin"], rho = tick_measures(ce, be)
        hy = pd.DataFrame({"day": day, "asset": root, "lag_ms": LAGS, "rho_native": rho})
        agg = load_agg(sym, day)
        if agg is not None:
            at, ap, sell = agg
            rb, ba_ = asof(at[sell], ap[sell], grid); ra, aa_ = asof(at[~sell], ap[~sell], grid)
            rec = np.where((ba_ < STALE) & (aa_ < STALE) & ~cl, (rb + ra) / 2, np.nan)
            v = np.isfinite(rec) & np.isfinite(TM)
            bm_ = pd.Series(np.where(sell, ap, np.nan)).ffill().values; am_ = pd.Series(np.where(~sell, ap, np.nan)).ffill().values
            mm = (bm_ + am_) / 2; ok = np.isfinite(mm); rt, rmid = last_per_ms(at[ok], mm[ok])
            pk, _, _, _, rho_r = tick_measures(ce, moves(rt, rmid, d0, d1))
            hy["rho_rebuilt"] = rho_r
            valid = dict(day=day, asset=root, rebuilt_valid=np.isfinite(rec[~cl & have]).mean(),
                         rebuilt_err_bps=np.median(np.abs(rec[v] / TM[v] - 1) * 1e4) if v.any() else np.nan,
                         LLR_native=rq3["LLR"], LLR_rebuilt=xcorr(lc, np.log(rec))[0],
                         HY_peak_native_ms=rq3["HY_peak_ms"], HY_peak_rebuilt_ms=pk)
    return rq3, hy, valid, reopens(day, root, nat, cme, d0, d1, cinfo)


def reopens(day, root, nat, cme, d0, d1, cinfo):
    ev = reopen_events(root)
    if ev.empty:
        return []
    ev = ev[(ev.reopen.astype("int64") // 1_000_000 >= d0) & (ev.reopen.astype("int64") // 1_000_000 < d1)]
    rows = []
    for _, e in ev.iterrows():
        t1 = e.reopen.value // 1_000_000; q = t1 + np.array(KS) * 1000
        if nat["t"][0] > t1 - 60_000 or nat["t"][-1] < q[-1]:
            continue                                              # native file does not cover the reopen
        bb, age = asof(nat["t"], nat["bid"], q); ba, _ = asof(nat["t"], nat["ask"], q)
        bq, _ = asof(nat["t"], nat["bq"], q); aq, _ = asof(nat["t"], nat["aq"], q)
        bmid = (bb + ba) / 2
        ct1, cc1, ci1 = load_cme_1s(root)
        inst = ci1[np.searchsorted(ct1, t1 + 1000, side="left")]    # contract of the main analysis
        have_mbp = cme is not None and cme["inst"] == inst and cme["t"][0] <= t1 and cme["t"][-1] >= q[-1]
        if have_mbp:
            cmid, _ = asof(cme["t"], cme["mid"], q)
        else:
            k = ci1 == inst; cmid, _ = asof(ct1[k], cc1[k], q)
        cmid[0] = np.exp(e.c0)                                    # CME opening print, as in the main analysis
        g = (np.log(bmid) - np.log(cmid) - e.base_pre) * 1e4
        r = dict(day=day, asset=root, kind=e.kind, reopen_utc=str(e.reopen), cme_quotes="mbp-1" if have_mbp else "1s close",
                 native_age_max_s=np.max(age) / 1000)
        # basis from the native mid (only when the native file covers the 30 min before the close)
        t0 = e.close.value // 1_000_000; pre = t0 + 1000 - np.arange(0, 31) * 60_000
        if nat["t"][0] <= pre[-1]:
            pb, _ = asof(nat["t"], nat["bid"], pre); pa, _ = asof(nat["t"], nat["ask"], pre)
            k = ci1 == inst; pc, _ = asof(ct1[k], cc1[k], pre)
            r["basis_native_minus_trade_bps"] = (np.nanmedian(np.log((pb + pa) / 2) - np.log(pc)) - e.base_pre) * 1e4
        for kk, v in zip(KS, g):
            r[f"g{kk}"] = v
        r["g60_last"] = (e.b1 - e.c1 - e.base_pre) * 1e4                # main analysis: last Binance trade
        r["g960_last"] = (e.b16 - e.c16 - e.base_pre) * 1e4
        i5 = KS.index(300)
        r["bin_move_5m_bps"] = -(np.log(bmid[i5]) - np.log(bmid[0])) * 1e4
        r["cme_move_5m_bps"] = (np.log(cmid[i5]) - np.log(cmid[0])) * 1e4
        en, ex = KS.index(60), KS.index(960)
        r["bin_spread_entry_bps"] = (ba[en] - bb[en]) / bmid[en] * 1e4
        r["signal_last"] = int(np.sign(r["g60_last"])) if abs(r["g60_last"]) > THETA else 0
        r["signal_native"] = int(np.sign(g[en])) if abs(g[en]) > THETA else 0
        if r["signal_native"]:
            s = r["signal_native"]
            r["trade_side"] = "sell Binance / buy CME" if s > 0 else "buy Binance / sell CME"
            r["gross_mid_bps"] = -s * (g[ex] - g[en])
            r["bin_depth_entry_usd"] = (bq[en] * bb[en]) if s > 0 else (aq[en] * ba[en])
            if have_mbp:
                cb, _ = asof(cme["t"], cme["bid"], q); ca, _ = asof(cme["t"], cme["ask"], q)
                b_in, b_out = (bb[en], ba[ex]) if s > 0 else (ba[en], bb[ex])
                c_in, c_out = (ca[en], cb[ex]) if s > 0 else (cb[en], ca[ex])
                r["exec_gross_bps"] = (s * (np.log(b_in) - np.log(b_out)) + s * (np.log(c_out) - np.log(c_in))) * 1e4
                r["fees_bps"] = 2 * FEE_T + (cinfo.loc[root, "cme_fee_rt"] if cinfo is not None and root in cinfo.index else 0.0)
                r["exec_net_bps"] = r["exec_gross_bps"] - r["fees_bps"]
        rows.append(r)
    return rows


def save(rows, path, keys=("day", "asset")):
    if not rows:
        return
    new = pd.concat(rows, ignore_index=True) if isinstance(rows[0], pd.DataFrame) else pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old.set_index(list(keys)).index.isin(new.set_index(list(keys)).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def fmt(v):
    return f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)


if __name__ == "__main__":
    args = sys.argv[1:]
    assets = list(PAIRS)
    if "--assets" in args:
        i = args.index("--assets"); assets = args[i + 1].split(","); args = args[:i] + args[i + 2:]
    cinfo = contract_info()
    for day in args:
        R, H, V, O = [], [], [], []
        for root in assets:
            rq3, hy, valid, rr = run_day(day, root, cinfo)
            if rq3 is None:
                print(f"{day} {root}: no native quotes", flush=True); continue
            R.append(rq3); H += [hy] if hy is not None else []; V += [valid] if valid else []; O += rr
            print(f"{day} {root}: " + ", ".join(f"{k}={fmt(v)}" for k, v in rq3.items() if k not in ("day", "asset")), flush=True)
            if valid:
                print("   rebuilt vs native: " + ", ".join(f"{k}={fmt(v)}" for k, v in valid.items() if k not in ("day", "asset")), flush=True)
            for r in rr:
                print("   reopen: " + ", ".join(f"{k}={fmt(r[k])}" for k in ("kind", "cme_quotes", "g0", "g60", "g60_last", "g300", "g960",
                      "signal_last", "signal_native", "gross_mid_bps", "exec_net_bps", "bin_depth_entry_usd") if k in r), flush=True)
        save(R, "results/q7_rq3_days.csv"); save(H, "results/q7_hy_days.csv")
        save(V, "results/q7_valid_days.csv"); save(O, "results/q7_reopen_native.csv")
