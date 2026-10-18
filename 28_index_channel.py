"""Index channel (second review, point 2).  The Binance TradFi perpetuals are marked and funded against an index
built from per-second vendor quotes.  Is the CME lead in joint hours only the perpetual following that index?
Inputs: data/tardis/derivative_ticker/<day>/<SYM>.csv.gz (index_price, mark_price, last_price, funding_rate, ~1 s),
        native Binance quotes and CME mbp-1 through 24_native_multiday.py.
CME OPEN (100 ms grid, days with CME mbp-1):
  index update interval; lead-lag CME->index, perp->index, CME->perp (LLR, peak);
  perp 100 ms returns on lags 1..L of CME and index returns: R2 with index lags only, CME lags only, both;
  share of the CME R2 that survives the index lags; summed CME coefficients with and without index lags.
CME CLOSED (every day file): index changes, perp premium to the index, mark - index gap and the 3% band,
  each funding settlement against the average premium over its interval.
Usage: python 28_index_channel.py 2026-06-01 ... [--assets GC,SI] [--closed-only]
       python 28_index_channel.py --regimes   (weekend slope before/after the 8 May 2026 index change)
Outputs (re-running a day replaces its rows): results/q8_index_open.csv, results/q8_index_closed.csv,
results/q8_index_funding.csv"""
import importlib.util, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("nm", "24_native_multiday.py")
nm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nm)
PAIRS = {k: v for k, v in nm.PAIRS.items() if k != "BTC"}
STEP, L = 100, 30                      # grid step (ms) and regression lags (3 s)


def load_index(sym, day):
    f = f"data/tardis/derivative_ticker/{day}/{sym}.csv.gz"
    if not os.path.exists(f):
        return None
    x = pd.read_csv(f, usecols=["timestamp", "funding_timestamp", "funding_rate", "last_price", "index_price", "mark_price"])
    x = x.dropna(subset=["index_price", "mark_price"]).sort_values("timestamp", kind="stable")
    t = x.timestamp.values // 1000
    keep = np.r_[t[1:] != t[:-1], True]
    return dict(t=t[keep], idx=x.index_price.values[keep], mark=x.mark_price.values[keep], last=x.last_price.values[keep],
                fr=x.funding_rate.values[keep], fts=x.funding_timestamp.values[keep] // 1000)


def lagged_r2(rp, rc, ri, ok, L=L):
    """perp return on L lags of CME and index returns (+ const).  ok = rows usable (y and all lags in open time)"""
    rows = np.flatnonzero(ok); rows = rows[rows >= L]
    K = 1 + 2 * L
    XtX = np.zeros((K, K)); Xty = np.zeros(K); yty = 0.0; sy = 0.0; n = 0
    for c0 in range(0, len(rows), 100_000):
        r = rows[c0:c0 + 100_000]
        X = np.empty((len(r), K)); X[:, 0] = 1.0
        for j in range(1, L + 1):
            X[:, j] = rc[r - j]; X[:, L + j] = ri[r - j]
        y = rp[r]
        XtX += X.T @ X; Xty += X.T @ y; yty += y @ y; sy += y.sum(); n += len(r)
    sst = yty - sy ** 2 / n
    def fit(cols):
        A = XtX[np.ix_(cols, cols)]; b = np.linalg.solve(A, Xty[cols])
        return 1 - (yty - b @ Xty[cols]) / sst, b
    c = np.arange(1, L + 1); i = np.arange(L + 1, 2 * L + 1)
    r2c, bc = fit(np.r_[0, c]); r2i, _ = fit(np.r_[0, i]); r2b, bb = fit(np.r_[0, c, i])
    return dict(n_reg=n, R2_cme=r2c, R2_index=r2i, R2_both=r2b,
                cme_survives=(r2b - r2i) / r2c if r2c > 0 else np.nan,
                index_survives=(r2b - r2c) / r2i if r2i > 0 else np.nan,
                cme_sum_alone=bc[1:].sum(), cme_sum_with_index=bb[1:L + 1].sum(), index_sum_with_cme=bb[L + 1:].sum(),
                cme_sum_1s_with_index=bb[1:min(11, L + 1)].sum())


def run_open(day, root, ix):
    sym = PAIRS[root]; d0 = pd.Timestamp(day, tz="UTC").value // 1_000_000; d1 = d0 + 86_400_000
    if (root, day) in nm.LISTING_DAY:
        return None
    nat = nm.load_native(sym, day)
    if nat is None:
        return None
    grid = np.arange(d0, d1, STEP); cl = nm.closed(grid)
    have = (grid >= max(nat["t"][0], ix["t"][0])) & (grid <= min(nat["t"][-1], ix["t"][-1])) & ~cl
    if have.sum() * STEP < 3.6e6:
        return None
    cme = nm.load_cme(root, day, d0, d1)
    if cme is None:
        return None
    B, _ = nm.asof(nat["t"], nat["bid"], grid); A, _ = nm.asof(nat["t"], nat["ask"], grid)
    C, _ = nm.asof(cme["t"], cme["mid"], grid); I, _ = nm.asof(ix["t"], ix["idx"], grid)
    lp, lc, li = np.log((B + A) / 2), np.log(C), np.log(I)
    for v in (lp, lc, li):
        v[~have] = np.nan
    # index updates on CME-open time
    tt = ix["t"]; ch = np.r_[False, np.diff(ix["idx"]) != 0]; op = ~nm.closed(tt) & (tt >= d0) & (tt < d1)
    tch = tt[ch & op]
    out = dict(day=day, asset=root, open_hours=have.sum() * STEP / 3.6e6,
               index_changes_per_min=len(tch) / (have.sum() * STEP / 60_000),
               index_gap_med_ms=np.median(np.diff(tch)) if len(tch) > 2 else np.nan,
               ticker_gap_med_ms=np.median(np.diff(tt[op])) if op.sum() > 2 else np.nan)
    out["LLR_cme_index"], out["peak_cme_index_ms"], out["rho_cme_index"] = nm.xcorr(lc, li)
    out["LLR_perp_index"], out["peak_perp_index_ms"], out["rho_perp_index"] = nm.xcorr(lp, li)
    out["LLR_cme_perp"], out["peak_cme_perp_ms"], out["rho_cme_perp"] = nm.xcorr(lc, lp)
    # does the perp move before its own index?  share of index moves whose sign the perp already showed 1 s earlier
    rp, rc, ri = np.diff(lp), np.diff(lc), np.diff(li)
    ok = np.isfinite(rp) & np.isfinite(rc) & np.isfinite(ri)
    okl = ok.copy()
    for j in range(1, L + 1):
        okl[j:] &= ok[:-j]
    okl[:L] = False
    rp0, rc0, ri0 = (np.where(ok, v, 0.0) * 1e4 for v in (rp, rc, ri))
    out.update(lagged_r2(rp0, rc0, ri0, okl))
    # same regression on a 1 s grid (the index's own cadence): 1 s returns, 5 lags
    l1p, l1c, l1i = lp[9::10], lc[9::10], li[9::10]
    q = [np.diff(v) for v in (l1p, l1c, l1i)]; ok1 = np.isfinite(q[0]) & np.isfinite(q[1]) & np.isfinite(q[2])
    ok1l = ok1.copy()
    for j in range(1, 6):
        ok1l[j:] &= ok1[:-j]
    ok1l[:5] = False
    r1 = lagged_r2(*(np.where(ok1, v, 0.0) * 1e4 for v in q), ok1l, L=5)
    out.update({f"{k}_1s": v for k, v in r1.items() if k in ("n_reg", "R2_cme", "R2_index", "R2_both", "cme_survives", "index_survives", "cme_sum_with_index")})
    # same regression on a 1 s grid (the index's own cadence): 1 s returns, 5 lags
    l1p, l1c, l1i = lp[9::10], lc[9::10], li[9::10]
    q = [np.diff(v) for v in (l1p, l1c, l1i)]; ok1 = np.isfinite(q[0]) & np.isfinite(q[1]) & np.isfinite(q[2])
    ok1l = ok1.copy()
    for j in range(1, 6):
        ok1l[j:] &= ok1[:-j]
    ok1l[:5] = False
    r1 = lagged_r2(*(np.where(ok1, v, 0.0) * 1e4 for v in q), ok1l, L=5)
    out.update({f"{k}_1s": v for k, v in r1.items() if k in ("n_reg", "R2_cme", "R2_index", "R2_both", "cme_survives", "index_survives", "cme_sum_with_index")})
    # basis to the index: perp - index (bps) and how much of it CME explains
    dev = (lp - li) * 1e4
    out["perp_minus_index_med_bps"] = np.nanmedian(dev); out["perp_minus_index_iqr_bps"] = np.subtract(*np.nanpercentile(dev, [75, 25]))
    cdev = (lc - li) * 1e4
    g = np.isfinite(dev) & np.isfinite(cdev)
    out["corr_dev_cmedev"] = np.corrcoef(dev[g], cdev[g])[0, 1] if g.sum() > 1000 else np.nan
    return out


def run_closed(day, root, ix):
    t = ix["t"]; cl = nm.closed(t)
    if cl.sum() < 600:
        return None, []
    idx, mark, last = ix["idx"][cl], ix["mark"][cl], ix["last"][cl]
    prem = np.log(last / idx) * 1e4; mg = (mark / idx - 1) * 1e4
    seg = np.r_[0, np.cumsum(np.diff(t[cl]) > 120_000)]          # separate closed stretches in the file
    out = dict(day=day, asset=root, closed_hours=len(np.unique(t[cl] // 1000)) / 3600, closed_stretches=seg.max() + 1,
               index_changes=int((np.diff(idx)[np.diff(seg) == 0] != 0).sum()),
               prem_med_abs_bps=np.nanmedian(np.abs(prem)), prem_max_abs_bps=np.nanmax(np.abs(prem)),
               mark_gap_max_abs_bps=np.max(np.abs(mg)), mark_at_band_share=np.mean(np.abs(mg) >= 295),
               mark_minus_last_med_bps=np.nanmedian(np.abs(np.log(mark / last)) * 1e4),
               index_follows_last_bps=np.nanmedian(np.abs(np.log(idx / pd.Series(last).ffill().values)) * 1e4))
    # funding settlements: the rate shown just before each funding time, against the premium averaged since the previous one
    fund = []
    for ft in np.unique(ix["fts"]):
        if not (t[0] < ft <= t[-1] + 1000):
            continue
        k = np.flatnonzero(ix["fts"] == ft)
        if len(k) == 0:
            continue
        rate = ix["fr"][k[-1]]
        prev = ft - 4 * 3_600_000
        w = (t >= prev) & (t < ft)
        if w.sum() < 600:
            continue
        p = np.log(ix["last"][w] / ix["idx"][w]) * 1e4; p = p[np.isfinite(p)]
        fund.append(dict(day=day, asset=root, funding_utc=pd.Timestamp(ft, unit="ms").strftime("%Y-%m-%d %H:%M"),
                         closed_share=nm.closed(t[w]).mean(), rate_bps=rate * 1e4, prem_avg_bps=p.mean(),
                         index_flat=bool(np.all(np.diff(ix["idx"][w]) == 0)), coverage=w.sum() / 14400))
    return out, fund


def funding_regimes(switch="2026-05-09"):
    """Weekend RQ1 slope before and after Binance's 8 May 2026 change of the closed-market index (frozen -> follows
    the perpetual), pooled with market fixed effects, HC3; and the share of weekend funding payments that are zero."""
    import glob, statsmodels.api as sm
    sw = pd.Timestamp(switch, tz="UTC")
    d = pd.concat([pd.read_csv(f"results/q1_events_{r}.csv") for r in PAIRS if os.path.exists(f"results/q1_events_{r}.csv")],
                  ignore_index=True).dropna(subset=["cme_gap", "bin_ret"])
    d = d[d.kind == "weekend"].reset_index(drop=True)
    d["post"] = (pd.to_datetime(d.close_ts, utc=True) >= sw).astype(int); d["G"], d["R"] = d.cme_gap * 1e4, d.bin_ret * 1e4
    rows = []
    for p in (0, 1):
        x = d[d.post == p].reset_index(drop=True)
        m = sm.OLS(x.G, pd.DataFrame({"R": x.R}).join(pd.get_dummies(x.asset, dtype=float))).fit(cov_type="HC3")
        rows.append(dict(regime="after" if p else "before", n=len(x), beta=m.params["R"], se=m.bse["R"]))
    X = pd.DataFrame({"R": d.R, "R_after": d.R * d.post}).join(pd.get_dummies(d.asset, dtype=float))
    m = sm.OLS(d.G, X).fit(cov_type="HC3")
    rows.append(dict(regime="difference", n=len(d), beta=m.params["R_after"], se=m.bse["R_after"], p=m.pvalues["R_after"]))
    pd.DataFrame(rows).to_csv("results/q8_funding_regimes.csv", index=False); print(pd.DataFrame(rows).round(3))
    z = []
    for r, s in PAIRS.items():
        fs = sorted(glob.glob(f"data/binance/{s}/funding/*.csv"))
        if not fs:
            continue
        f = pd.concat(pd.read_csv(x) for x in fs); t = pd.to_datetime(f.calc_time, unit="ms", utc=True).dt.floor("min")
        wk = (t.dt.dayofweek == 5) | ((t.dt.dayofweek == 6) & (t.dt.hour <= 20))
        for p, k in (("before", wk & (t < sw)), ("after", wk & (t >= sw))):
            z.append(dict(asset=r, regime=p, payments=int(k.sum()), zero_share=(f.last_funding_rate[k.values] == 0).mean()))
    pd.DataFrame(z).to_csv("results/q8_funding_zero_share.csv", index=False)
    print(pd.DataFrame(z).pivot(index="asset", columns="regime", values="zero_share").round(2))


if __name__ == "__main__":
    args = sys.argv[1:]; assets = list(PAIRS); closed_only = "--closed-only" in args
    if "--regimes" in args:
        funding_regimes(); sys.exit()
    args = [a for a in args if a != "--closed-only"]
    if "--assets" in args:
        i = args.index("--assets"); assets = args[i + 1].split(","); args = args[:i] + args[i + 2:]
    for day in args:
        O, C, F = [], [], []
        for root in assets:
            ix = load_index(PAIRS[root], day)
            if ix is None:
                print(f"{day} {root}: no ticker file", flush=True); continue
            c, f = run_closed(day, root, ix)
            if c:
                C.append(c); F += f
                print(f"{day} {root} closed: " + ", ".join(f"{k}={nm.fmt(v)}" for k, v in c.items() if k not in ("day", "asset")), flush=True)
            if not closed_only:
                o = run_open(day, root, ix)
                if o:
                    O.append(o)
                    print(f"{day} {root} open: " + ", ".join(f"{k}={nm.fmt(v)}" for k, v in o.items() if k not in ("day", "asset")), flush=True)
        nm.save(O, "results/q8_index_open.csv"); nm.save(C, "results/q8_index_closed.csv")
        nm.save(F, "results/q8_index_funding.csv", keys=("day", "asset", "funding_utc"))
