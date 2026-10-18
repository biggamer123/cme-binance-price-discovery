"""Q3b: sub-minute shape of the post-reopen convergence (tick window 24 Apr - 5 May 2026).

CME     : Databento mbp-1 top of book -> mid (only bid < ask).
Binance : aggTrades -> trade-reconstructed quotes (last seller-initiated trade ~ bid, last buyer-initiated ~ ask),
          mid valid when both sides are < 10 s old; also the last trade price as a fallback.
Event   : each CME reopen (17:00 CT). t_open = first CME quote after the reopen with spread <= 2x its normal
          spread (median over reopen+30..60 min); the first few quotes after the open are often 50-300 bps wide.
          Events with a contract roll between the open and reopen+60 min are dropped.
Gap     : g(s) = log b(t_open + s) - log c(t_open + s) - basis,  g(0) uses Binance just BEFORE t_open.
          basis = median log(b/c) over [reopen+30, reopen+60] min ('post'), or the last 30 min before the
          CME close ('pre').
Output  : fraction of the gap still open at s = slope of g(s) on g(0); half-life in seconds; also
          Binance spread at the reopen and the CME spread at its first quote.
Run: python 12_q3b_reopen_ticks.py [DATA_DIR] [OUT_DIR]
"""
import glob, os, sys
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(OUT, exist_ok=True)
PAIRS = {"CL": "CLUSDT", "BZ": "BZUSDT", "NG": "NATGASUSDT",
         "GC": "XAUUSDT", "SI": "XAGUSDT", "PL": "XPTUSDT", "PA": "XPDUSDT", "HG": "COPPERUSDT"}
GROUP = {k: ("energy" if k in ("CL", "BZ", "NG") else "metals") for k in PAIRS}
S = np.r_[0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600]     # seconds after t_open
NS = 1_000_000_000
STALE = 10 * NS


def cme_quotes(root):
    pf = pq.ParquetFile(f"{DATA}/cme/{root}_mbp1.parquet")
    ts, mid, spr, inst = [], [], [], []
    for b in pf.iter_batches(batch_size=2_000_000, columns=["ts_event", "instrument_id", "bid_px_00", "ask_px_00"]):
        d = b.to_pandas()
        t = d.ts_event.astype("int64").values
        h = (t // (3600 * NS)) % 24
        keep = (h >= 20) & (d.bid_px_00.values > 0) & (d.ask_px_00.values > d.bid_px_00.values)
        d = d[keep]
        ts.append(t[keep]); mid.append(((d.bid_px_00 + d.ask_px_00) / 2).values)
        spr.append(((d.ask_px_00 - d.bid_px_00) / ((d.bid_px_00 + d.ask_px_00) / 2)).values)
        inst.append(d.instrument_id.values)
    t, m, s, i = map(np.concatenate, (ts, mid, spr, inst))
    o = np.argsort(t, kind="stable")
    return t[o], m[o], s[o], i[o]


def binance_trades(sym):
    x = pd.concat([pd.read_csv(f, usecols=["price", "transact_time", "is_buyer_maker"])
                   for f in sorted(glob.glob(f"{DATA}/binance/{sym}/aggTrades/*.csv"))])
    t = x.transact_time.values.astype("int64")
    t = np.where(t > 1e14, t * 1000, t * 1_000_000)                                        # -> ns
    o = np.argsort(t, kind="stable")
    t, p = t[o], x.price.values[o]
    sell = x.is_buyer_maker.astype(str).str.lower().eq("true").values[o]
    return t, p, sell


def asof(t, v, q):
    i = np.searchsorted(t, q, side="right") - 1
    ok = i >= 0
    out = np.full(len(q), np.nan); age = np.full(len(q), np.inf)
    out[ok] = v[i[ok]]; age[ok] = q[ok] - t[i[ok]]
    return out, age


def binance_mid(bt, bp, bsell, q):
    bid, ab = asof(bt[bsell], bp[bsell], q)
    ask, aa = asof(bt[~bsell], bp[~bsell], q)
    mid = np.where((ab < STALE) & (aa < STALE), (bid + ask) / 2, np.nan)
    last, _ = asof(bt, bp, q)
    spread = np.where((ab < STALE) & (aa < STALE), (ask - bid) / ((ask + bid) / 2), np.nan)
    return mid, last, spread


def run_asset(root, sym):
    ct, cm, cs, ci = cme_quotes(root)
    bt, bp, bsell = binance_trades(sym)
    days = pd.date_range("2026-04-24", "2026-05-05", freq="D", tz="UTC")
    rows = []
    for d in days:
        reopen = d + pd.Timedelta(hours=22)                                                # 17:00 CDT
        dow = reopen.tz_convert("America/Chicago").dayofweek
        if dow in (4, 5): continue                                                         # Fri/Sat: no reopen
        kind = "weekend" if dow == 6 else "daily_break"
        close = reopen - (pd.Timedelta(hours=49) if kind == "weekend" else pd.Timedelta(hours=1))
        r0 = reopen.value
        # 'normal' CME spread = median over [reopen+30, reopen+60] min. Clock starts at t_open = first CME quote
        # after the reopen whose spread is <= 2x normal (the very first quotes after the open are often
        # one-sided-thin with 50-300 bps spreads, so their mid is not a usable price).
        w = (ct >= r0 + 1800 * NS) & (ct < r0 + 3600 * NS)
        if w.sum() < 100: continue
        normal = np.median(cs[w])
        cand = np.where((ct >= r0) & (ct <= r0 + 600 * NS) & (cs <= 2 * normal))[0]
        if not len(cand): continue
        j = cand[0]
        j_first = np.searchsorted(ct, r0)
        t_open = ct[j]
        # contract roll between the opening quote and the post-reopen basis window -> skip
        if ci[j] != ci[np.searchsorted(ct, r0 + 3600 * NS) - 1]: continue
        # pre-open quotes on CME (valid, uncrossed) in the hour before the reopen?
        n_preopen = int(((ct > r0 - 3600 * NS) & (ct < r0)).sum())
        # baselines
        g_post = r0 + np.arange(1800, 3600) * NS
        g_pre = close.value - np.arange(1, 1800) * NS
        def base(grid):
            c, ca = asof(ct, cm, grid); mid, last, _ = binance_mid(bt, bp, bsell, grid)
            b = np.where(np.isfinite(mid), mid, last)
            ok = (ca < 60 * NS) & np.isfinite(b)
            return np.nanmedian(np.log(b[ok]) - np.log(c[ok])) if ok.sum() > 100 else np.nan
        bpost, bpre = base(g_post), base(g_pre)
        q = t_open + (S * NS).astype(np.int64)
        c, _ = asof(ct, cm, q)
        mid, last, bspr = binance_mid(bt, bp, bsell, q)
        mid0, last0, bspr0 = binance_mid(bt, bp, bsell, np.array([t_open - 1]))
        b = np.where(np.isfinite(mid), mid, last); b[0] = mid0[0] if np.isfinite(mid0[0]) else last0[0]
        # Binance trades in the first 60 s (how much activity there is to trade against)
        n_bin_60s = int(((bt >= t_open) & (bt < t_open + 60 * NS)).sum())
        rec = dict(asset=root, group=GROUP[root], kind=kind, reopen=reopen,
                   first_quote_delay_ms=(ct[j_first] - r0) / 1e6, normal_quote_delay_ms=(t_open - r0) / 1e6,
                   cme_preopen_quotes=n_preopen, base_pre=bpre, base_post=bpost, cme_normal_spread_bps=normal * 1e4,
                   cme_spread_first_bps=cs[j_first] * 1e4, cme_spread_open_bps=cs[j] * 1e4,
                   bin_spread_open_bps=bspr0[0] * 1e4, bin_trades_first60s=n_bin_60s,
                   bin_mid_ok_at_open=bool(np.isfinite(mid0[0])))
        for s, x, y in zip(S, np.log(b), np.log(c)):
            rec[f"b{s:g}"] = x; rec[f"c{s:g}"] = y
        rows.append(rec)
    return pd.DataFrame(rows)


def gap(d, s, base="post"):
    return d[f"b{s:g}"] - d[f"c{s:g}"] - d[f"base_{base}"]


def curves(ev, base="post"):
    out, summ = [], []
    groups = [((g, k), d) for (g, k), d in ev.groupby(["group", "kind"])] + \
             [(("ALL", k), d) for k, d in ev.groupby("kind")] + \
             [((a, "all_reopens"), d) for a, d in ev.groupby("asset")]
    for (grp, kind), d in groups:
        d = d.dropna(subset=[f"b{s:g}" for s in S] + [f"c{s:g}" for s in S] + [f"base_{base}"])
        if len(d) < 4: continue
        g0 = gap(d, 0, base)
        betas = []
        for s in S:
            gs = gap(d, s, base)
            m = sm.OLS(gs, sm.add_constant(g0)).fit(cov_type="HC1")
            bshare = sm.OLS(-(d[f"b{s:g}"] - d["b0"]), sm.add_constant(g0)).fit().params.iloc[1]
            cshare = sm.OLS(d[f"c{s:g}"] - d["c0"], sm.add_constant(g0)).fit().params.iloc[1]
            out.append(dict(group=grp, kind=kind, n=len(d), s=s, frac_open=m.params.iloc[1], se=m.bse.iloc[1],
                            binance_share=bshare, cme_share=cshare, mean_abs_gap_bps=gs.abs().mean() * 1e4))
            betas.append(m.params.iloc[1])
        hl = np.nan
        for j in range(1, len(S)):
            if betas[j] <= 0.5:
                b0, b1 = betas[j - 1], betas[j]
                hl = S[j - 1] + (b0 - 0.5) / (b0 - b1) * (S[j] - S[j - 1]) if b0 > b1 else S[j]
                break
        # per-event time to half (events with |g0| >= 5 bps)
        t_half = []
        for _, r in d.iterrows():
            g0r = r[f"b0"] - r["c0"] - r[f"base_{base}"]
            if abs(g0r) < 5e-4: continue
            for s in S[1:]:
                gs = r[f"b{s:g}"] - r[f"c{s:g}"] - r[f"base_{base}"]
                if np.sign(gs) != np.sign(g0r) or abs(gs) <= 0.5 * abs(g0r):
                    t_half.append(s); break
            else:
                t_half.append(np.inf)
        summ.append(dict(group=grp, kind=kind, n=len(d), mean_abs_gap0_bps=g0.abs().mean() * 1e4,
                         pooled_half_life_s=hl, events_ge5bps=len(t_half),
                         median_event_half_life_s=np.median(t_half) if t_half else np.nan,
                         **{f"open_{s:g}s": b for s, b in zip(S, betas) if s in (1, 5, 10, 30, 60, 300)}))
    return pd.DataFrame(summ), pd.DataFrame(out)


def main():
    ev = pd.concat([run_asset(r, s) for r, s in PAIRS.items()], ignore_index=True)
    ev.to_csv(f"{OUT}/q3b_tick_events.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    info = ev[["asset", "kind", "reopen", "first_quote_delay_ms", "normal_quote_delay_ms", "cme_normal_spread_bps",
               "cme_spread_first_bps", "cme_spread_open_bps", "bin_spread_open_bps", "bin_trades_first60s"]].copy()
    info["gap0_bps"] = (ev["b0"] - ev["c0"] - ev["base_post"]) * 1e4
    for s in (1, 5, 30, 60, 300):
        info[f"gap{s}s_bps"] = (ev[f"b{s}"] - ev[f"c{s}"] - ev["base_post"]) * 1e4
    print(info.round(1).to_string(index=False))
    for base in ("post", "pre"):
        sm_, cv = curves(ev, base)
        sm_.to_csv(f"{OUT}/q3b_tick_halflife_{base}.csv", index=False)
        cv.to_csv(f"{OUT}/q3b_tick_curves_{base}.csv", index=False)
        print(f"\n== {base} basis\n", sm_.round(2).to_string(index=False))
    print("\n== curve (post)\n", pd.read_csv(f"{OUT}/q3b_tick_curves_post.csv").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
