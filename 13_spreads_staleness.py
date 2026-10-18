"""Execution-cost inputs for the feasibility test, from the tick window (24 Apr - 5 May 2026), CME open hours.
 CME: quoted spread from mbp-1 (median, bps), overall and 1-6 min after each reopen.
 Binance: (a) staleness = age of the older side of the trade-reconstructed quote, on a 1 s grid;
          (b) effective-spread proxy = gap between the last buyer- and last seller-initiated trade prices
              when both are < 1 s old (median, bps). Real Binance quotes are not available.
 Also: price level (for converting CME per-contract fees into bps)."""
import glob, numpy as np, pandas as pd, pyarrow.parquet as pq
PAIRS = {"GC": "XAUUSDT", "SI": "XAGUSDT", "PL": "XPTUSDT", "PA": "XPDUSDT", "HG": "COPPERUSDT",
         "CL": "CLUSDT", "BZ": "BZUSDT", "NG": "NATGASUSDT", "ES": "SPYUSDT", "BTC": "BTCUSDT"}
def ct_open_mask(t_ms):
    ct = pd.to_datetime(t_ms, unit="ms", utc=True).tz_convert("America/Chicago")
    closed = (ct.hour == 16) | ((ct.dayofweek == 4) & (ct.hour >= 16)) | (ct.dayofweek == 5) | ((ct.dayofweek == 6) & (ct.hour < 17))
    post = (ct.hour == 17) & (ct.minute >= 1) & (ct.minute < 6)                    # 1-6 min after reopen
    return ~np.asarray(closed), np.asarray(post)
rows = []
for r, sym in PAIRS.items():
    sp_all, sp_post, px = [], [], []
    for b in pq.ParquetFile(f"data/cme/{r}_mbp1.parquet").iter_batches(batch_size=2_000_000,
            columns=["ts_event", "bid_px_00", "ask_px_00"]):
        d = b.to_pandas(); d = d[(d.bid_px_00 > 0) & (d.ask_px_00 > d.bid_px_00)]
        t = pd.to_datetime(d.ts_event, utc=True).astype("int64").values // 1_000_000
        m = (d.bid_px_00 + d.ask_px_00).values / 2; s = (d.ask_px_00 - d.bid_px_00).values / m * 1e4
        op, po = ct_open_mask(t); sp_all.append(s[op][::20]); sp_post.append(s[po]); px.append(m[op][::200])
    x = pd.concat(pd.read_csv(f, usecols=["agg_trade_id", "price", "transact_time", "is_buyer_maker"])
                  for f in sorted(glob.glob(f"data/binance/{sym}/aggTrades/*.csv"))).sort_values("agg_trade_id")
    t = x.transact_time.values.astype("int64"); t = t // 1000 if t[0] > 1e14 else t
    sell = x.is_buyer_maker.astype(str).str.lower().eq("true").values; p = x.price.values
    tb, pb = t[sell], p[sell]; ta, pa = t[~sell], p[~sell]
    grid = np.arange(t[0], t[-1], 1000); op, _ = ct_open_mask(grid); grid = grid[op]
    ib = np.searchsorted(tb, grid, "right") - 1; ia = np.searchsorted(ta, grid, "right") - 1
    ok = (ib >= 0) & (ia >= 0); g = grid[ok]; ib, ia = ib[ok], ia[ok]
    age = np.maximum(g - tb[ib], g - ta[ia]) / 1000
    fresh = (g - tb[ib] < 1000) & (g - ta[ia] < 1000)
    eff = np.abs(pa[ia] - pb[ib])[fresh] / ((pa[ia] + pb[ib])[fresh] / 2) * 1e4
    rows.append(dict(asset=r, price_level=np.median(np.concatenate(px)),
                     cme_spread_bps=np.median(np.concatenate(sp_all)), cme_spread_post_reopen_bps=np.median(np.concatenate(sp_post)) if sum(map(len, sp_post)) else np.nan,
                     bin_quote_age_median_s=np.median(age), bin_quote_age_p90_s=np.percentile(age, 90),
                     bin_share_fresh_2s=np.mean(age < 2), bin_eff_spread_bps=np.median(eff) if len(eff) else np.nan))
    print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False, header=len(rows) == 1), flush=True)
pd.DataFrame(rows).to_csv("results/costs_staleness.csv", index=False)
