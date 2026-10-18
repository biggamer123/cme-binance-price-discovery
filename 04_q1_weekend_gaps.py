"""Q1: does Binance price the CME closure? (weekend gaps + daily breaks)
Inputs
  data/cme/{ROOT}_ohlcv1s.parquet   Databento GLBX.MDP3 ohlcv-1s, continuous .v.0 (to_df().to_parquet)
  data/binance/{SYM}/klines1m/*.csv Binance perp 1m klines (see README, Getting the data)
Outputs
  results/q1_events_{ROOT}.csv, results/q1_summary.csv, results/q1_scatter_{ROOT}.png
Method
  Closures = gaps between consecutive CME bars > 30 min. Weekend if > 24h, else daily break.
  CME gap     = log(first CME price after reopen / last CME price before close)
  Binance ret = log(Binance price at reopen / Binance price at close)   (same timestamps)
  Regress CME gap on Binance ret: slope ~1 => Binance already priced the closure.
  Closures where the CME contract (instrument_id) changes are roll events and are dropped."""
import glob, os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PAIRS = {"GC": "XAUUSDT", "SI": "XAGUSDT", "PL": "XPTUSDT", "PA": "XPDUSDT", "HG": "COPPERUSDT",
         "CL": "CLUSDT", "BZ": "BZUSDT", "NG": "NATGASUSDT", "ES": "SPYUSDT", "NQ": "QQQUSDT", "BTC": "BTCUSDT"}
os.makedirs("results", exist_ok=True)
# AVG_MIN=5 -> robustness: Binance price = average of the last 5 one-minute closes (less noise in thin markets)
AVG_MIN = int(os.environ.get("AVG_MIN", "1")); SUF = "" if AVG_MIN == 1 else f"_avg{AVG_MIN}"

def load_cme(root):
    df = pd.read_parquet(f"data/cme/{root}_ohlcv1s.parquet")
    if "ts_event" in df.columns: df = df.set_index("ts_event")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()[["open", "close", "instrument_id"]]

def load_binance(sym):
    files = sorted(glob.glob(f"data/binance/{sym}/klines1m/*.csv"))
    parts = []
    for f in files:
        d = pd.read_csv(f, header=None)
        if not str(d.iloc[0, 0]).isdigit(): d = d.iloc[1:]          # newer files have a header row
        d = d.iloc[:, [0, 1, 4]].astype(float); d.columns = ["t", "open", "close"]
        parts.append(d)
    b = pd.concat(parts).drop_duplicates("t")
    unit = "us" if b["t"].iloc[0] > 1e14 else "ms"
    b.index = pd.to_datetime(b["t"].astype("int64"), unit=unit, utc=True)
    return b.sort_index()

def price_at(b, ts):
    """Binance price at time ts: close of the last fully completed 1m bar before ts."""
    s = b["close"].loc[ts - pd.Timedelta(minutes=AVG_MIN): ts - pd.Timedelta(minutes=1)]
    return s.mean() if len(s) else np.nan

# Days Databento flags as degraded for GLBX.MDP3 (from the download warning); events touching them are dropped
DEGRADED = {"2026-01-31", "2026-03-15", "2026-03-16", "2026-03-21", "2026-04-10", "2026-05-24", "2026-08-29"}

def events(cme, b, root):
    """CME Globex closes 16:00 CT and reopens 17:00 CT (daily), Fri 16:00 -> Sun 17:00 CT (weekend).
    A closure = gap > 45 min between CME bars whose reopen falls at 17:00-17:15 Chicago time.
    This ignores intraday no-trade gaps in thin contracts."""
    t = cme.index.to_series()
    gap = t.diff()
    ct = cme.index.tz_convert("America/Chicago")
    is_reopen = (gap > pd.Timedelta(minutes=45)).values & (ct.hour == 17) & (ct.minute < 15)
    rows = []
    for i in np.where(is_reopen)[0]:
        t0, t1 = cme.index[i - 1], cme.index[i]
        if cme["instrument_id"].iloc[i] != cme["instrument_id"].iloc[i - 1]:
            continue                                                 # contract roll, skip
        if {t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d")} & DEGRADED:
            continue
        hrs = (t1 - t0).total_seconds() / 3600
        kind = "daily_break" if hrs < 3 else ("weekend" if t1.tz_convert("America/Chicago").dayofweek == 6
                                              and 40 < hrs < 60 else "holiday")
        p0, p1 = cme["close"].iloc[i - 1], cme["open"].iloc[i]
        b0, b1 = price_at(b, t0 + pd.Timedelta(seconds=1)), price_at(b, t1)
        # Binance move in the first 5 / 30 min after reopen (does Binance keep adjusting after CME opens?)
        b5, b30 = price_at(b, t1 + pd.Timedelta(minutes=5)), price_at(b, t1 + pd.Timedelta(minutes=30))
        c5 = cme["close"].loc[: t1 + pd.Timedelta(minutes=5)].iloc[-1]
        rows.append(dict(asset=root, close_ts=t0, reopen_ts=t1, hours=hrs, kind=kind,
                         cme_gap=np.log(p1 / p0), bin_ret=np.log(b1 / b0),
                         bin_post5=np.log(b5 / b1), bin_post30=np.log(b30 / b1),
                         cme_post5=np.log(c5 / p1)))
    ev = pd.DataFrame(rows).dropna()
    ev.to_csv(f"results/q1_events_{root}{SUF}.csv", index=False)
    return ev

def regress(ev):
    X = sm.add_constant(ev["bin_ret"])
    m = sm.OLS(ev["cme_gap"], X).fit(cov_type="HC1")
    hit = (np.sign(ev["cme_gap"]) == np.sign(ev["bin_ret"])).mean()
    return dict(n=len(ev), slope=m.params["bin_ret"], slope_se=m.bse["bin_ret"],
                alpha_bps=m.params["const"] * 1e4, r2=m.rsquared, hit_rate=hit,
                mean_abs_gap_bps=ev["cme_gap"].abs().mean() * 1e4,
                mean_abs_residual_bps=m.resid.abs().mean() * 1e4)

def main():
    summ = []
    for root, sym in PAIRS.items():
        if sym is None or not os.path.exists(f"data/cme/{root}_ohlcv1s.parquet"): continue
        ev = events(load_cme(root), load_binance(sym), root)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, kind in zip(ax, ["weekend", "daily_break"]):
            e = ev[ev.kind == kind]
            if len(e) < 5: continue
            r = regress(e); r.update(asset=root, binance=sym, kind=kind); summ.append(r)
            a.scatter(e.bin_ret * 1e4, e.cme_gap * 1e4, s=12)
            lim = np.nanmax(np.abs(np.r_[e.bin_ret, e.cme_gap])) * 1e4 * 1.1
            a.plot([-lim, lim], [-lim, lim], lw=0.8, ls="--")
            a.set(title=f"{root} {kind}: slope {r['slope']:.2f}, R² {r['r2']:.2f}, n={r['n']}",
                  xlabel=f"{sym} return during closure (bps)", ylabel="CME reopen gap (bps)")
        fig.tight_layout(); fig.savefig(f"results/q1_scatter_{root}{SUF}.png", dpi=150); plt.close(fig)
    out = pd.DataFrame(summ)
    out.to_csv(f"results/q1_summary{SUF}.csv", index=False)
    print(out.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
