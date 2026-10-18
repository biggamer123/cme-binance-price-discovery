"""Native Binance quotes (Tardis book_ticker, free first-day-of-month files) around CME closures.
 A) Binance spread and top-of-book depth by CME state: open (normal), closed (daily break / weekend),
    first minutes after the CME reopen (17:01-17:06 CT).
 B) Stale-price check of the reopen trade signal: basis-adjusted gap 1 min after the reopen computed with
    the last Binance TRADE (as in the main analysis) vs the true Binance MID at the same instant, and the
    15-min hedged convergence with each.  CME side and basis exactly as in 11_q3_reopen_halflife.py.
Outputs: results/q6_spread_state.csv, results/q6_spread_state_days.csv, results/q6_gap_check.csv"""
import glob, importlib.util, os, numpy as np, pandas as pd
PAIRS = {"GC": "XAUUSDT", "SI": "XAGUSDT", "PL": "XPTUSDT", "PA": "XPDUSDT", "HG": "COPPERUSDT",
         "CL": "CLUSDT", "BZ": "BZUSDT", "NG": "NATGASUSDT", "ES": "SPYUSDT", "BTC": "BTCUSDT"}
LISTING_DAY = {("CL", "2026-04-01"), ("BZ", "2026-04-01"), ("NG", "2026-04-01")}

def bbo(path):
    ts, b, a, qb, qa = [], [], [], [], []
    for x in pd.read_csv(path, chunksize=4_000_000, usecols=["timestamp", "bid_price", "ask_price", "bid_amount", "ask_amount"]):
        t = x.timestamp.values // 1000; k = np.r_[t[1:] != t[:-1], True]
        ts.append(t[k]); b.append(x.bid_price.values[k]); a.append(x.ask_price.values[k])
        qb.append(x.bid_amount.values[k]); qa.append(x.ask_amount.values[k])
    t, b, a, qb, qa = map(np.concatenate, (ts, b, a, qb, qa))
    ok = (b > 0) & (a > b); return t[ok], b[ok], a[ok], qb[ok], qa[ok]

def asof(t, v, q):
    i = np.searchsorted(t, q, side="right") - 1
    return np.where(i >= 0, v[np.clip(i, 0, None)], np.nan)

def state(grid_ms):
    ct = pd.to_datetime(grid_ms, unit="ms", utc=True).tz_convert("America/Chicago")
    h, dow, mi = ct.hour.values, ct.dayofweek.values, ct.minute.values
    weekend = ((dow == 4) & (h >= 16)) | (dow == 5) | ((dow == 6) & (h < 17))
    brk = (h == 16) & ~weekend
    post = (h == 17) & (mi >= 1) & (mi < 6) & (dow != 5)
    early = (h == 17) & (mi < 30) & (dow != 5)
    s = np.where(weekend, "closed_weekend", np.where(brk, "closed_break", np.where(post, "post_reopen_1_6m",
                 np.where(early, "post_reopen_other", "open_normal"))))
    return s

import sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "A"
os.makedirs("results/q6_cache", exist_ok=True)
rows = []
for f in sorted(glob.glob("data/tardis/book_ticker/*/*.csv.gz")) if MODE == "A" else []:
    day, sym = f.split("/")[-2], os.path.basename(f).split(".")[0]
    root = [k for k, v in PAIRS.items() if v == sym][0]
    if (root, day) in LISTING_DAY: continue
    cache = f"results/q6_cache/{day}_{root}.csv"
    if os.path.exists(cache): rows += pd.read_csv(cache).to_dict("records"); continue
    n0 = len(rows)
    t, b, a, qb, qa = bbo(f)
    d0 = pd.Timestamp(day, tz="UTC").value // 10**6
    g = np.arange(d0, d0 + 86_400_000, 1000)
    B, A, QB, QA = (asof(t, v, g) for v in (b, a, qb, qa))
    m = (B + A) / 2; spr = (A - B) / m * 1e4; dep = (QB * B + QA * A) / 2
    st = state(g)
    for s in np.unique(st):
        k = (st == s) & np.isfinite(spr)
        if k.sum() < 60: continue
        rows.append(dict(asset=root, day=day, state=s, seconds=int(k.sum()), spread_bps_med=np.median(spr[k]),
                         spread_bps_mean=np.mean(spr[k]), depth_usd_med=np.median(dep[k])))
    pd.DataFrame(rows[n0:]).to_csv(cache, index=False); print(day, root, flush=True)
if MODE == "B": rows = pd.read_csv("results/q6_spread_state_days.csv").to_dict("records")
sd = pd.DataFrame(rows); sd.to_csv("results/q6_spread_state_days.csv", index=False)
# pooled: median over days of the daily medians
agg = sd.groupby(["asset", "state"]).agg(days=("day", "nunique"), spread_bps=("spread_bps_med", "median"),
                                         depth_usd=("depth_usd_med", "median")).reset_index()
agg.to_csv("results/q6_spread_state.csv", index=False)
print(agg.pivot(index="asset", columns="state", values="spread_bps").round(2).to_string())
print(agg.pivot(index="asset", columns="state", values="depth_usd").round(0).to_string())

# ---------- B) stale-price check at reopen entry
if MODE == "A": sys.exit()
spec = importlib.util.spec_from_file_location("q3", "11_q3_reopen_halflife.py"); q3 = importlib.util.module_from_spec(spec)
sys.argv = ["x"]; spec.loader.exec_module(q3)
MIN = 60_000
out = []
for root in ["GC", "SI", "PL", "PA", "HG", "CL", "BZ", "NG"]:
    sym = PAIRS[root]; ev = q3.events(root, sym)
    if ev.empty: continue
    for _, e in ev.iterrows():
        day = e.reopen.strftime("%Y-%m-%d"); f = f"data/tardis/book_ticker/{day}/{sym}.csv.gz"
        if not os.path.exists(f) or (root, day) in LISTING_DAY: continue
        t, b, a, _, _ = bbo(f); t1 = e.reopen.value // 10**6
        r = dict(asset=root, kind=e.kind, reopen=e.reopen)
        for k in (0, 1, 16):
            q = t1 + k * MIN; mid = (asof(t, b, np.array([q])) + asof(t, a, np.array([q])))[0] / 2
            r[f"g{k}_last"] = (e[f"b{k}"] - e[f"c{k}"] - e.base_pre) * 1e4
            r[f"g{k}_true"] = (np.log(mid) - e[f"c{k}"] - e.base_pre) * 1e4
            r[f"spread{k}_bps"] = ((asof(t, a, np.array([q])) - asof(t, b, np.array([q]))) / mid * 1e4)[0]
        r["pnl_last"] = -np.sign(r["g1_last"]) * (r["g16_last"] - r["g1_last"])
        r["pnl_true"] = -np.sign(r["g1_true"]) * (r["g16_true"] - r["g1_true"])
        out.append(r)
gc = pd.DataFrame(out); gc.to_csv("results/q6_gap_check.csv", index=False)
pd.set_option("display.width", 200)
print(gc.round(1).to_string(index=False))
