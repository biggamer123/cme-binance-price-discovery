"""Robustness for Q2: full CME top-of-book (mbp-1 = every best bid/ask change, not just at trades)
for the assets where Binance appeared to lead (SI, BZ, PA, PL, BTC), 24 Apr - 6 May 2026. Quotes first, asks YES before spending credit.
Keeps only the columns needed so files stay small."""
import os, sys, getpass, databento as db
key = os.environ.get("DATABENTO_API_KEY") or getpass.getpass("Paste your Databento API key (hidden) and press Enter: ").strip()
c = db.Historical(key)
ROOTS = ["SI", "BZ", "PA", "PL", "BTC", "HG", "NG", "GC", "CL", "ES"]   # priority order; ES last (largest)
BUDGET = 24.0
A, B = "2026-04-24", "2026-05-06"
os.makedirs("data/cme", exist_ok=True)
todo, total = [], 0.0
for r in ROOTS:
    out = f"data/cme/{r}_mbp1.parquet"
    if os.path.exists(out): print("already have", out); continue
    cost = c.metadata.get_cost(dataset="GLBX.MDP3", symbols=[f"{r}.v.0"], stype_in="continuous",
                               schema="mbp-1", start=A, end=B)
    if total + cost <= BUDGET:
        total += cost; todo.append((r, out)); print(f"mbp-1 {r:4s} ${cost:6.2f}  -> will download")
    else:
        print(f"mbp-1 {r:4s} ${cost:6.2f}  -> SKIPPED (would exceed ${BUDGET})")
print(f"TOTAL ${total:.2f}  (you have about $24.86 credit left)")
if not todo: sys.exit("Nothing affordable to download. Tell Claude.")
if input("Type YES to download: ").strip() != "YES":
    sys.exit("Cancelled.")
for r, out in todo:
    print(f"downloading mbp-1 {r} ...", flush=True)
    d = c.timeseries.get_range(dataset="GLBX.MDP3", symbols=[f"{r}.v.0"], stype_in="continuous",
                               schema="mbp-1", start=A, end=B).to_df()
    d[["ts_event", "instrument_id", "bid_px_00", "ask_px_00"]].to_parquet(out)
    print(f"  saved {out}  ({len(d):,} quote updates)")
print("MBP-1 DONE")
