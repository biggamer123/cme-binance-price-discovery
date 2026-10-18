"""Download CME data from Databento within the $125 free credit.
Bars start when each Binance perp was listed (no point paying for CME bars before that).
Quotes the total first and ABORTS if it exceeds BUDGET. Re-runnable: skips files already saved."""
import os, sys, databento as db
BUDGET = 120.0
import getpass
key = os.environ.get("DATABENTO_API_KEY") or getpass.getpass("Paste your Databento API key (hidden) and press Enter: ").strip()
c = db.Historical(key)
END = "2026-10-01"
BARS = {  # CME root: start date = Binance perp listing date (or 8 Jan)
    "GC": "2026-01-08", "SI": "2026-01-08", "PL": "2026-01-30", "PA": "2026-01-30",
    "HG": "2026-03-06", "CL": "2026-04-01", "BZ": "2026-04-01", "NG": "2026-04-01",
    "ES": "2026-04-06", "BTC": "2026-01-08"}           # NQ dropped: too costly, ES covers US equities
TICKS = ["GC", "SI", "PL", "PA", "HG", "CL", "BZ", "NG", "ES", "BTC"]
JOBS = [("ohlcv-1s", r, a, END, f"data/cme/{r}_ohlcv1s.parquet") for r, a in BARS.items()] + \
       [("tbbo", r, "2026-04-24", "2026-05-06", f"data/cme/{r}_tbbo.parquet") for r in TICKS]
os.makedirs("data/cme", exist_ok=True)
todo, total = [], 0.0
for schema, r, a, b, out in JOBS:
    if os.path.exists(out): continue
    cost = c.metadata.get_cost(dataset="GLBX.MDP3", symbols=[f"{r}.v.0"], stype_in="continuous",
                               schema=schema, start=a, end=b)
    total += cost; todo.append((schema, r, a, b, out, cost))
    print(f"{schema:9s} {r:4s} {a}->{b} ${cost:7.2f}")
print(f"TOTAL to download: ${total:.2f}  (budget ${BUDGET})")
if total > BUDGET:
    sys.exit("Over budget - nothing downloaded. Tell Claude.")
if input("Type YES to download (this uses your Databento credit): ").strip() != "YES":
    sys.exit("Cancelled.")
for schema, r, a, b, out, cost in todo:
    print(f"downloading {schema} {r} ...", flush=True)
    d = c.timeseries.get_range(dataset="GLBX.MDP3", symbols=[f"{r}.v.0"], stype_in="continuous",
                               schema=schema, start=a, end=b)
    d.to_df().to_parquet(out)
    print(f"  saved {out}")
print("ALL CME DATA DOWNLOADED")
