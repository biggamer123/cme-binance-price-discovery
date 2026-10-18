"""CME top-of-book (mbp-1: every best bid/ask change) for any extra UTC days you have Binance native quotes for.
Usage (in Terminal, in this folder):
    python 21_databento_mbp1_days.py 2026-08-19 2026-08-20
Asks for the Databento API key (hidden), prints the cost per market and the total, and downloads only after
you type YES.  QUOTE=1 python 21_databento_mbp1_days.py <days>   -> prices only, never downloads.
ROOTS=GC,SI,PL,PA,HG,CL,BZ,NG python ...                           -> a subset of markets (e.g. without ES). Files: data/cme/days/<ROOT>_mbp1_<date>.parquet (time, contract id, best bid/ask and their sizes)."""
import os, sys, getpass, time
import pandas as pd
import databento as db

ROOTS = os.environ.get("ROOTS", "GC,SI,PL,PA,HG,CL,BZ,NG,ES").split(",")   # add BTC only if you also want Bitcoin
days = sys.argv[1:]
if not days:
    sys.exit("give one or more UTC dates, e.g. python 21_databento_mbp1_days.py 2026-08-19 2026-08-20")
key = os.environ.get("DATABENTO_API_KEY") or getpass.getpass("Paste the Databento API key (hidden) and press Enter: ").strip()
if os.environ.get("QUOTE"):
    BUDGET = float("inf")
elif os.environ.get("BUDGET"):
    BUDGET = float(os.environ["BUDGET"])
else:   # Databento's API does not report the account balance, so ask for it (portal -> Billing) and never spend more
    BUDGET = float(input("Your remaining Databento credit in $ (from the Billing page), e.g. 21.40: ").strip().lstrip("$"))
print(f"Spending cap: ${BUDGET:.2f}")
c = db.Historical(key)


def retry(fn, *a, **k):
    """Databento sometimes answers 504/502 (gateway timeout); wait and try again instead of crashing"""
    for i in range(6):
        try:
            return fn(*a, **k)
        except db.BentoServerError as e:
            print(f"  server busy ({str(e).splitlines()[0][:60]}), retrying in {10 * (i + 1)} s ...", flush=True)
            time.sleep(10 * (i + 1))
    return fn(*a, **k)
os.makedirs("data/cme/days", exist_ok=True)
todo, total = [], 0.0
for day in days:
    a = pd.Timestamp(day).strftime("%Y-%m-%d"); b = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for r in ROOTS:
        out = f"data/cme/days/{r}_mbp1_{a}.parquet"
        if os.path.exists(out):
            print("already have", out); continue
        cost = retry(c.metadata.get_cost, dataset="GLBX.MDP3", symbols=[f"{r}.v.0"], stype_in="continuous",
                                   schema="mbp-1", start=a, end=b)
        if total + cost > BUDGET:
            print(f"{a} {r:3s} ${cost:6.2f}  SKIPPED (budget ${BUDGET:.0f})"); continue
        total += cost; todo.append((r, a, b, out)); print(f"{a} {r:3s} ${cost:6.2f}")
print(f"TOTAL ${total:.2f}")
if os.environ.get("QUOTE"):
    sys.exit("Quote only - nothing downloaded.")
if not todo or input("Type YES to download: ").strip() != "YES":
    sys.exit("Nothing downloaded.")
for r, a, b, out in todo:
    print(f"downloading {r} {a} ...", flush=True)
    d = retry(c.timeseries.get_range, dataset="GLBX.MDP3", symbols=[f"{r}.v.0"], stype_in="continuous",
                               schema="mbp-1", start=a, end=b).to_df()
    d[["ts_event", "instrument_id", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]].to_parquet(out)
    print(f"  saved {out} ({len(d):,} quote updates)")
print("DONE")
