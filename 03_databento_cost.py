"""Step 3: quote Databento cost BEFORE downloading anything. Needs DATABENTO_API_KEY env var.
Downloads nothing, costs nothing."""
import databento as db
c = db.Historical()
SYMS = ["GC.v.0", "SI.v.0", "PL.v.0", "PA.v.0", "HG.v.0", "CL.v.0", "BZ.v.0", "NG.v.0", "ES.v.0", "NQ.v.0", "BTC.v.0"]
jobs = [("ohlcv-1s", "2026-01-08", "2026-10-01"),   # Q1 weekend gaps
        ("tbbo",     "2026-04-24", "2026-05-06")]   # Q2 tick lead-lag
total = 0
for schema, a, b in jobs:
    for s in SYMS:
        cost = c.metadata.get_cost(dataset="GLBX.MDP3", symbols=[s], stype_in="continuous",
                                   schema=schema, start=a, end=b)
        total += cost
        print(f"{schema:9s} {s:8s} {a}->{b}  ${cost:8.2f}")
print(f"TOTAL ${total:.2f}  (free credit $125)")
