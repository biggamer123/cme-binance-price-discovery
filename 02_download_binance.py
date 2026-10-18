"""Step 2: download free Binance archive data for the paper.
  tick window : aggTrades daily, 24 Apr - 5 May 2026
  bar window  : 1m klines monthly (daily fallback), 8 Jan - 30 Sep 2026
  funding     : fundingRate monthly
Usage: python 02_download_binance.py XAUUSDT XAGUSDT BTCUSDT
Re-runnable: skips files already downloaded."""
import sys, os, io, zipfile, requests, datetime as dt
BASE = "https://data.binance.vision/data/futures/um"
OUT = "data/binance"
TICK = ("2026-04-24", "2026-05-05")
MONTHS = [f"2026-{m:02d}" for m in range(1, 10)]
# Binance perp -> CME pair. All listed before 24 Apr 2026 and archived on data.binance.vision
SYMBOLS = ["XAUUSDT", "XAGUSDT", "XPTUSDT", "XPDUSDT", "COPPERUSDT",   # GC SI PL PA HG
           "CLUSDT", "BZUSDT", "NATGASUSDT",                          # CL BZ NG
           "SPYUSDT", "QQQUSDT", "BTCUSDT"]                           # ES NQ BTC

def days(a, b):
    d, e = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    while d <= e:
        yield d.isoformat(); d += dt.timedelta(days=1)

def fetch(url, dest):
    if os.path.exists(dest): return "skip"
    import time
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=120); break
        except requests.exceptions.RequestException:
            time.sleep(5 * (attempt + 1))
    else:
        return "MISSING network"
    if r.status_code != 200: return f"MISSING {r.status_code}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        with z.open(name) as src, open(dest, "wb") as out: out.write(src.read())
    return "ok"

LISTED = {"COPPERUSDT": "2026-03", "CLUSDT": "2026-04", "BZUSDT": "2026-04", "NATGASUSDT": "2026-04",
          "SPYUSDT": "2026-04", "QQQUSDT": "2026-04"}

def run(sym):
    log = []
    for d in days(*TICK):
        log.append(("aggTrades", d, fetch(f"{BASE}/daily/aggTrades/{sym}/{sym}-aggTrades-{d}.zip",
                                          f"{OUT}/{sym}/aggTrades/{sym}-aggTrades-{d}.csv")))
    for m in [m for m in MONTHS if m >= LISTED.get(sym, "0")]:
        st = fetch(f"{BASE}/monthly/klines/{sym}/1m/{sym}-1m-{m}.zip", f"{OUT}/{sym}/klines1m/{sym}-1m-{m}.csv")
        if st.startswith("MISSING"):   # monthly not published yet -> daily files
            y, mo = map(int, m.split("-"))
            last = (dt.date(y + (mo == 12), mo % 12 + 1, 1) - dt.timedelta(days=1)).isoformat()
            sts = [fetch(f"{BASE}/daily/klines/{sym}/1m/{sym}-1m-{d}.zip", f"{OUT}/{sym}/klines1m/{sym}-1m-{d}.csv")
                   for d in days(f"{m}-01", min(last, "2026-09-30"))]
            st = f"daily: {sum(s in ('ok','skip') for s in sts)}/{len(sts)}"
        log.append(("klines1m", m, st))
        log.append(("funding", m, fetch(f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip",
                                        f"{OUT}/{sym}/funding/{sym}-fundingRate-{m}.csv")))
    bad = [l for l in log if str(l[2]).startswith("MISSING") or "daily: 0/" in str(l[2])]
    print(f"{sym}: {len(log)-len(bad)}/{len(log)} files OK" + (f"; missing: {bad[:6]}" if bad else ""))

if __name__ == "__main__":
    for s in sys.argv[1:] or SYMBOLS:
        run(s)
