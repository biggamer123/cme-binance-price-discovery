"""Step 1: list Binance USD-M perps that are TradFi (non-crypto) + BTC, with listing dates.
Writes data/meta/perp_universe.csv"""
import requests, datetime as dt, csv, os
os.makedirs("data/meta", exist_ok=True)
info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=30).json()
rows = []
for s in info["symbols"]:
    ut, sub = s.get("underlyingType", ""), s.get("underlyingSubType", [])
    tradfi = ut not in ("COIN", "INDEX") or any(k in str(sub).upper() for k in ("TRADFI", "COMMODITY", "STOCK", "EQUITY"))
    if tradfi or s["symbol"] in ("BTCUSDT", "PAXGUSDT", "XAUTUSDT"):
        rows.append(dict(symbol=s["symbol"], underlyingType=ut, underlyingSubType="|".join(sub),
                         contractType=s.get("contractType"), status=s["status"],
                         onboard=dt.datetime.utcfromtimestamp(s["onboardDate"]/1000).date().isoformat()))
rows.sort(key=lambda r: r["onboard"])
with open("data/meta/perp_universe.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
for r in rows:
    flag = "OK (before 24 Apr)" if r["onboard"] < "2026-04-24" else "too late"
    print(f'{r["symbol"]:14s} {r["underlyingType"]:10s} {r["underlyingSubType"]:22s} {r["onboard"]}  {r["status"]:8s} {flag}')
