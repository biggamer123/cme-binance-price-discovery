"""Quote-age sensitivity of the trade-rebuilt Binance quote (third review, point 4), tick window 24 Apr - 5 May 2026,
CME mbp-1.  The rebuilt mid is used only when both its bid and its ask side are younger than a threshold (2 s in the
paper).  For thresholds of 0.5, 1, 2, 5, 10 and 60 s: coverage (share of CME-open 100 ms points with a rebuilt mid),
median age of the older side, LLR, bounded lead share and the cross-correlation peak.
Usage: python 32_quote_age.py GC SI ...   Output: results/q11_quote_age_<ROOT>.csv (merged with --merge)"""
import glob, importlib.util, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ["CME_SRC"] = "mbp1"
spec = importlib.util.spec_from_file_location("q2", "06_q2_leadlag.py")
q2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(q2)
THS = [500, 1000, 2000, 5000, 10000, 60000]


def run(root):
    sym = q2.PAIRS[root]
    ct, cm, cinst = q2.cme_mid(root)
    (bt, bp), (at, ap) = q2.binance_quotes(sym)
    grid = np.arange(q2.START.value // 1_000_000, q2.END.value // 1_000_000, q2.STEP_MS)
    c, _ = q2.sample(ct, cm, grid); inst, _ = q2.sample(ct, cinst.astype(float), grid)
    bid, bage = q2.sample(bt, bp, grid); ask, aage = q2.sample(at, ap, grid)
    ts = pd.to_datetime(grid, unit="ms", utc=True).tz_convert("America/Chicago")
    closed = (ts.hour == 16) | ((ts.dayofweek == 4) & (ts.hour >= 16)) | (ts.dayofweek == 5) | ((ts.dayofweek == 6) & (ts.hour < 17))
    c[closed] = np.nan; lc = np.log(c); op = np.isfinite(lc); age = np.maximum(bage, aage)
    rows = []
    for th in THS:
        lb = np.log(np.where((bage < th) & (aage < th), (bid + ask) / 2, np.nan))
        xc = q2.xcorr(lc, lb, inst)
        pos, neg = (xc[xc.lag_s > 0].rho ** 2).sum(), (xc[xc.lag_s < 0].rho ** 2).sum()
        ok = op & np.isfinite(lb)
        rows.append(dict(asset=root, threshold_s=th / 1000, coverage=ok.sum() / op.sum(),
                         age_med_s=np.median(age[ok]) / 1000 if ok.any() else np.nan,
                         LLR=pos / neg, lead_share=pos / (pos + neg), peak_ms=int(round(xc.loc[xc.rho.idxmax(), "lag_s"] * 1000))))
    return rows


if __name__ == "__main__":
    if "--merge" in sys.argv:
        x = pd.concat(pd.read_csv(f) for f in sorted(glob.glob("results/q11_quote_age_*.csv")) if not f.endswith("_all.csv"))
        x.to_csv("results/q11_quote_age_all.csv", index=False); print(x.round(3).to_string(index=False)); sys.exit()
    for r in sys.argv[1:]:
        out = pd.DataFrame(run(r)); out.to_csv(f"results/q11_quote_age_{r}.csv", index=False)
        print(out.round(3).to_string(index=False), flush=True)
