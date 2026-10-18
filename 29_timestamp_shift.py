"""Timestamp-shift sensitivity (third review, point 2).  The two venues stamp events with separate clocks (CME:
matching-engine time in ns; Binance: matching-engine time in ms).  To see whether the CME lead could be a clock
artefact, every Binance timestamp is moved by delta ms (delta < 0 = Binance events treated as EARLIER, which works
against a CME lead) and the lead-lag measures are recomputed on every full native-quote day:
  100 ms grid cross-correlation (LLR, peak), Hayashi-Yoshida peak, directional response D(100 ms) both ways.
This is a sensitivity test, not a clock calibration.
Usage: python 29_timestamp_shift.py [--days 2026-06-01,...] [--assets GC,SI] [--shifts -100,-50,...] [--part k/n]
       python 29_timestamp_shift.py --merge
Output: results/q9_shift_<part>.csv per worker (market-days already done are skipped), merged into results/q9_shift.csv"""
import importlib.util, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("nm", "24_native_multiday.py")
nm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nm)
SHIFTS = [-100, -75, -50, -25, -10, -5, 0, 5, 10, 25, 50]


def full_days():
    d = pd.read_csv("results/q7_rq3_days.csv")
    d = d[(d.cme_ticks.astype(str).str.lower() == "true") & (d.open_hours >= 12)]
    return d[["day", "asset"]].values.tolist()


def run(day, root, shifts):
    sym = nm.PAIRS[root]; d0 = pd.Timestamp(day, tz="UTC").value // 1_000_000; d1 = d0 + 86_400_000
    nat = nm.load_native(sym, day); cme = nm.load_cme(root, day, d0, d1)
    if nat is None or cme is None:
        return []
    grid = np.arange(d0, d1, nm.STEP); cl = nm.closed(grid)
    C, _ = nm.asof(cme["t"], cme["mid"], grid); C[cl] = np.nan; lc = np.log(C)
    ce = nm.moves(cme["t"], cme["mid"], d0, d1)
    mid = (nat["bid"] + nat["ask"]) / 2
    b0 = nm.moves(nat["t"], mid, d0, d1)                 # Binance mid changes; a shift moves their times only
    rows = []
    for dl in shifts:
        tb = nat["t"] + dl
        B, _ = nm.asof(tb, nat["bid"], grid); A, _ = nm.asof(tb, nat["ask"], grid); TM = (B + A) / 2
        have = (grid >= tb[0]) & (grid <= tb[-1]); TM[cl | ~have] = np.nan
        llr, pk, rho = nm.xcorr(lc, np.log(TM))
        be = (b0[0] + dl, b0[1], b0[2])
        hyp, hyr, dbc, dcb, _ = nm.tick_measures(ce, be)
        rows.append(dict(day=day, asset=root, shift_ms=dl, LLR=llr, lead_share=llr / (1 + llr), grid_peak_ms=pk,
                         HY_peak_ms=hyp, HY_rho=hyr, D_bin_after_cme=dbc, D_cme_after_bin=dcb))
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    get = lambda k: a[a.index(k) + 1].split(",") if k in a else None
    days, assets = get("--days"), get("--assets")
    shifts = [int(x) for x in get("--shifts")] if get("--shifts") else SHIFTS
    import glob, os
    if "--merge" in a:
        x = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/q9_shift_p*.csv"))]).drop_duplicates(["day", "asset", "shift_ms"], keep="last")
        x.sort_values(["asset", "day", "shift_ms"]).to_csv("results/q9_shift.csv", index=False); print(len(x), "rows"); sys.exit()
    k, n = (int(v) for v in get("--part")[0].split("/")) if get("--part") else (0, 1)
    out = f"results/q9_shift_p{k}.csv"
    done = set()
    for f in glob.glob("results/q9_shift_p*.csv"):
        x = pd.read_csv(f); c = x.groupby(["day", "asset"]).shift_ms.nunique()
        done |= set(c[c >= len(shifts)].index)
    todo = [(d, r) for d, r in full_days() if (days is None or d in days) and (assets is None or r in assets)]
    todo = [t for i, t in enumerate(todo) if i % n == k and tuple(t) not in done]
    for d, r in todo:
        rows = run(d, r, shifts)
        if rows:
            nm.save(rows, out, keys=("day", "asset", "shift_ms"))
            x = pd.DataFrame(rows)
            print(d, r, " ".join(f"{s:+d}:{p}/{h}/{q:.2f}" for s, p, h, q in zip(x.shift_ms, x.grid_peak_ms, x.HY_peak_ms, x.D_bin_after_cme - x.D_cme_after_bin)), flush=True)
