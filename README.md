# Who prices the weekend? CME futures vs 24/7 Binance perpetuals

Analysis code for the paper "Who Prices the Weekend? Price Discovery between CME Futures and 24/7 Binance
Perpetuals on Traditional Assets" (K. Kalra and Y. Jangra). Ten pairs: gold, silver, platinum, palladium, copper,
WTI, Brent, natural gas, S&P 500 (ES vs SPYUSDT) and Bitcoin.

The repository holds the estimation and inference code only. Data are not included (CME and native-quote data
cannot be redistributed); "Getting the data" below lists what each script expects. Every script writes its
outputs to `results/`.

## Scripts (run from the repository folder, in this order)

| Research question | Script | What it estimates |
|---|---|---|
| RQ1 weekend pricing | 04 | reopen gaps and the regression of the CME gap on the Binance closure return (weekends, daily breaks) |
| RQ1 inference | 30 | 95% intervals, tests of beta = 0 and 1 (Holm), leave-one-out, Ljung-Box, pooled tests clustered by weekend |
| RQ2 convergence | 07, 11_q3, 12_q3b | catch-up regression; gap path after each reopen (1-minute and tick data), half-lives, who closes the gap |
| RQ2 uncertainty | 31 | bootstrap intervals for the gap left after 1 and 5 min; tick check starting 5 / 10 s after the reopen |
| RQ3 lead-lag | 06, 11_q2 | 100 ms cross-correlation, LLR, VECM information and component shares (tbbo vs mbp-1); Hayashi-Yoshida, D(h) |
| RQ3 native quotes | 17, 24, 25, 26 | rebuilt vs native Binance quotes on 1 May; all native days (LLR, HY, IS, ILS, half-life); summaries |
| RQ3 robustness | 29, 32 | Binance timestamps shifted by -100..+50 ms; quote-age thresholds 0.5-60 s for rebuilt quotes |
| Contract checks | 27, 28 | review checks (HC3, reverse bound, funding, ILS); index channel, closed-market index and funding regimes |
| Trading costs | 13, 14, 18, 19 | spreads, quote staleness, two-sided reopen trade after costs, native spreads and depth |

Python 3.11+, packages in `requirements.txt`.

## Getting the data

| Folder expected by the scripts | Source | Content |
|---|---|---|
| `data/binance/<SYMBOL>/aggTrades/<SYMBOL>-aggTrades-<YYYY-MM-DD>.csv` | data.binance.vision, futures/um/daily/aggTrades (free) | trades, 24 Apr - 5 May 2026 and native days |
| `data/binance/<SYMBOL>/klines1m/*.csv` | data.binance.vision, futures/um/monthly/klines/1m (free) | 1-minute candles, Jan - Sep 2026 |
| `data/binance/<SYMBOL>/funding/*.csv` | data.binance.vision, futures/um/monthly/fundingRate (free) | funding payments |
| `data/cme/<ROOT>_ohlcv1s.parquet` | Databento GLBX.MDP3, schema ohlcv-1s, symbol `<ROOT>.v.0`, stype continuous (paid) | 1-second bars from 8 Jan (or the Binance listing) to 30 Sep 2026 |
| `data/cme/<ROOT>_tbbo.parquet`, `data/cme/<ROOT>_mbp1.parquet` | Databento GLBX.MDP3, schemas tbbo and mbp-1 (paid) | 24 Apr - 6 May 2026 |
| `data/cme/days/<ROOT>_mbp1_<YYYY-MM-DD>.parquet` | Databento GLBX.MDP3, schema mbp-1 (paid) | each native-quote day |
| `data/tardis/book_ticker/<YYYY-MM-DD>/<SYMBOL>.csv.gz` | datasets.tardis.dev, binance-futures/book_ticker (first day of each month is free) | native best bid/offer |
| `data/tardis/derivative_ticker/<YYYY-MM-DD>/<SYMBOL>.csv.gz` | datasets.tardis.dev, binance-futures/derivative_ticker | index, mark price and funding about once a second |
| `data/native_provider/<SYMBOL>/<YYYY-MM-DD>/*.parquet` | a recording of Binance's bookTicker stream (columns recv_ns, T_ms, E_ms, e, u, bid, bid_qty, ask, ask_qty) | native quotes on 14, 16, 19-20 Jul and 19-20 Aug 2026 |

CME roots and Binance symbols: GC/XAUUSDT, SI/XAGUSDT, PL/XPTUSDT, PA/XPDUSDT, HG/COPPERUSDT, CL/CLUSDT,
BZ/BZUSDT, NG/NATGASUSDT, ES/SPYUSDT, BTC/BTCUSDT.

## Sample rules

- CME prices: continuous front contract by volume (`<ROOT>.v.0`). Closures where the instrument id changes between
  the last bar before the close and two hours after the reopen are dropped; on tick and native days the contract
  with most updates is used.
- Closures: reopen at 17:00-17:15 Chicago time after a gap > 45 min; weekend = Sunday reopen after 40-60 h; holiday
  closures and seven Databento-degraded days are excluded.
- Timestamps: CME ts_event (ns, matching engine), Binance transaction/event time (ms); all converted to UTC ms.
