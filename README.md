# Cross-market price discovery with asynchronous quotes: CME futures vs 24/7 Binance perpetuals

Code for the paper "Rethinking Cross-Market Price Discovery: The Impact of Asynchronous Quotes and Sampling Bias in
24/7 and Traditional Futures Markets" (K. Kalra and Y. Jangra). Ten pairs: gold, silver, platinum, palladium, copper,
WTI, Brent, natural gas, S&P 500 (ES vs SPYUSDT) and Bitcoin.

The repository holds the estimation, inference, table and figure code. **No market data are included or
redistributed**: CME data (Databento) and native Binance quotes (Tardis.dev) are licensed, and Binance's public
archive must be downloaded from Binance. "Getting the data" below lists exactly what each script expects. Every
script reads from `data/` and writes to `results/`.

## Which script produces which result in the paper

| Paper item | Scripts | Output in `results/` |
|---|---|---|
| Table 1 (activity, sample sizes) | 08, 13, 04, 26 | `table_main.csv` (trades and quote updates per day), `costs_staleness.csv` (rebuilt-quote age), `q1_summary.csv` (weekends, breaks), `q7_rq3_fulldays.csv` (native days) |
| Table 2, RQ1 columns | 04, 30 | `q1_summary.csv`; `q10_rq1_inference.csv` (beta, HC3 95% CI, Holm p for beta = 0 and 1, leave-one-out, Ljung-Box), `q10_rq1_pooled.txt` (pooled tests clustered by weekend date) |
| Table 2, RQ2 columns | 11_q3, 31 | `q3_halflife_curves.csv`; `q12_reopen_boot.csv` (gap open after 1 and 5 min, bootstrap CI) |
| Fig. 1 (gap convergence) | 11_q3, 12_q3b, 15 | `fig5_reopen_convergence.png` |
| RQ2 text (who closes the gap, catch-up, tick start at 5/10 s) | 07, 11_q3, 12_q3b, 31 | `q1b_catchup.csv`, `q3_halflife_summary.csv`, `q12_reopen_tick_start.csv` |
| RQ1 index-regime and funding comparison | 27, 28 | `review_funding_payments.csv`, `review_funding_weekends.csv`, `q8_funding_regimes.csv`, `q8_funding_zero_share.csv` |
| Table 3 (native quotes) | 24, 26, 33 | `q7_rq3_days.csv` (one row per market-day), `q7_rq3_fulldays.csv`, `q7_native_shares_by_asset.csv`, `q13_day_level.csv` |
| RQ3 index test | 28 | `q8_index_open.csv`, `q8_index_closed.csv`, `q8_index_funding.csv` |
| Table 4 (tbbo vs mbp-1; native vs rebuilt on 1 May) | 06, 11_q2, 08, 17 | `q2_summary*.csv` (`_mbp1` suffix = every quote update), `q2_tick_*.csv`, `q5_real_1may_*.csv` |
| Table 5a (timestamp shift) | 29, 33 | `q9_shift.csv`, `q9_shift_summary.csv` |
| Table 5b (quote-age threshold) | 32 | `q11_quote_age_all.csv` |
| Trading paragraph (costs, depth) | 13, 14, 18, 19 | `two_sided_fixed_rule.csv`, `costs_two_sided_native.csv` |

Run order: 04, 06, 07, 11_q2, 11_q3, 12_q3b, 13, 14, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, then 08
and 15. Script 29 is slow; it can be split with `--part k/n` and merged with `--merge`.

Python 3.11 or later; packages and minimum versions in `requirements.txt` (pandas, numpy, scipy, statsmodels,
pyarrow, matplotlib).

## Getting the data

| Folder expected by the scripts | Source | Period retrieved |
|---|---|---|
| `data/binance/<SYMBOL>/klines1m/*.csv` | data.binance.vision, futures/um/monthly/klines/1m (free) | 8 Jan 2026 (or the listing date) to 30 Sep 2026 |
| `data/binance/<SYMBOL>/funding/*.csv` | data.binance.vision, futures/um/monthly/fundingRate (free) | same as above |
| `data/binance/<SYMBOL>/aggTrades/<SYMBOL>-aggTrades-<YYYY-MM-DD>.csv` | data.binance.vision, futures/um/daily/aggTrades (free) | 24 Apr to 5 May 2026, and each native-quote date |
| `data/cme/<ROOT>_ohlcv1s.parquet` | Databento GLBX.MDP3, schema ohlcv-1s, symbol `<ROOT>.v.0`, stype_in continuous (paid) | 8 Jan 2026 (or the Binance listing date) to 30 Sep 2026 |
| `data/cme/<ROOT>_tbbo.parquet`, `data/cme/<ROOT>_mbp1.parquet` | Databento GLBX.MDP3, schemas tbbo and mbp-1, symbol `<ROOT>.v.0` (paid) | 24 Apr to 6 May 2026 (tick window: eight CME sessions) |
| `data/cme/days/<ROOT>_mbp1_<YYYY-MM-DD>.parquet` | Databento GLBX.MDP3, schema mbp-1 (paid) | each native-quote date |
| `data/tardis/book_ticker/<YYYY-MM-DD>/<SYMBOL>.csv.gz` | datasets.tardis.dev, exchange binance-futures, data type book_ticker (first day of each month is free) | first day of each month, Feb to Oct 2026 |
| `data/tardis/derivative_ticker/<YYYY-MM-DD>/<SYMBOL>.csv.gz` | datasets.tardis.dev, binance-futures, derivative_ticker | first-of-month native dates (index, mark price, funding) |
| `data/native_provider/<SYMBOL>/<YYYY-MM-DD>/*.parquet` | a recording of Binance's bookTicker stream (columns recv_ns, T_ms, E_ms, e, u, bid, bid_qty, ask, ask_qty) | 14, 16, 19-20 Jul and 19-20 Aug 2026 |

Native-quote dates that enter Table 3 (eight independent dates, 65 market-days of the nine traditional markets):
1 May (gold and platinum only), 1 Jun, 1 Jul, 14 Jul, 16 Jul, 19 Aug, 1 Sep and 1 Oct 2026. A market-day counts
when it has at least 12 hours of CME-open time with native quotes; shorter days are reported separately by 26.

Contract identifiers (CME root / Binance USDT-margined perpetual): GC/XAUUSDT, SI/XAGUSDT, PL/XPTUSDT, PA/XPDUSDT,
HG/COPPERUSDT, CL/CLUSDT, BZ/BZUSDT, NG/NATGASUSDT, ES/SPYUSDT, BTC/BTCUSDT.

## Sample rules and exclusions

- CME prices: continuous front contract by volume (`<ROOT>.v.0`). Closures where the instrument id changes between
  the last bar before the close and two hours after the reopen are dropped; on tick and native days the contract
  with most updates is used.
- Closures: reopen at 17:00-17:15 Chicago time after a gap of more than 45 min; weekend = Sunday reopen after
  40-60 h; holiday closures and seven days Databento flags as degraded are excluded.
- Native-quote information shares: the three energy market-days of 19 Aug 2026 are excluded (the two venues
  referenced different delivery months), so the shares cover 62 of the 65 market-days.
- Rebuilt Binance quotes: last seller-initiated trade = bid, last buyer-initiated trade = ask; the mid is used only
  when both sides are younger than 2 s (carried forward up to 60 s for the VECM).
- Timestamps: CME ts_event (ns, matching engine), Binance transaction/event time (ms); all converted to UTC ms.
  No clock-synchronisation correction is applied; script 29 tests sensitivity to offsets instead.
- Binance index regime: closed-market index held at its last value until 8 May 2026 21:00 UTC, order-book based
  afterwards (announced 5 May 2026).
