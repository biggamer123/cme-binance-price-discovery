# Who prices the weekend? CME futures vs 24/7 Binance perpetuals

Code for the paper "Who Prices the Weekend? Price Discovery between CME Futures and 24/7 Binance Perpetuals on
Traditional Assets" (K. Kalra and Y. Jangra). Ten pairs: gold, silver, platinum, palladium, copper, WTI, Brent, natural gas,
S&P 500 (ES vs SPYUSDT) and Bitcoin.

No data are included. Binance trades, 1-minute candles and funding rates are public (data.binance.vision, script 02).
CME data come from Databento (GLBX.MDP3, paid; scripts 03/05/09/21 ask for your own API key and show the cost
before buying). Native Binance quotes come from Tardis.dev (first-of-month files are free) and from a recording of
Binance's best bid/offer stream (converted with 22).

## Pipeline (run in this order, from the repository folder; data go to data/, outputs to results/)

| Step | Script | What it does |
|---|---|---|
| Data | 01, 02 | list Binance TradFi perpetuals; download Binance trades, candles, funding |
| Data | 03, 05, 09, 21 | CME cost check and download (1 s bars, tbbo, mbp-1, mbp-1 for single days) |
| RQ1 | 04, 07 | weekend/daily-break reopen-gap regressions; post-reopen catch-up |
| RQ3 | 06, 10, 11_q2_ticktime | 100 ms lead-lag, VECM information shares, Hayashi-Yoshida tick-time measures |
| RQ2 | 11_q3, 12_q3b, 12_tick | reopen half-lives (1 min and tick data), who closes the gap |
| Costs | 13, 14, 18, 19 | spreads, staleness, two-sided trade, native spreads and depth |
| Figures | 08, 15, 20 | tables and figures |
| Native | 17, 22, 24, 25, 26 | native-quote validation (1 May; all native days), summaries, Table 4 numbers |
| Review | 27 | HC3/leave-one-out/reverse-regression checks, weekend funding check, ILS, event counts |
| Index | 28 | Binance index vs CME and perpetual (joint hours), closed-market index, mark band and funding (Tardis derivative_ticker files) |
| Revision | 29 | timestamp-shift sensitivity: Binance times moved by -100..+50 ms, LLR / HY / D(100 ms) on every native day |
| Revision | 30 | weekend-regression inference: 95% intervals, tests of beta = 0 and 1 (Holm), leave-one-out, Ljung-Box, pooled tests clustered by weekend |
| Revision | 31 | reopen uncertainty: bootstrap intervals for the gap left after 1 and 5 min; tick check starting 5 / 10 s after the reopen |
| Revision | 32 | quote-age sensitivity of the trade-rebuilt Binance quote (0.5-60 s): coverage, lead share, peak |

Python 3.11+, packages in requirements.txt.

## Data and reproduction notes

- CME: Databento GLBX.MDP3, continuous front contract by volume (`<root>.v.0`, roots GC SI PL PA HG CL BZ NG ES BTC).
  Closures where the instrument id changes between the last bar before the close and two hours after the reopen are
  dropped; on tick and native days the contract with most updates is used.
- Closures: reopen at 17:00-17:15 Chicago time after a gap > 45 min; weekend = Sunday reopen after 40-60 h; holiday
  closures and seven Databento-degraded days are excluded.
- Timestamps: CME ts_event (ns, matching engine), Binance transaction/event time (ms); all converted to UTC ms.
- Data cannot be redistributed: Binance archive files are free (script 02), CME data need a Databento key (03, 05, 09, 21),
  native quotes come from Tardis.dev first-of-month files (free) or a recorded Binance stream (converted with 22).

