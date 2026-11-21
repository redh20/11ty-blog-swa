# Signal backtest results (run 2026-11-21, data through 2026-11-20)

**Scope:** NVDA (1999-), GOOGL (2004-), AMZN (1997-), SPY as a control, plus 24 other AI-watchlist stocks as a robustness check.
**SpaceX (SPCX) has only 113 trading days (since 2026-06-12): it cannot be backtested.** Rules below are applied to it going forward only.

## Method (no look-ahead)
- 34 signals (20 buy-side, 14 sell-side): RSI, MACD, 50/200-day averages and crosses, breakouts, Bollinger, stochastic, volume spikes, gaps, flush days.
- Signal is computed on the close of day t. Entry is the open of day t+1. Forward return over 5/10/20/60 days on split- and dividend-adjusted prices.
- Each signal is compared with the stock's own baseline (all eligible days), since these stocks rose a lot anyway. Signals are de-clustered (10+ trading days apart). Permutation test for significance.
- Early period (before 2013) vs late period (2013 on). Walk-forward: choose on early, test on late.
- Strategy backtests: 0.1% cost per trade, trade at next open.
- Reproduce: `python backtest.py --universe && python analyze.py && python analyze2.py` (needs pandas, numpy).

## Headline findings
1. **Almost no signal beats chance.** Signals significant at p<0.05 (NVDA+GOOGL+AMZN pooled): 2 / 3 / 4 / 0 of 34 at 5 / 10 / 20 / 60 days. About 1.7 are expected from luck alone.
2. **Early success did not carry over.** The 5 best buy signals in the early period (+4.1% excess over 20 days) delivered +0.5% later, versus +0.3% for the average signal.
3. **Effects are small relative to noise.** Typical edge is 1-3% over 20 days; the 20-day return standard deviation is 8% (GOOGL) to 16% (NVDA, AMZN).
4. **Timing rules did not beat buy-and-hold on return** for these stocks. The 200-day trend filter was the one rule that consistently cut drawdowns.
5. **Mean-reversion exits hurt:** "buy RSI<30 bounce, sell RSI>70" ended at 0.7x for NVDA vs 1,456x buy-and-hold.

## Signals that were same-direction in every cut (20-day, excess vs baseline %)
| Side | Signal | 3 stocks | p | early | late | 24 others | p |
|---|---|---|---|---|---|---|---|
| Buy | Day gain 8%+ on 2x+ volume | +3.5 (n=96) | 0.02 | +5.0 | +0.6 | +1.6 (n=596) | 0.01 |
| Buy | Golden cross (50d over 200d) | +3.1 (n=42) | 0.04 | +4.9 | +1.5 | +0.5 | 0.15 |
| Buy | Gap up >3% on 1.5x+ volume | +0.9 | 0.24 | +1.7 | +0.2 | +0.8 | 0.03 |
| Sell | Close below 50d on 1.5x+ volume | -1.2 (n=141) | 0.11 | -2.3 | -0.6 | -0.9 | 0.01 |
| Sell | Close crosses below 50d | -0.4 | 0.33 | -0.7 | -0.1 | -0.7 | 0.01 |
| Sell | Gap down >3% on 1.5x+ volume | -2.2 (n=180) | 0.01 | -2.5 | -1.9 | -0.2 | 0.30 |

10 of 33 signal/side combos kept the same direction in all five cuts (the 6 above had p<=0.10 in at least one). Pure chance would give roughly 1-4, and the cuts overlap, so there is a mild real tendency, but edges shrink in the late period and are small next to the noise. Treat these as weak context, not trade triggers.

## Strategy backtests, full history (final multiple, CAGR, max drawdown)
| | Buy & hold | Long above 200d, else cash | Golden/death cross |
|---|---|---|---|
| NVDA | 1456x, 31%, -90% | 4510x, 37%, -54% | 1191x, 30%, -72% |
| GOOGL | 37.5x, 19%, -65% | 17.6x, 15%, -35% | 12.8x, 13%, -43% |
| AMZN | 647x, 26%, -94% | 50x, 15%, -88% | 214x, 21%, -64% |
| SPY | 26x, 10.5%, -55% | 10.4x, 7%, -30% | 18.4x, 9%, -34% |
Since 2013: NVDA 445x vs 438x (drawdown -66% vs -49%); GOOGL 15.1x vs 8.1x; AMZN 19.7x vs 4.3x; SPY 6.1x vs 3.7x.
The NVDA full-history gain comes largely from sidestepping the 2000-2002 crash.

## Caveats
- Survivorship/hindsight: NVDA, GOOGL, AMZN are among the biggest winners of the period, so buy-and-hold is very hard to beat.
- p-values are optimistic: forward windows overlap and signals cluster in a few market-wide episodes.
- ~140 signal/horizon tests were run, so some will look good by chance.
- Daily bars only; no intraday, options, fundamentals or news.
- Not financial advice.
