# Momentum study: does anything predict a stock's forward direction? (run 2026-11-21)

Research only, not part of the monitor. Code: `momentum_study.py`, `momentum_models.py` (needs pandas, numpy, scipy, scikit-learn).

## Scope
- **11 testable stocks** from the 13-ticker watchlist: AMZN, NVDA, GOOGL, META, MSFT, AAPL, MU, AMD, DELL, LLY, MRNA (history 7 to 33 years).
- **ANT (2 trading days) and SPCX (113 trading days) cannot be backtested.**
- **Hold-out set of 22 stocks not on the list** (AVGO, TSM, QCOM, AMAT, LRCX, KLAC, ASML, ORCL, CRM, INTC, JPM, XOM, KO, PG, JNJ, WMT, COST, HD, CAT, BA, UNH, PEP), used to check whether anything generalizes.

## Method (no look-ahead)
- 37 features computed on the close of day t: returns over 1 to 252 days, 12-1 momentum, distance from 20/50/200-day averages and their slopes, RSI, MACD, stochastic, ADX, Bollinger, volatility, volume ratios, drawdown, position in the 52-week range, strength vs SPY, streaks and gaps.
- Forward return = close[t+h] / open[t+1] - 1 for h = 5, 10, 20, 60 days, on adjusted prices.
- A. Time-series rank correlation (IC) of each feature with each stock's own forward return, early (before 2016) vs late (2016+). t-stats adjusted for overlapping windows.
- B. Cross-sectional IC: does a feature pick which of the 11 stocks does better? (non-overlapping dates)
- C. "Personality": variance ratios (above 1 = trends, below 1 = snaps back).
- D. Trend and volatility regimes, and what follows entering an uptrend or downtrend.
- E. Decision-tree rules mined on pre-2016 data, tested on 2016+ and on the hold-out stocks.
- F. Walk-forward models (logistic regression, gradient boosting): each year predicted by a model trained only on earlier years (2012-2026), then turned into a trading rule and compared with simple rules.

## Headline: no reliable directional edge was found
1. **Single features:** at 10, 20 and 60 days, none of 37 features reaches |z|>2 (about 1.7 expected by chance each). Best are the 200-day slope and 60-day relative strength at z of about 1.7, and early vs late disagree.
2. **5 days only:** 6 of 37 pass |z|>2 (about 1.7 expected). All are short-term reversal or volume features (5/10-day return, streak, MACD histogram, RSI2, volume trend). The effect is tiny (IC about -0.02) and not tradable after costs.
3. **Which stock to pick:** RSI14 and 12-month return ranked stocks well before 2016 (RSI14 IC +0.12, t 4.2) but not after (+0.01, t 0.3). It decayed.
4. **Rules mined before 2016 failed after:** tree leaf with a 35% up-rate in training had 72% in 2016+; another went 42% to 66%. They flipped.
5. **Models have no skill:** out-of-sample AUC 0.50 (logistic) and 0.50 (boosting) on the 11 stocks, 0.49 on the 22 hold-out stocks. Rank IC is positive in 10 of 15 years for one model and 6 of 15 for the other, i.e. random.
6. **Model-based trading lost to buy-and-hold** and to simple trend rules (below).
7. **No stock has a feature that survives:** for every one of the 11 stocks, 0 of 37 features have |t|>2.5 at 20 days (about 0.5 expected per stock).

## What the data does show (risk structure, not direction)
- **Trend regime separates risk more than return.** Uptrend (price above the 200-day and 50-day above 200-day) vs downtrend, 20-day forward: mean +2.8% vs +1.6%, up-rate 59.5% vs 53.6%, standard deviation 11.9% vs 15.1%. The uptrend beat the downtrend in 6 of 11 stocks.
- **Entering a downtrend** was followed by -1.6% excess over 20 days (328 events, negative in 8 of 11 stocks). Entering an uptrend: +0.5% (456 events, positive in 6 of 11). Weak.
- **Volatility is about risk, not direction:** high-volatility regimes had forward 20-day standard deviation of 14.5% vs 11.0% in low-volatility regimes, with similar average returns (2.8% vs 2.1%).
- **Trend filter (hold only above the 200-day), 2012-2026, equal-weight:**
  - 11 stocks: Sharpe 1.54 vs 1.37, max drawdown -25% vs -42%, CAGR 29.8% vs 37.2%.
  - 22 hold-out stocks: Sharpe 1.01 vs 1.10, max drawdown -21% vs -33%, CAGR 11.9% vs 20.8%.
  - It consistently cuts drawdowns, but does not reliably improve risk-adjusted return.

## Stock personalities (60-day variance ratio; below 1 = snaps back)
| Stock | VR60 | z | early | late | Reading |
|---|---|---|---|---|---|
| MSFT | 0.68 | -3.3 | 0.71 | 0.59 | Snaps back, stable over time |
| LLY | 0.69 | -3.2 | 0.63 | 0.73 | Snaps back, stable over time |
| AMZN | 0.84 | -1.5 | 0.86 | 0.76 | Mild snap-back |
| GOOGL | 0.88 | -1.0 | 1.02 | 0.75 | Mild snap-back, mostly recent |
| NVDA, AMD, MU, MRNA | 1.0 to 1.2 | under 1.4 | mixed | 0.86 to 1.19 | Slight momentum lean, not significant |
- After a rising 60 days, LLY was less likely to rise again (up-rate 56% vs 63% after a falling 60 days). MU (+11 pts), MRNA (+11), NVDA (+6) and AMD (+5) went the other way.
- Best per-stock hints (hypotheses only, not significant after multiple testing): **MU** momentum-lean (60-day vol-adjusted return IC +0.10, same sign early and late); **LLY** reversal-lean (60-day return IC -0.11, same sign early and late, t -2.2).

## Strategy test 2012-2026 (equal-weight, 0.1% cost, next-open execution)
| Strategy | 11 stocks: final x, CAGR, Sharpe, max DD | 22 hold-out: final x, CAGR, Sharpe, max DD |
|---|---|---|
| Buy and hold | 110x, 37.2%, 1.37, -42% | 16.6x, 20.8%, 1.10, -33% |
| Above 200-day | 48x, 29.8%, 1.54, -25% | 5.3x, 11.9%, 1.01, -21% |
| 60-day return > 0 | 33x, 26.5%, 1.50, -24% | 4.1x, 9.9%, 0.92, -18% |
| 12-1 momentum > 0 | 44x, 29.0%, 1.36, -28% | 7.5x, 14.5%, 1.00, -26% |
| Logistic model, long when p > base | 4.6x, 10.9%, 0.78, -38% | 2.8x, 7.3%, 0.72, -18% |
| Gradient boosting, long when p > base | 9.8x, 16.6%, 1.12, -28% | 2.3x, 5.9%, 0.61, -20% |

## What this suggests looking for in the future
- Don't expect a pattern to tell you the next direction. Use these only for risk: being below the 200-day or entering a downtrend means bigger swings and a lower up-rate; high volatility means bigger moves both ways.
- Position size and stops matter more than entry signals.
- MSFT and LLY have historically snapped back after strong 60-day runs; MU leans toward continuing. Treat as hypotheses to keep watching.

## Caveats
- Survivorship and hindsight: these are today's winners; buy-and-hold is very hard to beat.
- p-values are approximate (overlapping windows, correlated stocks) and 37 features x 4 horizons x 11 stocks were tested.
- Daily data only; no earnings, news, options or fundamentals.
- DELL (from 2016) and MRNA (from 2018) have no "early" period.
- Not financial advice.
