#!/usr/bin/env python3
"""Signal backtest: do common technical signals precede above/below-normal returns?

Method (no look-ahead):
  * Signals are computed on the close of day t, entry is the OPEN of day t+1.
  * Forward return over h days = close[t+h] / open[t+1] - 1, on split/dividend-adjusted prices.
  * Each signal is compared with the baseline (every eligible day of the same stock), so a
    stock that simply went up a lot doesn't make every buy signal look good.
  * Signals are de-clustered (>= 10 trading days apart).
  * Permutation test: is the signal mean better than the mean of N randomly chosen days?
  * Split at SPLIT: rules ranked on the early period, then checked on the later period.
Usage: python backtest.py [--universe]   (needs pandas, numpy; downloads to backtest/.cache)
Not financial advice.
"""
import json, sys, time, urllib.request
from pathlib import Path
from statistics import NormalDist
import numpy as np, pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / ".cache"; CACHE.mkdir(exist_ok=True)
SPLIT = pd.Timestamp("2013-01-01")
HORIZONS = [5, 10, 20, 60]
WARMUP = 252
GAP = 10
rng = np.random.default_rng(7)

def load(sym):
    f = CACHE / f"{sym}.json"
    if not f.exists() or time.time() - f.stat().st_mtime > 86400:
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=730000000&period2={int(time.time())}&interval=1d"
        r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))["chart"]["result"][0]
        f.write_text(json.dumps(r))
    r = json.loads(f.read_text())
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({"o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"], "v": q["volume"],
                       "adj": r["indicators"]["adjclose"][0]["adjclose"]},
                      index=pd.to_datetime(r["timestamp"], unit="s").normalize())
    df = df[~df.index.duplicated()].dropna(subset=["c", "adj", "o"])
    k = df["adj"] / df["c"]
    for col in "ohlc": df[col] = df[col] * k          # adjusted OHLC
    df["v"] = df["v"].fillna(0)
    return df[["o", "h", "l", "c", "v"]]

def rsi(c, n):
    d = c.diff(); g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def cross_up(a, b): return (a > b) & (a.shift() <= b.shift())
def cross_dn(a, b): return (a < b) & (a.shift() >= b.shift())

def indicators(df):
    o, h, l, c, v = df.o, df.h, df.l, df.c, df.v
    I = pd.DataFrame(index=df.index)
    for n in (20, 50, 100, 200): I[f"sma{n}"] = c.rolling(n).mean()
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    I["macd"] = ema12 - ema26; I["macds"] = I.macd.ewm(span=9, adjust=False).mean()
    I["rsi14"], I["rsi2"] = rsi(c, 14), rsi(c, 2)
    sd = c.rolling(20).std(); I["bbu"] = I.sma20 + 2 * sd; I["bbl"] = I.sma20 - 2 * sd
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    k = ((c - lo14) / (hi14 - lo14) * 100).rolling(3).mean(); I["stk"], I["stkd"] = k, k.rolling(3).mean()
    I["vr"] = v / v.rolling(20).mean().shift().replace(0, np.nan)
    I["ret"] = c.pct_change(); I["gap"] = o / c.shift() - 1
    I["hi20"], I["lo20"] = c.rolling(20).max().shift(), c.rolling(20).min().shift()
    I["hi55"], I["hi252"] = c.rolling(55).max().shift(), c.rolling(252).max().shift()
    I["off_low"] = c / l - 1
    I["ext50"] = c / I.sma50 - 1
    return I

def signals(df, I):
    o, h, l, c, v = df.o, df.h, df.l, df.c, df.v
    B, S = {}, {}
    B["RSI14 crosses up through 30"] = cross_up(I.rsi14, pd.Series(30, index=c.index))
    B["RSI14<30 & close >2% off day low"] = (I.rsi14 < 30) & (I.off_low > 0.02)
    B["RSI14<20 (extreme oversold)"] = I.rsi14 < 20
    B["RSI2<10 in uptrend (>200d)"] = (I.rsi2 < 10) & (c > I.sma200)
    B["MACD crosses above signal"] = cross_up(I.macd, I.macds)
    B["MACD cross up while MACD<0"] = cross_up(I.macd, I.macds) & (I.macd < 0)
    B["Golden cross (50d over 200d)"] = cross_up(I.sma50, I.sma200)
    B["Close crosses above 50d"] = cross_up(c, I.sma50)
    B["Close crosses above 200d"] = cross_up(c, I.sma200)
    B["20-day high breakout"] = c > I.hi20
    B["55-day high breakout"] = c > I.hi55
    B["52-week high breakout"] = c > I.hi252
    B["Back above lower Bollinger band"] = cross_up(c, I.bbl)
    B["Stoch %K crosses up %D below 20"] = cross_up(I.stk, I.stkd) & (I.stk < 20)
    B["Volume spike up (>=2x, +2% day)"] = (I.vr >= 2) & (I.ret > 0.02)
    B["Capitulation: vol>=2x, RSI<35, close upper half"] = (I.vr >= 2) & (I.rsi14 < 35) & ((c - l) / (h - l).replace(0, np.nan) > 0.5)
    B["Gap up >3% on >=1.5x volume"] = (I.gap > 0.03) & (I.vr >= 1.5)
    B["RSI14 crosses up through 50 above 200d"] = cross_up(I.rsi14, pd.Series(50, index=c.index)) & (c > I.sma200)
    B["3 down days then an up day"] = (I.ret > 0) & (I.ret.shift() < 0) & (I.ret.shift(2) < 0) & (I.ret.shift(3) < 0)
    B["Day drop of 8%+ (flush)"] = I.ret <= -0.08
    B["Day gain of 8%+ on >=2x volume"] = (I.ret >= 0.08) & (I.vr >= 2)
    S["RSI14 crosses down through 70"] = cross_dn(I.rsi14, pd.Series(70, index=c.index))
    S["RSI14>80 (extreme overbought)"] = I.rsi14 > 80
    S["MACD crosses below signal"] = cross_dn(I.macd, I.macds)
    S["Death cross (50d under 200d)"] = cross_dn(I.sma50, I.sma200)
    S["Close crosses below 50d"] = cross_dn(c, I.sma50)
    S["Close crosses below 200d"] = cross_dn(c, I.sma200)
    S["20-day low breakdown"] = c < I.lo20
    S["Volume spike down (>=2x, -2% day)"] = (I.vr >= 2) & (I.ret < -0.02)
    S["Back below upper Bollinger band"] = cross_dn(c, I.bbu)
    S["Stoch %K crosses down %D above 80"] = cross_dn(I.stk, I.stkd) & (I.stk > 80)
    S["Extended: >30% above 50d"] = I.ext50 > 0.30
    S["Close below 50d on >=1.5x volume"] = cross_dn(c, I.sma50) & (I.vr >= 1.5)
    S["Gap down >3% on >=1.5x volume"] = (I.gap < -0.03) & (I.vr >= 1.5)
    return B, S

def declust(mask):
    idx = np.flatnonzero(mask.fillna(False).to_numpy()); keep, last = [], -10**9
    for i in idx:
        if i - last >= GAP: keep.append(i); last = i
    return np.array(keep, dtype=int)

def fwd(df, h): return (df.c.shift(-h) / df.o.shift(-1) - 1)

def perm_p(sig, base, side):
    if len(sig) < 5: return np.nan
    null = rng.choice(base, size=(4000, len(sig))).mean(axis=1)
    return float((null >= sig.mean()).mean() if side == "buy" else (null <= sig.mean()).mean())

def evaluate(sym, df, I, B, S):
    rows = []; valid_from = WARMUP
    for h in HORIZONS:
        f = fwd(df, h); ok = f.notna().to_numpy().copy(); ok[:valid_from] = False
        for per, sel in (("all", ok), ("early", ok & (df.index < SPLIT)), ("late", ok & (df.index >= SPLIT))):
            base = f.to_numpy()[sel]
            if len(base) < 100: continue
            for side, D in (("buy", B), ("sell", S)):
                for name, m in D.items():
                    ix = declust(m); ix = ix[sel[ix]] if len(ix) else ix
                    if len(ix) < 5: continue
                    s = f.to_numpy()[ix]
                    rows.append(dict(ticker=sym, side=side, signal=name, h=h, period=per, n=len(ix),
                                     mean=s.mean(), median=np.median(s), hit=(s > 0).mean(),
                                     base_mean=base.mean(), base_hit=(base > 0).mean(),
                                     excess=s.mean() - base.mean(), p=perm_p(s, base, side)))
    return pd.DataFrame(rows)

# ---------- strategy backtests ----------
def run_pos(df, pos, cost=0.001):
    tgt = pos.shift(1).fillna(0); held = tgt.shift(1).fillna(0)
    ov = df.o / df.c.shift() - 1; intra = df.c / df.o - 1
    r = (1 + held * ov.fillna(0)) * (1 + tgt * intra) - 1 - (tgt - held).abs() * cost
    return r.fillna(0), tgt

def stats(r, tgt):
    n = len(r); eq = (1 + r).cumprod()
    if n < 50: return None
    yrs = n / 252; cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252); dd = (eq / eq.cummax() - 1).min()
    return dict(final=eq.iloc[-1], cagr=cagr, sharpe=(r.mean() * 252) / vol if vol else np.nan, maxdd=dd,
                exposure=tgt.mean(), trades=int((tgt.diff().abs() > 0).sum() / 2))

def strategies(df, I):
    c = df.c; P = {}
    P["Buy & hold"] = pd.Series(1.0, index=c.index)
    P["Long above 200d, else cash"] = (c > I.sma200).astype(float)
    P["Long above 50d, else cash"] = (c > I.sma50).astype(float)
    P["Golden/death cross (50d vs 200d)"] = (I.sma50 > I.sma200).astype(float)
    P["MACD above signal"] = (I.macd > I.macds).astype(float)
    # Turtle: enter 55d high, exit 20d low
    pos, p = [], 0.0
    for cl, h55, l20 in zip(c, I.hi55, I.lo20):
        if p == 0 and cl > h55: p = 1.0
        elif p == 1 and cl < l20: p = 0.0
        pos.append(p)
    P["55d breakout / 20d low exit"] = pd.Series(pos, index=c.index)
    # RSI2 pullback in uptrend: buy RSI2<10 above 200d, exit close>5d avg
    sma5 = c.rolling(5).mean(); pos, p = [], 0.0
    for cl, r2, s200, s5 in zip(c, I.rsi2, I.sma200, sma5):
        if p == 0 and r2 < 10 and cl > s200: p = 1.0
        elif p == 1 and (cl > s5 or cl < s200 * 0.9): p = 0.0
        pos.append(p)
    P["RSI2 pullback in uptrend"] = pd.Series(pos, index=c.index)
    # Dip-buy / take-profit swing: buy RSI14<30 bounce, sell RSI14>70
    pos, p = [], 0.0
    for r14, off in zip(I.rsi14, I.off_low):
        if p == 0 and r14 < 30 and off > 0.02: p = 1.0
        elif p == 1 and r14 > 70: p = 0.0
        pos.append(p)
    P["Buy RSI<30 bounce, sell RSI>70"] = pd.Series(pos, index=c.index)
    P["Above 200d AND MACD up"] = ((c > I.sma200) & (I.macd > I.macds)).astype(float)
    return P

def strat_table(sym, df, I):
    out = []
    for per, lo, hi in (("full", None, None), ("late", SPLIT, None), ("last5y", df.index[-1] - pd.DateOffset(years=5), None)):
        d = df if lo is None else df[df.index >= lo]
        P = strategies(df, I)                         # signals from full history (no look-ahead), sliced after
        for name, pos in P.items():
            r, t = run_pos(df, pos); sl = (r.index >= (lo if lo is not None else r.index[0])) & (np.arange(len(r)) >= WARMUP)
            s = stats(r[sl], t[sl])
            if s: out.append(dict(ticker=sym, period=per, strategy=name, **s))
    return pd.DataFrame(out)

if __name__ == "__main__":
    TICKERS = ["NVDA", "GOOGL", "AMZN"]; CTRL = ["SPY"]
    res, strat = [], []
    for sym in TICKERS + CTRL:
        df = load(sym); I = indicators(df); B, S = signals(df, I)
        res.append(evaluate(sym, df, I, B, S)); strat.append(strat_table(sym, df, I))
        print(sym, len(df), df.index[0].date(), "->", df.index[-1].date(), flush=True)
    pd.concat(res).to_csv(HERE / "signals_main.csv", index=False)
    pd.concat(strat).to_csv(HERE / "strategies_main.csv", index=False)
    if "--universe" in sys.argv:
        L = json.load(open(HERE.parent / "stock-monitor" / "watchlists.json"))
        U = [s for v in L.values() for s in v if s not in ("SPY", "QQQ", "SMH") + tuple(TICKERS)]
        res = []
        for sym in U:
            try: df = load(sym)
            except Exception as e: print("skip", sym, e); continue
            if len(df) < 1000: print("skip (short history)", sym, len(df)); continue
            I = indicators(df); B, S = signals(df, I); res.append(evaluate(sym, df, I, B, S)); print("univ", sym, len(df), flush=True)
        pd.concat(res).to_csv(HERE / "signals_universe.csv", index=False)
