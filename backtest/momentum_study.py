#!/usr/bin/env python3
"""Does anything measurable predict a stock's forward direction? (research only, not part of the monitor)

Parts:  A time-series feature IC per stock | B cross-sectional IC | C trend/mean-reversion personality
        D regime tables | E decision-tree rule mining | F walk-forward models + strategy test
No look-ahead: features use data up to the close of day t; forward return = close[t+h]/open[t+1]-1.
Usage: python momentum_study.py            (needs pandas, numpy, scipy, scikit-learn)
Not financial advice.
"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score
import backtest as bt
warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
CORE = ["AMZN", "NVDA", "GOOGL", "META", "MSFT", "AAPL", "MU", "AMD", "DELL", "LLY", "MRNA"]
HOLD = ["AVGO", "TSM", "QCOM", "AMAT", "LRCX", "KLAC", "ASML", "ORCL", "CRM", "INTC",
        "JPM", "XOM", "KO", "PG", "JNJ", "WMT", "COST", "HD", "CAT", "BA", "UNH", "PEP"]
HZ = [5, 10, 20, 60]
SPLIT = pd.Timestamp("2016-01-01")
WARM = 260

def adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    plus = ((up > dn) & (up > 0)) * up; minus = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1/n, adjust=False).mean() / a; mdi = 100 * minus.ewm(alpha=1/n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1/n, adjust=False).mean(), pdi - mdi

def streak(ret):
    s, cur = np.zeros(len(ret)), 0
    for i, r in enumerate(ret.to_numpy()):
        cur = (cur + 1 if cur > 0 else 1) if r > 0 else ((cur - 1 if cur < 0 else -1) if r < 0 else 0)
        s[i] = cur
    return pd.Series(s, index=ret.index)

def features(df, spy):
    o, h, l, c, v = df.o, df.h, df.l, df.c, df.v
    F = {}
    ret = c.pct_change()
    for k in (1, 5, 10, 20, 60, 120, 252): F[f"ret_{k}d"] = c.pct_change(k)
    F["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    sma = {n: c.rolling(n).mean() for n in (20, 50, 200)}
    for n in sma: F[f"dist_sma{n}"] = c / sma[n] - 1
    F["sma50_slope"] = sma[50] / sma[50].shift(20) - 1; F["sma200_slope"] = sma[200] / sma[200].shift(20) - 1
    F["ma_stack"] = (sma[20] > sma[50]).astype(float) + (sma[50] > sma[200]) + (c > sma[20])
    F["rsi14"], F["rsi2"] = bt.rsi(c, 14), bt.rsi(c, 2)
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean(); m = e12 - e26
    F["macd_hist"] = (m - m.ewm(span=9, adjust=False).mean()) / c
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max(); F["stoch_k"] = (c - lo14) / (hi14 - lo14) * 100
    F["adx14"], F["di_diff"] = adx(h, l, c)
    sd = c.rolling(20).std(); bbu, bbl = sma[20] + 2 * sd, sma[20] - 2 * sd
    F["bb_pctb"] = (c - bbl) / (bbu - bbl); F["bb_width"] = (bbu - bbl) / sma[20]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    F["atr_pct"] = tr.rolling(14).mean() / c
    F["rv20"] = ret.rolling(20).std(); F["rv_ratio"] = ret.rolling(20).std() / ret.rolling(60).std()
    vm = v.rolling(20).mean().replace(0, np.nan)
    F["vol_ratio"] = v / vm.shift(); F["vol_trend"] = v.rolling(5).mean() / vm
    upv = (v * (ret > 0)).rolling(20).sum(); dnv = (v * (ret < 0)).rolling(20).sum()
    F["updown_vol"] = np.log((upv + 1) / (dnv + 1))
    F["dd_252"] = c / c.rolling(252).max() - 1
    hi, lo = c.rolling(252).max(), c.rolling(252).min(); F["pos_52w"] = (c - lo) / (hi - lo)
    sp = spy.reindex(df.index).ffill()
    F["rel_20d"] = F["ret_20d"] - sp.pct_change(20); F["rel_60d"] = F["ret_60d"] - sp.pct_change(60)
    F["streak"] = streak(ret); F["gap"] = o / c.shift() - 1; F["day_range_pos"] = (c - l) / (h - l).replace(0, np.nan)
    F["ret_20d_vol_adj"] = F["ret_20d"] / F["rv20"]; F["ret_60d_vol_adj"] = F["ret_60d"] / F["rv20"]
    X = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan)
    return X

def fwd(df, h): return df.c.shift(-h) / df.o.shift(-1) - 1

def load_all(tickers):
    spy = bt.load("SPY").c; D = {}
    for t in tickers:
        try: df = bt.load(t)
        except Exception as e: print("skip", t, e); continue
        if len(df) < 1100: print("skip (short)", t, len(df)); continue
        X = features(df, spy); X["_idx"] = np.arange(len(df)); D[t] = (df, X)
    return D

# ---------------- A. time-series feature IC ----------------
def ic_rows(D):
    rows = []
    for t, (df, X) in D.items():
        for h in HZ:
            y = fwd(df, h)
            for f in [c for c in X.columns if c != "_idx"]:
                d = pd.concat([X[f], y], axis=1, keys=["x", "y"]).iloc[WARM:].dropna()
                for per, dd in (("all", d), ("early", d[d.index < SPLIT]), ("late", d[d.index >= SPLIT])):
                    if len(dd) < 250: continue
                    ic = sps.spearmanr(dd.x, dd.y)[0]; ne = max(len(dd) / h - 2, 3)
                    rows.append(dict(ticker=t, feature=f, h=h, period=per, n=len(dd), ic=ic, t=ic * np.sqrt(ne) / np.sqrt(max(1 - ic ** 2, 1e-9))))
    return pd.DataFrame(rows)

def pool_ic(R):
    g = []
    for (f, h, per), d in R.groupby(["feature", "h", "period"]):
        k = len(d); g.append(dict(feature=f, h=h, period=per, stocks=k, mean_ic=d.ic.mean(), z=d.t.sum() / np.sqrt(k),
                                  same_sign=int((np.sign(d.ic) == np.sign(d.ic.mean())).sum())))
    return pd.DataFrame(g)

# ---------------- B. cross-sectional IC ----------------
def xs_ic(D, hs=(20, 60)):
    feats = [c for c in next(iter(D.values()))[1].columns if c != "_idx"]
    rows = []
    for h in hs:
        Y = pd.DataFrame({t: fwd(df, h) for t, (df, X) in D.items()})
        for f in feats:
            Xp = pd.DataFrame({t: X[f] for t, (df, X) in D.items()})
            m = Xp.notna() & Y.notna() & (pd.Series(np.arange(len(Xp)), index=Xp.index) >= 0).to_numpy()[:, None]
            Xr, Yr = Xp.where(m).rank(axis=1), Y.where(m).rank(axis=1)
            ic = Xr.corrwith(Yr, axis=1); cnt = m.sum(axis=1)
            ic = ic[(cnt >= 8) & (ic.index >= Xp.index[0] + pd.Timedelta(days=370))].iloc[::h]   # non-overlapping
            for per, s in (("all", ic), ("early", ic[ic.index < SPLIT]), ("late", ic[ic.index >= SPLIT])):
                s = s.dropna()
                if len(s) >= 12: rows.append(dict(feature=f, h=h, period=per, dates=len(s), mean_ic=s.mean(), t=s.mean() / (s.std() / np.sqrt(len(s)))))
    return pd.DataFrame(rows)

# ---------------- C. personality ----------------
def personality(D):
    rows = []
    for t, (df, X) in D.items():
        lr = np.log(df.c).diff().iloc[WARM:].dropna(); T = len(lr); v1 = lr.var()
        r = dict(ticker=t, years=round(T / 252, 1), ac1=lr.autocorr(1), ac5=lr.autocorr(5))
        for q in (5, 20, 60):
            vq = lr.rolling(q).sum().dropna().var(); vr = vq / (q * v1)
            r[f"VR{q}"] = vr; r[f"z{q}"] = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
        y = fwd(df, 20)
        for k in (20, 60, 252):
            d = pd.concat([X[f"ret_{k}d"], y], axis=1, keys=["p", "y"]).iloc[WARM:].dropna()
            r[f"P(up|past{k}>0)"] = (d[d.p > 0].y > 0).mean(); r[f"P(up|past{k}<0)"] = (d[d.p < 0].y > 0).mean()
        rows.append(r)
    return pd.DataFrame(rows)

# ---------------- D. regimes ----------------
def regime(df, X):
    c = df.c; s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    up = (c > s200) & (s50 > s200); dn = (c < s200) & (s50 < s200)
    return pd.Series(np.where(up, "up", np.where(dn, "down", "mixed")), index=df.index)

def regime_table(D):
    rows = []
    for t, (df, X) in D.items():
        reg = regime(df, X); rv = X.rv20; vreg = pd.Series(np.where(rv > rv.rolling(756, min_periods=252).quantile(.75), "high-vol", np.where(rv < rv.rolling(756, min_periods=252).quantile(.25), "low-vol", "mid-vol")), index=df.index)
        for h in (20, 60):
            y = fwd(df, h)
            for nm, rg in (("trend", reg), ("vol", vreg)):
                for lab in rg.unique():
                    sel = (rg == lab) & y.notna(); sel.iloc[:WARM] = False
                    if sel.sum() < 100: continue
                    rows.append(dict(ticker=t, kind=nm, regime=lab, h=h, n=int(sel.sum()), mean=y[sel].mean(), hit=(y[sel] > 0).mean(), sd=y[sel].std(),
                                     base_mean=y.iloc[WARM:].mean(), base_hit=(y.iloc[WARM:].dropna() > 0).mean()))
    return pd.DataFrame(rows)

def transitions(D):
    rows = []
    for t, (df, X) in D.items():
        reg = regime(df, X)
        ev = {"enter UPTREND": (reg == "up") & (reg.shift() != "up"), "enter DOWNTREND": (reg == "down") & (reg.shift() != "down"),
              "leave UPTREND": (reg != "up") & (reg.shift() == "up")}
        for h in (20, 60):
            y = fwd(df, h); ok = y.notna().to_numpy().copy(); ok[:WARM] = False; base = y.to_numpy()[ok]
            for nm, m in ev.items():
                ix = bt.declust(m); ix = ix[ok[ix]] if len(ix) else ix
                if len(ix) < 4: continue
                s = y.to_numpy()[ix]
                rows.append(dict(ticker=t, event=nm, h=h, n=len(ix), mean=s.mean(), excess=s.mean() - base.mean(), hit=(s > 0).mean(), base_hit=(base > 0).mean()))
    return pd.DataFrame(rows)

# ---------------- E/F. panel for models ----------------
def panel(D, h=20):
    parts = []
    for t, (df, X) in D.items():
        Z = X.drop(columns="_idx").rolling(756, min_periods=756).rank(pct=True)     # within-stock, 3y trailing, no look-ahead
        Z["y"] = fwd(df, h); Z["ticker"] = t; Z["date"] = df.index
        parts.append(Z)
    P = pd.concat(parts).reset_index(drop=True)
    return P

def dumb_baselines(D, start):
    out = {}
    for name, rule in (("Buy & hold", lambda df, X: pd.Series(1.0, index=df.index)),
                       ("Above 200d", lambda df, X: (df.c > df.c.rolling(200).mean()).astype(float)),
                       ("60d return > 0", lambda df, X: (X.ret_60d > 0).astype(float)),
                       ("12-1 momentum > 0", lambda df, X: (X.mom_12_1 > 0).astype(float))):
        rs, ts = [], []
        for t, (df, X) in D.items():
            r, tg = bt.run_pos(df, rule(df, X)); m = r.index >= start; rs.append(r[m]); ts.append(tg[m])
        out[name] = (pd.concat(rs, axis=1).mean(axis=1), pd.concat(ts, axis=1).mean(axis=1))
    return out

def port_stats(r, tg):
    s = bt.stats(r, tg); return s

if __name__ == "__main__":
    D = load_all(CORE); print("core:", list(D))
    R = ic_rows(D); R.to_csv(HERE / "mom_ic_by_stock.csv", index=False)
    PI = pool_ic(R); PI.to_csv(HERE / "mom_ic_pooled.csv", index=False)
    XS = xs_ic(D); XS.to_csv(HERE / "mom_ic_crosssection.csv", index=False)
    PE = personality(D); PE.to_csv(HERE / "mom_personality.csv", index=False)
    RG = regime_table(D); RG.to_csv(HERE / "mom_regimes.csv", index=False)
    TR = transitions(D); TR.to_csv(HERE / "mom_transitions.csv", index=False)
    print("A-D done")
