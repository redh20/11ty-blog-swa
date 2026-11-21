import pandas as pd, numpy as np
from statistics import NormalDist
nd = NormalDist()
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 52); pd.set_option("display.max_rows", 200)
M = pd.read_csv("signals_main.csv"); U = pd.read_csv("signals_universe.csv")
T3 = ["NVDA", "GOOGL", "AMZN"]

def pool(df):
    g = []
    for (side, sig, h, per), d in df.groupby(["side", "signal", "h", "period"]):
        n = d.n.sum()
        w = d.n / n
        z = [nd.inv_cdf(min(max(1 - p, 1e-6), 1 - 1e-6)) for p in d.p.fillna(.5)]
        zc = sum(zz * np.sqrt(nn) for zz, nn in zip(z, d.n)) / np.sqrt(d.n.sum())
        g.append(dict(side=side, signal=sig, h=h, period=per, n=n, tickers=len(d),
                      excess=(d.excess * w).sum(), mean=(d["mean"] * w).sum(), hit=(d.hit * w).sum(), base_hit=(d.base_hit * w).sum(),
                      p=1 - nd.cdf(zc), same_sign=int((np.sign(d.excess) == (1 if side == "buy" else -1)).sum())))
    return pd.DataFrame(g)

P3 = pool(M[M.ticker.isin(T3)]); PU = pool(U)
P3.to_csv("pooled_3.csv", index=False); PU.to_csv("pooled_universe.csv", index=False)

print("=== BASELINES (all eligible days, after 1y warmup): mean fwd return and % positive ===")
b = M[(M.period == "all")].groupby(["ticker", "h"]).first()[["base_mean", "base_hit"]].unstack("h")
print((b * 100).round(1))

def show(side, h, per="all", top=8, df=P3):
    d = df[(df.side == side) & (df.h == h) & (df.period == per)].copy()
    d = d.sort_values("excess", ascending=(side == "sell")).head(top)
    d["excess%"] = (d.excess * 100).round(1); d["mean%"] = (d["mean"] * 100).round(1)
    d["hit%"] = (d.hit * 100).round(0); d["base%"] = (d.base_hit * 100).round(0); d["p"] = d.p.round(3)
    return d[["signal", "n", "mean%", "excess%", "hit%", "base%", "p", "same_sign"]].to_string(index=False)

for h in (20, 60):
    print(f"\n=== BUY signals, NVDA+GOOGL+AMZN pooled, {h}-day forward, ALL years (top by excess return) ===\n" + show("buy", h))
    print(f"\n=== SELL signals, {h}-day forward, ALL years (most negative excess = best exits) ===\n" + show("sell", h))
