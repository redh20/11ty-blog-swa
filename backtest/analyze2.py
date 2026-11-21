import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 50); pd.set_option("display.max_rows", 300)
P3 = pd.read_csv("pooled_3.csv"); PU = pd.read_csv("pooled_universe.csv")

print("=== 1. HOW MANY SIGNALS LOOK 'SIGNIFICANT'? (all years, NVDA+GOOGL+AMZN pooled; p<0.05 in the right direction) ===")
for h in (5, 10, 20, 60):
    d = P3[(P3.h == h) & (P3.period == "all")]
    print(f"h={h:>2}: {int((d.p<0.05).sum())} of {len(d)} signals at p<0.05  (about {0.05*len(d):.1f} expected by pure chance)")

print("\n=== 2. DOES EARLY PERFORMANCE PREDICT LATER? (rank correlation of excess return, early<2013 vs late>=2013, across signals) ===")
for side in ("buy", "sell"):
    for h in (5, 10, 20, 60):
        e = P3[(P3.side == side) & (P3.h == h) & (P3.period == "early")].set_index("signal").excess
        l = P3[(P3.side == side) & (P3.h == h) & (P3.period == "late")].set_index("signal").excess
        j = pd.concat([e, l], axis=1, keys=["e", "l"]).dropna()
        print(f"{side:>4} h={h:>2}: n_signals={len(j):>2}  spearman={j.e.rank().corr(j.l.rank()):+.2f}")

print("\n=== 3. WALK-FORWARD: pick the 5 best signals on EARLY data, see what they did LATER (20-day, excess return %) ===")
for side in ("buy", "sell"):
    h = 20
    e = P3[(P3.side == side) & (P3.h == h) & (P3.period == "early")].set_index("signal")
    l = P3[(P3.side == side) & (P3.h == h) & (P3.period == "late")].set_index("signal")
    pick = e.sort_values("excess", ascending=(side == "sell")).head(5).index
    print(f"{side}: picked on early -> early excess {e.loc[pick].excess.mean()*100:+.1f}%, LATER excess {l.loc[pick].excess.mean()*100:+.1f}%   | avg of ALL signals later {l.excess.mean()*100:+.1f}%")
    print("      picks:", "; ".join(pick))

print("\n=== 4. SIGNALS THAT HOLD UP EVERYWHERE (20-day): same direction in early AND late AND on 24 other stocks, and p<0.10 on NVDA/GOOGL/AMZN ===")
rows = []
for side in ("buy", "sell"):
    sgn = 1 if side == "buy" else -1
    for sig in P3[P3.side == side].signal.unique():
        def g(df, per, h=20):
            r = df[(df.side == side) & (df.signal == sig) & (df.h == h) & (df.period == per)]
            return r.iloc[0] if len(r) else None
        a, e, l, u, ul = g(P3, "all"), g(P3, "early"), g(P3, "late"), g(PU, "all"), g(PU, "late")
        if any(x is None for x in (a, e, l, u, ul)): continue
        ok = all(np.sign(x.excess) == sgn for x in (a, e, l, u, ul)) 
        rows.append(dict(side=side, signal=sig, n3=int(a.n), ex3=a.excess*100, p3=a.p, early=e.excess*100, late=l.excess*100,
                         nU=int(u.n), exU=u.excess*100, pU=u.p, exU_late=ul.excess*100, consistent=ok))
R = pd.DataFrame(rows).round(2)
print(R[R.consistent].sort_values(["side", "ex3"], ascending=[True, False]).drop(columns="consistent").to_string(index=False))
print(f"\n({int(R.consistent.sum())} of {len(R)} signal/side combos are same-direction in all five cuts)")
R.to_csv("robustness_20d.csv", index=False)
