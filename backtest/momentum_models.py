#!/usr/bin/env python3
"""Parts E/F of the momentum study: rule mining (decision tree) and walk-forward models."""
import numpy as np, pandas as pd
from momentum_study import *

D = load_all(CORE); HD = load_all(HOLD)
P = panel(D); PH = panel(HD)
feat = [c for c in P.columns if c not in ("y", "ticker", "date")]
P = P.dropna(subset=feat).reset_index(drop=True); PH = PH.dropna(subset=feat).reset_index(drop=True)
print(f"core rows {len(P)}  holdout rows {len(PH)}  features {len(feat)}  core dates {P.date.min().date()}->{P.date.max().date()}")
PURGE = pd.Timedelta(days=32)

# ---------------- E. decision-tree rule mining ----------------
tr = P[(P.date < SPLIT - PURGE) & P.y.notna()]
print(f"\n=== E. DECISION TREE rules mined on core stocks BEFORE 2016 ({len(tr)} rows), tested on 2016+ and on {len(HOLD)} other stocks ===")
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=400, random_state=0).fit(tr[feat], tr.y > 0)
print(export_text(tree, feature_names=feat, decimals=2))
def leafstats(df, name):
    d = df[df.y.notna()].copy(); d["leaf"] = tree.apply(d[feat]); g = d.groupby("leaf").y.agg(n="size", up=lambda s: (s > 0).mean(), mean="mean")
    g.columns = [f"{name}_{c}" for c in g.columns]; return g
tab = pd.concat([leafstats(tr, "train"), leafstats(P[P.date >= SPLIT], "core2016+"), leafstats(PH[PH.date >= SPLIT], "hold2016+")], axis=1)
base = {n: (d[d.y.notna()].y > 0).mean() for n, d in (("train", tr), ("core2016+", P[P.date >= SPLIT]), ("hold2016+", PH[PH.date >= SPLIT]))}
print("base up-rate:", {k: round(v, 3) for k, v in base.items()})
print(tab.round(3).to_string())
tab.to_csv(HERE / "mom_tree_leaves.csv")

# ---------------- F. walk-forward models ----------------
def wf(panel_list, years=range(2012, 2027)):
    outs = {n: [] for n, _ in panel_list}
    for Y in years:
        cut = pd.Timestamp(f"{Y}-01-01"); end = pd.Timestamp(f"{Y+1}-01-01")
        trn = P[(P.date < cut - PURGE) & P.y.notna()]
        if len(trn) < 5000: continue
        thr = (trn.y > 0).mean()
        lr = LogisticRegression(C=0.05, max_iter=300).fit(trn[feat], trn.y > 0)
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120, min_samples_leaf=300, l2_regularization=1.0, random_state=0).fit(trn[feat], trn.y > 0)
        for n, pan in panel_list:
            te = pan[(pan.date >= cut) & (pan.date < end)].copy()
            if len(te) == 0: continue
            te["p_lr"] = lr.predict_proba(te[feat])[:, 1]; te["p_gb"] = gb.predict_proba(te[feat])[:, 1]; te["thr"] = thr; te["year"] = Y
            outs[n].append(te[["date", "ticker", "y", "p_lr", "p_gb", "thr", "year"]])
    return {n: pd.concat(v) for n, v in outs.items()}
W = wf([("core", P), ("hold", PH)])
for n, w in W.items(): w.to_csv(HERE / f"mom_wf_preds_{n}.csv", index=False)

def metrics(w, tag):
    d = w[w.y.notna()]; rows = []
    for m in ("p_lr", "p_gb"):
        base = (d.y > 0).mean(); acc = ((d[m] > d.thr) == (d.y > 0)).mean()
        rows.append(dict(set=tag, model=m, rows=len(d), AUC=roc_auc_score(d.y > 0, d[m]), IC=sps.spearmanr(d[m], d.y)[0],
                         acc_at_thr=acc, always_up_acc=base,
                         top_q_mean=d[d[m] >= d[m].quantile(.8)].y.mean(), bot_q_mean=d[d[m] <= d[m].quantile(.2)].y.mean(), all_mean=d.y.mean(),
                         top_q_up=(d[d[m] >= d[m].quantile(.8)].y > 0).mean(), bot_q_up=(d[d[m] <= d[m].quantile(.2)].y > 0).mean()))
    return rows
M = pd.DataFrame(metrics(W["core"], "core 11 stocks") + metrics(W["hold"], "hold-out 22 stocks"))
print("\n=== F1. WALK-FORWARD PREDICTION QUALITY (each year predicted by a model trained only on earlier years) ===")
print("AUC 0.50 = coin flip. IC 0 = no rank relationship. Quintile = top/bottom 20% of model confidence; fwd = 20-day return.")
print(M.round(3).to_string(index=False)); M.to_csv(HERE / "mom_wf_metrics.csv", index=False)
print("\nrank IC by year (core, logistic | gradient boosting):")
yy = W["core"][W["core"].y.notna()].groupby("year").apply(lambda d: pd.Series(dict(lr=sps.spearmanr(d.p_lr, d.y)[0], gb=sps.spearmanr(d.p_gb, d.y)[0])))
print(yy.round(3).T.to_string())
print(f"years with positive IC (core): lr {(yy.lr>0).sum()}/{len(yy)}, gb {(yy.gb>0).sum()}/{len(yy)}")

# ---------------- F2. strategy test ----------------
def strat(Dd, w, model, delta, start):
    rs, ts = [], []
    for t, (df, X) in Dd.items():
        d = w[w.ticker == t].drop_duplicates("date").set_index("date")
        if d.empty: continue
        pos = (d[model] > d.thr + delta).astype(float).reindex(df.index).fillna(0.0)
        r, tg = bt.run_pos(df, pos); m = r.index >= start; keep = m & (r.index >= d.index.min())
        rs.append(r[keep]); ts.append(tg[keep])
    return pd.concat(rs, axis=1).mean(axis=1), pd.concat(ts, axis=1).mean(axis=1)
out = []
for tag, Dd, w in (("core 11", D, W["core"]), ("hold-out 22", HD, W["hold"])):
    start = pd.Timestamp("2012-01-01")
    base = dumb_baselines(Dd, start)
    cand = {k: v for k, v in base.items()}
    for model in ("p_lr", "p_gb"):
        for delta in (0.0, 0.05):
            cand[f"Model {model[2:].upper()} long if p > base+{delta:.2f}"] = strat(Dd, w, model, delta, start)
    for name, (r, tg) in cand.items():
        s = bt.stats(r.dropna(), tg.reindex(r.index).fillna(0))
        if s: out.append(dict(set=tag, strategy=name, **s))
S = pd.DataFrame(out); S.to_csv(HERE / "mom_wf_strategies.csv", index=False)
print("\n=== F2. STRATEGY TEST, 2012-2026, equal-weight portfolio, 0.1% cost, trade at next open ===")
for tag in S.set.unique():
    d = S[S.set == tag].copy(); d["x"] = d.final.round(2); d["CAGR%"] = (d.cagr * 100).round(1); d["maxDD%"] = (d.maxdd * 100).round(0); d["expo%"] = (d.exposure * 100).round(0); d["sharpe"] = d.sharpe.round(2)
    print(f"--- {tag}"); print(d[["strategy", "x", "CAGR%", "sharpe", "maxDD%", "expo%"]].to_string(index=False))
