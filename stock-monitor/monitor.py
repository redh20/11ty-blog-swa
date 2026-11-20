#!/usr/bin/env python3
"""AI-stock bottom monitor. Stdlib only. Data: Yahoo Finance chart endpoint (unofficial, may be delayed/rate-limited).

Usage:
  python monitor.py            # one snapshot
  python monitor.py --loop 15  # repeat every 15 min (Ctrl+C to stop)
Optional push alerts: set NTFY_TOPIC (ntfy.sh) in the environment.
Not financial advice.
"""
import json, os, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
CFG = {
    "off_low_pct": 2.0,        # a name counts as "bouncing" if > this % above day's low
    "min_names": 5,            # alert when this many names are bouncing
    "spx_reclaim": (7005, 7055),
}
UA = {"User-Agent": "Mozilla/5.0"}

def fetch(sym):
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           + urllib.parse.quote(sym) + "?range=1y&interval=1d")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
                res = json.load(r)["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            rows = [(o, h, l, c) for o, h, l, c in zip(q["open"], q["high"], q["low"], q["close"]) if None not in (o, h, l, c)]
            if len(rows) < 15:
                return sym, None
            closes = [r[3] for r in rows]
            m = res["meta"]
            last = m.get("regularMarketPrice", closes[-1])
            o, h, l, _ = rows[-1]
            h = max(h, m.get("regularMarketDayHigh", h)); l = min(l, m.get("regularMarketDayLow", l))
            closes[-1] = last
            return sym, dict(sym=sym, last=last, open=o, high=h, low=l,
                             prev=closes[-2], rsi=rsi(closes), sma200=sma(closes, 200))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return sym, None

def sma(c, n):
    return sum(c[-n:]) / n if len(c) >= n else None

def rsi(c, n=14):
    d = [b - a for a, b in zip(c, c[1:])]
    g = sum(max(x, 0) for x in d[:n]) / n
    l = sum(max(-x, 0) for x in d[:n]) / n
    for x in d[n:]:
        g = (g * (n - 1) + max(x, 0)) / n
        l = (l * (n - 1) + max(-x, 0)) / n
    return 100.0 if l == 0 else 100 - 100 / (1 + g / l)

def metrics(d):
    rng = d["high"] - d["low"]
    d["above_low"] = (d["last"] - d["low"]) / d["low"] * 100
    d["range_pos"] = (d["last"] - d["low"]) / rng * 100 if rng else 0
    d["from_open"] = (d["last"] / d["open"] - 1) * 100
    d["day_chg"] = (d["last"] / d["prev"] - 1) * 100
    d["vs200"] = (d["last"] / d["sma200"] - 1) * 100 if d["sma200"] else None
    return d

def notify(msg):
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"https://ntfy.sh/{topic}", data=msg.encode(), headers={"Title": "Stock monitor"}), timeout=15)
        except Exception as e:
            print("ntfy failed:", e)

def snapshot():
    lists = json.loads((HERE / "watchlists.json").read_text())
    syms = [s for v in lists.values() for s in v]
    with ThreadPoolExecutor(6) as ex:
        data = dict(ex.map(fetch, syms + ["^GSPC"]))
    spx = data.pop("^GSPC")
    print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    if spx:
        spx = metrics(spx)
        lo, hi = CFG["spx_reclaim"]
        state = "ABOVE reclaim zone" if spx["last"] > hi else "inside zone" if spx["last"] >= lo else "below zone"
        print(f"S&P 500 {spx['last']:,.2f} ({spx['day_chg']:+.2f}%)  day low {spx['low']:,.2f}  {state} ({lo}-{hi})  vs200d {spx['vs200']:+.1f}%")
    bouncing = []
    hdr = f"{'Tkr':<6}{'Last':>9}{'Day%':>7}{'AbvLow%':>9}{'Range%':>8}{'FromOpen%':>10}{'RSI14':>7}{'vs200d%':>9}"
    for name, tickers in lists.items():
        print(f"\n{name}\n{hdr}")
        for t in tickers:
            d = data.get(t)
            if not d:
                print(f"{t:<6}  (no data)"); continue
            d = metrics(d)
            if d["above_low"] > CFG["off_low_pct"]:
                bouncing.append(t)
            v = f"{d['vs200']:>+9.1f}" if d["vs200"] is not None else f"{'n/a':>9}"
            print(f"{t:<6}{d['last']:>9.2f}{d['day_chg']:>+7.2f}{d['above_low']:>9.2f}{d['range_pos']:>8.0f}{d['from_open']:>+10.2f}{d['rsi']:>7.0f}{v}")
    alerts = []
    if len(bouncing) >= CFG["min_names"]:
        alerts.append(f"{len(bouncing)} names >{CFG['off_low_pct']}% off day lows: {', '.join(bouncing)}")
    if spx and spx["last"] > CFG["spx_reclaim"][1]:
        alerts.append(f"S&P 500 reclaimed {CFG['spx_reclaim'][1]}: {spx['last']:,.2f}")
    print("\nBouncing (>%.1f%% off low): %s" % (CFG["off_low_pct"], ", ".join(bouncing) or "none"))
    for a in alerts:
        print("ALERT:", a); notify(a)

if __name__ == "__main__":
    mins = float(sys.argv[sys.argv.index("--loop") + 1]) if "--loop" in sys.argv else 0
    while True:
        snapshot()
        if not mins: break
        time.sleep(mins * 60)
