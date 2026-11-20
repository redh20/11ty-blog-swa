#!/usr/bin/env python3
"""AI-stock rebound monitor. Stdlib only. Data: Yahoo Finance chart endpoint (unofficial, may be delayed).

Rebound signal, per stock (checked every run):
  1. OVERSOLD  - daily RSI(14) was below 30 at any of the last 4 readings (incl. live price)
  2. BOUNCING  - live price is at least 2% above today's low
Both true -> "REBOUND" push (once per stock per day).
Group context: when >= 40% of the watchlist is oversold at once, pushes are upgraded to high priority
(a broad washout + a per-stock bounce had the best forward returns in a backtest).
S&P 500 and SMH are shown for context only; they never trigger alerts.

Usage:
  python monitor.py            # one snapshot
  python monitor.py --loop 5   # repeat every 5 min
Env: NTFY_TOPIC (push via ntfy.sh), STATE_FILE (dedupe state, default state.json next to script).
Not financial advice.
"""
import json, os, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
CFG = {
    "off_low_pct": float(os.environ.get("MON_OFF_LOW", 2.0)),   # bounce size off day low, %
    "rsi_max": float(os.environ.get("MON_RSI_MAX", 30)),        # oversold threshold
    "rsi_lookback": 4,                                          # readings checked (today + 3 prior closes)
    "washout_frac": 0.40,                                       # share of watchlist oversold = broad washout
    "digest_after_et": (15, 55),                                # one end-of-window summary push
}
STATE_FILE = Path(os.environ.get("STATE_FILE", HERE / "state.json"))
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
            if len(rows) < 20:
                return sym, None
            closes = [r[3] for r in rows]
            m = res["meta"]
            last = m.get("regularMarketPrice", closes[-1])
            o, h, l, _ = rows[-1]
            h = max(h, m.get("regularMarketDayHigh", h)); l = min(l, m.get("regularMarketDayLow", l))
            closes[-1] = last
            rs = rsi_series(closes)
            return sym, dict(sym=sym, last=last, open=o, high=h, low=l, prev=closes[-2],
                             rsi=rs[-1], rsi_min=min(rs[-CFG["rsi_lookback"]:]), sma200=sma(closes, 200))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return sym, None

def sma(c, n):
    return sum(c[-n:]) / n if len(c) >= n else None

def rsi_series(c, n=14):
    out = [None] * len(c)
    d = [b - a for a, b in zip(c, c[1:])]
    g = sum(max(x, 0) for x in d[:n]) / n
    l = sum(max(-x, 0) for x in d[:n]) / n
    out[n] = 100.0 if l == 0 else 100 - 100 / (1 + g / l)
    for i in range(n, len(d)):
        g = (g * (n - 1) + max(d[i], 0)) / n
        l = (l * (n - 1) + max(-d[i], 0)) / n
        out[i + 1] = 100.0 if l == 0 else 100 - 100 / (1 + g / l)
    return out

def metrics(d):
    rng = d["high"] - d["low"]
    d["above_low"] = (d["last"] / d["low"] - 1) * 100
    d["range_pos"] = (d["last"] - d["low"]) / rng * 100 if rng else 0
    d["from_open"] = (d["last"] / d["open"] - 1) * 100
    d["day_chg"] = (d["last"] / d["prev"] - 1) * 100
    d["vs200"] = (d["last"] / d["sma200"] - 1) * 100 if d["sma200"] else None
    d["oversold"] = d["rsi_min"] < CFG["rsi_max"]
    d["rebound"] = d["oversold"] and d["above_low"] >= CFG["off_low_pct"]
    return d

def notify(msg, title="Stock monitor", priority="default", tags=""):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return
    try:
        h = {"Title": title, "Priority": priority}
        if tags: h["Tags"] = tags
        urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode("utf-8"), headers=h), timeout=15)
    except Exception as e:
        print("ntfy failed:", e)

def load_state():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        s = json.loads(STATE_FILE.read_text())
        if s.get("date") == today:
            return s
    except Exception:
        pass
    return {"date": today, "alerted": [], "digest": False}

def et_now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return None

def snapshot():
    lists = json.loads((HERE / "watchlists.json").read_text())
    ctx = {"SPY", "QQQ", "SMH"}                       # context only, excluded from signal counts
    syms = [s for v in lists.values() for s in v]
    with ThreadPoolExecutor(6) as ex:
        data = dict(ex.map(fetch, syms + ["^GSPC"]))
    spx = data.pop("^GSPC")
    state = load_state()
    print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    if spx:
        spx = metrics(spx)
        print(f"S&P 500 {spx['last']:,.2f} ({spx['day_chg']:+.2f}%)  {spx['above_low']:+.2f}% off day low  RSI {spx['rsi']:.0f}")
    hdr = f"{'Tkr':<6}{'Last':>9}{'Day%':>7}{'OffLow%':>9}{'Range%':>8}{'RSI14':>7}{'vs200d%':>9}  Signal"
    for name, tickers in lists.items():
        print(f"\n{name}\n{hdr}")
        for t in tickers:
            d = data.get(t)
            if not d:
                print(f"{t:<6}  (no data)"); continue
            d = metrics(d)
            v = f"{d['vs200']:>+9.1f}" if d["vs200"] is not None else f"{'n/a':>9}"
            sig = "REBOUND" if d["rebound"] else "oversold" if d["oversold"] else ""
            print(f"{t:<6}{d['last']:>9.2f}{d['day_chg']:>+7.2f}{d['above_low']:>9.2f}{d['range_pos']:>8.0f}{d['rsi']:>7.0f}{v}  {sig}")
    stocks = [d for s, d in data.items() if d and s not in ctx]
    over = [d for d in stocks if d["oversold"]]
    reb = [d for d in stocks if d["rebound"]]
    washout = len(stocks) > 0 and len(over) / len(stocks) >= CFG["washout_frac"]
    new = [d for d in reb if d["sym"] not in state["alerted"]]
    smh = data.get("SMH")
    line = f"{len(over)}/{len(stocks)} oversold" + (" (BROAD WASHOUT)" if washout else "")
    if smh: line += f" | SMH {smh['day_chg']:+.1f}%, {smh['above_low']:+.1f}% off low"
    if spx: line += f" | S&P {spx['day_chg']:+.1f}%"
    print(f"\n{line}\nRebound signals now: {', '.join(d['sym'] for d in reb) or 'none'}")

    if new:
        body = "\n".join(f"{d['sym']} {d['last']:.2f}  +{d['above_low']:.1f}% off low, RSI {d['rsi_min']:.0f}"
                         + (f", {d['vs200']:+.0f}% vs 200d" if d["vs200"] is not None else "") for d in new)
        title = f"REBOUND: {', '.join(d['sym'] for d in new[:4])}" + ("..." if len(new) > 4 else "")
        print("ALERT:", title); print(body)
        notify(body + "\n" + line, title=title, priority="urgent" if washout else "high", tags="chart_with_upwards_trend")
        state["alerted"] += [d["sym"] for d in new]

    et = et_now()
    if et and (et.hour, et.minute) >= CFG["digest_after_et"] and et.hour == 15 and not state["digest"]:
        done = ", ".join(state["alerted"]) or "none"
        notify(f"Rebounds today: {done}\n{line}", title="Stock monitor: end of window")
        print("Digest sent."); state["digest"] = True
    try:
        STATE_FILE.write_text(json.dumps(state))
    except Exception as e:
        print("state not saved:", e)

if __name__ == "__main__":
    mins = float(sys.argv[sys.argv.index("--loop") + 1]) if "--loop" in sys.argv else 0
    while True:
        snapshot()
        if not mins: break
        time.sleep(mins * 60)
