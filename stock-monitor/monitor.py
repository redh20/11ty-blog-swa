#!/usr/bin/env python3
"""AI-stock rebound monitor. Stdlib only. Data: Yahoo Finance chart endpoint (unofficial, may be delayed).

Rebound signal, per stock (checked every run):
  1. OVERSOLD  - daily RSI(14) was below 30 at any of the last 4 readings (incl. live price)
  2. BOUNCING  - live price is at least 2% above today's low
Both true -> "REBOUND" push (once per stock per day).
Group context: when >= 40% of the watchlist is oversold at once, pushes are upgraded to high priority
(a broad washout + a per-stock bounce had the best forward returns in a backtest).
S&P 500 and SMH are shown for context only; they never trigger alerts.

Awareness alerts (information, not a bottom call; each fires once per stock per day):
  VOLUME  - volume so far today >= 2x the 20-day average full-day volume
  MOVE    - day change of 5% or more (re-alerts when it reaches the next 5% step: 10%, 15%...)
  RECLAIM - price crosses back above its 50-day or 200-day average (yesterday's level)
  EARNINGS- a stock reports today or on the next trading day (needs a free ALPHAVANTAGE_KEY)

Usage:
  python monitor.py            # one snapshot
  python monitor.py --loop 5   # repeat every 5 min
Env: NTFY_TOPIC (push via ntfy.sh), ALPHAVANTAGE_KEY (earnings dates), STATE_FILE (dedupe state, default state.json next to script).
Not financial advice.
"""
import csv, io, json, os, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).parent
CFG = {
    "off_low_pct": float(os.environ.get("MON_OFF_LOW", 2.0)),   # bounce size off day low, %
    "rsi_max": float(os.environ.get("MON_RSI_MAX", 30)),        # oversold threshold
    "rsi_lookback": 4,                                          # readings checked (today + 3 prior closes)
    "washout_frac": 0.40,                                       # share of watchlist oversold = broad washout
    "digest_after_et": (15, 55),                                # one end-of-window summary push
    "vol_mult": float(os.environ.get("MON_VOL_MULT", 2.0)),     # volume so far vs 20d avg full-day volume
    "move_pct": float(os.environ.get("MON_MOVE", 5.0)),         # big-move step, % on the day
    "reclaim_smas": (50, 200),
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
            rows = [(o, h, l, c, v or 0) for o, h, l, c, v in zip(q["open"], q["high"], q["low"], q["close"], q["volume"]) if None not in (o, h, l, c)]
            if len(rows) < 20:
                return sym, None
            closes = [r[3] for r in rows]
            vols = [r[4] for r in rows]
            m = res["meta"]
            last = m.get("regularMarketPrice", closes[-1])
            o, h, l, _, _ = rows[-1]
            h = max(h, m.get("regularMarketDayHigh", h)); l = min(l, m.get("regularMarketDayLow", l))
            prior = closes[:-1]                                   # closes before today
            reclaims = [n for n in CFG["reclaim_smas"] if len(prior) >= n and prior[-1] <= sma(prior, n) < last]
            avg_vol = sum(vols[-21:-1]) / 20 if len(vols) >= 21 else 0
            today_vol = m.get("regularMarketVolume") or vols[-1]
            closes[-1] = last
            rs = rsi_series(closes)
            return sym, dict(sym=sym, last=last, open=o, high=h, low=l, prev=closes[-2],
                             rsi=rs[-1], rsi_min=min(rs[-CFG["rsi_lookback"]:]), sma200=sma(closes, 200),
                             vol_ratio=(today_vol / avg_vol) if avg_vol else 0, reclaims=reclaims)
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
    d["unusual_vol"] = d["vol_ratio"] >= CFG["vol_mult"]
    d["move_tier"] = int(abs(d["day_chg"]) // CFG["move_pct"] * CFG["move_pct"]) if abs(d["day_chg"]) >= CFG["move_pct"] else 0
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
    return {"date": today, "sent": [], "digest": False}

def et_now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return None

def parse_earnings(text, syms):
    """Alpha Vantage EARNINGS_CALENDAR csv -> [[symbol, reportDate, timeOfTheDay], ...] for our symbols."""
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        if r.get("symbol") in syms:
            out.append([r["symbol"], r["reportDate"], r.get("timeOfTheDay", "")])
    return sorted(out, key=lambda x: x[1])

def next_weekday(d):
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d

def earnings_due(items, today):
    """Reports today or on the next trading day (weekends skipped; exchange holidays ignored)."""
    nxt = next_weekday(today)
    when = {"pre-market": "before the open", "post-market": "after the close"}
    out = []
    for sym, date, tod in items:
        if date == today.isoformat():
            out.append((sym, date, f"reports TODAY, {when.get(tod, 'time not listed')}"))
        elif date == nxt.isoformat():
            day = "tomorrow" if nxt == today + timedelta(days=1) else "next trading day"
            out.append((sym, date, f"reports {day}, {when.get(tod, 'time not listed')}"))
    return out

def load_earnings(state, syms):
    key = os.environ.get("ALPHAVANTAGE_KEY")
    if not key:
        return None
    if "earnings" in state:
        return state["earnings"]
    if state.get("earn_tries", 0) >= 3:
        return None
    state["earn_tries"] = state.get("earn_tries", 0) + 1
    try:
        url = "https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey=" + urllib.parse.quote(key)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            text = r.read().decode("utf-8", "replace")
        if text.lstrip().startswith("{") or "symbol" not in text[:40]:
            print("earnings: calendar not available (rate limit or bad key)"); return None
        state["earnings"] = parse_earnings(text, set(syms))
        return state["earnings"]
    except Exception as e:
        print("earnings fetch failed:", e); return None

def today_et():
    n = et_now() or datetime.now()
    return n.date()

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
    hdr = f"{'Tkr':<6}{'Last':>9}{'Day%':>7}{'OffLow%':>9}{'Range%':>8}{'RSI14':>7}{'VolX':>6}{'vs200d%':>9}  Signal"
    for name, tickers in lists.items():
        print(f"\n{name}\n{hdr}")
        for t in tickers:
            d = data.get(t)
            if not d:
                print(f"{t:<6}  (no data)"); continue
            d = metrics(d)
            v = f"{d['vs200']:>+9.1f}" if d["vs200"] is not None else f"{'n/a':>9}"
            sig = " ".join(x for x in ["REBOUND" if d["rebound"] else "oversold" if d["oversold"] else "",
                                       "VOLUME" if d["unusual_vol"] else "", "MOVE" if d["move_tier"] else "",
                                       "RECLAIM" + "/".join(map(str, d["reclaims"])) if d["reclaims"] else ""] if x)
            print(f"{t:<6}{d['last']:>9.2f}{d['day_chg']:>+7.2f}{d['above_low']:>9.2f}{d['range_pos']:>8.0f}{d['rsi']:>7.0f}{d['vol_ratio']:>6.1f}{v}  {sig}")
    stocks = [d for s, d in data.items() if d and s not in ctx]
    over = [d for d in stocks if d["oversold"]]
    reb = [d for d in stocks if d["rebound"]]
    washout = len(stocks) > 0 and len(over) / len(stocks) >= CFG["washout_frac"]
    sent = state["sent"]
    def fresh(key):
        if key in sent: return False
        sent.append(key); return True
    new = [d for d in reb if fresh("rebound:" + d["sym"])]
    smh = data.get("SMH")
    line = f"{len(over)}/{len(stocks)} oversold" + (" (BROAD WASHOUT)" if washout else "")
    if smh: line += f" | SMH {smh['day_chg']:+.1f}%, {smh['above_low']:+.1f}% off low"
    if spx: line += f" | S&P {spx['day_chg']:+.1f}%"
    print(f"\n{line}\nRebound signals now: {', '.join(d['sym'] for d in reb) or 'none'}")

    def push(kind, lines, title, priority="default", tags=""):
        print("ALERT:", title)
        for x in lines: print("  " + x)
        notify("\n".join(lines) + "\n" + line, title=title, priority=priority, tags=tags)
    def head(items):
        return ", ".join(items[:4]) + ("..." if len(items) > 4 else "")

    if new:
        push("rebound", [f"{d['sym']} {d['last']:.2f}  +{d['above_low']:.1f}% off low, RSI {d['rsi_min']:.0f}"
                         + (f", {d['vs200']:+.0f}% vs 200d" if d["vs200"] is not None else "") for d in new],
             f"REBOUND: {head([d['sym'] for d in new])}", "urgent" if washout else "high", "chart_with_upwards_trend")
    vol = [d for d in stocks if d["unusual_vol"] and fresh("volume:" + d["sym"])]
    if vol:
        push("volume", [f"{d['sym']} volume {d['vol_ratio']:.1f}x normal (day {d['day_chg']:+.1f}%)" for d in vol],
             f"VOLUME: {head([d['sym'] for d in vol])}", tags="bar_chart")
    mv = [d for d in stocks if d["move_tier"] and fresh(f"move:{d['sym']}:{'up' if d['day_chg'] > 0 else 'dn'}{d['move_tier']}")]
    if mv:
        push("move", [f"{d['sym']} {d['day_chg']:+.1f}% on the day at {d['last']:.2f}" for d in mv],
             f"MOVE: {head([d['sym'] for d in mv])}", tags="rotating_light")
    rc = [(d, n) for d in stocks for n in d["reclaims"] if fresh(f"reclaim:{d['sym']}:{n}")]
    if rc:
        push("reclaim", [f"{d['sym']} back above its {n}-day average ({d['last']:.2f})" for d, n in rc],
             f"RECLAIM: {head([d['sym'] for d, n in rc])}", tags="white_check_mark")
    items = load_earnings(state, [d["sym"] for d in stocks])
    if items is None and not os.environ.get("ALPHAVANTAGE_KEY") and "earn_note" not in state:
        print("earnings: skipped (set ALPHAVANTAGE_KEY to enable earnings warnings)"); state["earn_note"] = True
    if items:
        nxt14 = [x for x in items if x[1] <= (today_et() + timedelta(days=14)).isoformat()]
        if nxt14: print("Upcoming earnings (14d): " + ", ".join(f"{a} {b[5:]}" for a, b, c in nxt14))
        due = [(a, b, c) for a, b, c in earnings_due(items, today_et()) if fresh(f"earnings:{a}:{b}:{'today' if b == today_et().isoformat() else 'soon'}")]
        if due:
            push("earnings", [f"{a} {c}" for a, b, c in due], f"EARNINGS: {head([a for a, b, c in due])}", "high", "calendar")

    et = et_now()
    if et and (et.hour, et.minute) >= CFG["digest_after_et"] and et.hour == 15 and not state["digest"]:
        def of(kind): return ", ".join(sorted({k.split(":")[1] for k in sent if k.startswith(kind + ":")})) or "none"
        notify(f"Rebounds: {of('rebound')}\nVolume: {of('volume')}\nBig moves: {of('move')}\nReclaims: {of('reclaim')}\n{line}",
               title="Stock monitor: end of window")
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
