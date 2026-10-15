"""
Gold (XAUUSD) Macro Intelligence Dashboard  -  v2 (signals)
------------------------------------------------------------
Run with:  streamlit run app_v2.py --theme.base dark

Requires: streamlit>=1.37, yfinance, pandas, numpy, plyer

What's new vs v1:
  * Symmetric macro score (-4 .. +4)
  * Gold trend filter (EMA20 / EMA50) + ATR-based stop & target
  * Multi-refresh confirmation, cooldown, session filter, news blackout
  * Signal log (signals_log.csv) with automatic outcome tracking
  * Desktop notifications on every new confirmed signal

Educational tool only - not financial advice.
"""

import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
REFRESH_SECONDS = 300           # 5 minutes
STALE_AFTER_MINUTES = 30        # macro feed older than this is ignored in scoring
GOLD_MAX_AGE_MIN = 25           # gold bar older than this -> no signals
VIX_HI, VIX_LO = 20.0, 15.0     # VIX > 20 = risk-off (+1), < 15 = complacent (-1)
MAX_SCORE = 4
COOLDOWN_MIN = 30               # minimum gap between logged signals
EXPIRY_HOURS = 4                # open signals are closed after this long
BLACKOUT_BEFORE_MIN = 30
BLACKOUT_AFTER_MIN = 60
MIN_LIVE_FEEDS = 3              # of the 4 macro drivers

ASSETS = {
    "GOLD": ("GC=F", "Gold Futures", 2),
    "DXY": ("DX-Y.NYB", "US Dollar Index", 3),
    "TNX": ("^TNX", "10Y Treasury Yield", 3),
    "VIX": ("^VIX", "VIX Volatility", 2),
    "ES": ("ES=F", "S&P 500 Futures", 2),
}

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

LOG_PATH = Path(__file__).with_name("signals_log.csv")
LOG_COLS = [
    "id", "time_utc", "side", "entry", "stop", "target", "net_score", "atr",
    "entry_bar_utc", "status", "exit_price", "r_result", "closed_utc",
    "px_15m", "px_30m", "px_60m",
]
FLOAT_COLS = ["entry", "stop", "target", "atr", "exit_price", "r_result",
              "px_15m", "px_30m", "px_60m"]
OBJ_COLS = ["time_utc", "side", "entry_bar_utc", "status", "closed_utc"]

st.set_page_config(page_title="Gold Macro Signals", page_icon="🥇", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #0e1117; color: #e6e6e6; }
        div[data-testid="stMetric"] {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 12px; padding: 18px 20px;
        }
        div[data-testid="stMetricValue"] { font-size: 2.2rem; }
        .status-bar {
            background: #161b22; border: 1px solid #30363d; border-radius: 10px;
            padding: 10px 16px; font-size: 0.95rem; color: #c9d1d9;
        }
        h1, h2, h3 { color: #f5c542; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Desktop notifications (runs on the machine hosting Streamlit)
# ----------------------------------------------------------------------------
def desktop_notify(title: str, message: str) -> bool:
    try:
        from plyer import notification
        notification.notify(title=title, message=message,
                            app_name="Gold Macro Signals", timeout=15)
        return True
    except Exception:
        pass
    try:
        system = platform.system()
        if system == "Darwin":
            safe_t, safe_m = title.replace('"', "'"), message.replace('"', "'")
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{safe_m}" with title "{safe_t}" sound name "Glass"'],
                check=True, timeout=10)
            return True
        if system == "Linux":
            subprocess.run(["notify-send", title, message], check=True, timeout=10)
            return True
    except Exception:
        pass
    return False


# ----------------------------------------------------------------------------
# Market session (CME Globex, approximate; holidays not modelled)
# ----------------------------------------------------------------------------
def is_market_open(now_et: datetime) -> bool:
    wd = now_et.weekday()
    t = now_et.hour + now_et.minute / 60
    if wd == 5:
        return False
    if wd == 4 and t >= 17:
        return False
    if wd == 6 and t < 18:
        return False
    if wd in (0, 1, 2, 3) and 17 <= t < 18:
        return False
    return True


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def fetch_asset(symbol: str):
    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="5m")
        hist = hist.dropna(subset=["Close"])
        if len(hist) < 2:
            return None
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize("UTC")
        else:
            hist.index = hist.index.tz_convert("UTC")
        last, prev = float(hist["Close"].iloc[-1]), float(hist["Close"].iloc[-2])
        change = last - prev
        pct = (change / prev * 100) if prev else 0.0
        return {"last": last, "prev": prev, "change": change, "pct": pct,
                "bar_time": hist.index[-1], "hist": hist}
    except Exception:
        return None


def fetch_all() -> dict:
    keys = list(ASSETS.keys())
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        results = list(pool.map(lambda k: fetch_asset(ASSETS[k][0]), keys))
    return dict(zip(keys, results))


def add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


# ----------------------------------------------------------------------------
# Scoring (symmetric: -4 .. +4, positive = bullish gold)
# ----------------------------------------------------------------------------
def sgn(x) -> int:
    return int(np.sign(x))


def build_scorecard(data: dict):
    rules = [
        ("DXY", "Dollar falling +1 / rising -1", lambda d: -sgn(d["change"])),
        ("TNX", "Yields falling +1 / rising -1", lambda d: -sgn(d["change"])),
        ("VIX", f"VIX > {VIX_HI:.0f} +1 / < {VIX_LO:.0f} -1",
         lambda d: 1 if d["last"] > VIX_HI else (-1 if d["last"] < VIX_LO else 0)),
        ("ES", "S&P falling +1 / rising -1", lambda d: -sgn(d["change"])),
    ]
    now_utc = datetime.now(UTC)
    rows, total, live = [], 0, 0

    for key, rule, fn in rules:
        symbol, name, dec = ASSETS[key]
        d = data.get(key) if data else None
        if d is None:
            rows.append({"Driver": name, "Ticker": symbol, "Live Value": "N/A",
                         "5m Move": "N/A", "5m %": "N/A", "Rule": rule,
                         "Data": "❌ No data", "Score": 0})
            continue

        age_min = (now_utc - d["bar_time"]).total_seconds() / 60
        stale = age_min > STALE_AFTER_MINUTES
        pts = 0 if stale else fn(d)
        if not stale:
            live += 1
            total += pts
        arrow = "▲" if d["change"] > 0 else ("▼" if d["change"] < 0 else "■")
        rows.append({
            "Driver": name, "Ticker": symbol,
            "Live Value": f"{d['last']:,.{dec}f}",
            "5m Move": f"{arrow} {d['change']:+.{dec}f}",
            "5m %": f"{d['pct']:+.3f}%",
            "Rule": rule,
            "Data": "🟡 Stale (ignored)" if stale else "🟢 Live",
            "Score": pts,
        })
    return pd.DataFrame(rows), total, live


def classify(score: int):
    conf = abs(score) / MAX_SCORE * 100
    if conf >= 75:
        tier = "High Conviction"
    elif conf >= 50:
        tier = "Moderate Conviction"
    else:
        tier = "Low Conviction / Chop"
    if score >= 2:
        bias = "🟢 Bullish"
    elif score <= -2:
        bias = "🔴 Bearish"
    else:
        bias = "🟡 Neutral"
    return bias, conf, tier


# ----------------------------------------------------------------------------
# News blackout parsing
# ----------------------------------------------------------------------------
def parse_events(text: str):
    events, bad = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            dt = datetime.strptime(line[:16], "%Y-%m-%d %H:%M").replace(tzinfo=ET)
            events.append(dt)
        except ValueError:
            bad += 1
    return events, bad


# ----------------------------------------------------------------------------
# Signal engine
# ----------------------------------------------------------------------------
def evaluate_signal(score, live, gold, market_open, now_et, now_utc, events):
    no = lambda reason: {"side": "NO TRADE", "reason": reason}
    ss = st.session_state

    if gold is None or len(gold) < 60:
        return no("Not enough gold data yet")
    if not market_open:
        return no("Futures market closed")
    age = (now_utc - gold.index[-1]).total_seconds() / 60
    if age > GOLD_MAX_AGE_MIN:
        return no(f"Gold feed is {age:.0f} min old")
    if live < MIN_LIVE_FEEDS:
        return no(f"Only {live}/4 macro feeds are live")
    if ss.session_filter and not (3 <= now_et.hour < 16):
        return no("Outside London/NY session (03:00-16:00 ET)")
    for ev in events:
        if ev - timedelta(minutes=BLACKOUT_BEFORE_MIN) <= now_et <= ev + timedelta(minutes=BLACKOUT_AFTER_MIN):
            return no(f"News blackout around {ev:%H:%M} ET event")

    last = gold.iloc[-1]
    up = last["Close"] > last["ema20"] > last["ema50"]
    down = last["Close"] < last["ema20"] < last["ema50"]

    if score >= ss.min_score and up:
        side, sign = "LONG", 1
    elif score <= -ss.min_score and down:
        side, sign = "SHORT", -1
    elif abs(score) >= ss.min_score:
        return no("Macro is strong but gold trend (EMA20/50) doesn't confirm")
    else:
        return no(f"Net score {score:+d} is below the +/-{ss.min_score} threshold")

    entry = float(last["Close"])
    risk = ss.atr_mult * float(last["atr"])
    return {
        "side": side, "reason": "All conditions met",
        "entry": entry,
        "stop": entry - sign * risk,
        "target": entry + sign * ss.rr * risk,
        "atr": float(last["atr"]),
        "entry_bar_utc": gold.index[-1].isoformat(),
    }


def confirmed_side(history, n):
    if len(history) < n:
        return "NO TRADE"
    last = history[-n:]
    return last[0] if len(set(last)) == 1 and last[0] != "NO TRADE" else "NO TRADE"


# ----------------------------------------------------------------------------
# Signal log + outcome tracking
# ----------------------------------------------------------------------------
def load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        try:
            df = pd.read_csv(LOG_PATH)
        except Exception:
            df = pd.DataFrame(columns=LOG_COLS)
    else:
        df = pd.DataFrame(columns=LOG_COLS)
    for c in LOG_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[LOG_COLS].copy()
    df[FLOAT_COLS] = df[FLOAT_COLS].astype(float)
    df[OBJ_COLS] = df[OBJ_COLS].astype(object)
    return df.reset_index(drop=True)


def save_log(df: pd.DataFrame):
    try:
        df.to_csv(LOG_PATH, index=False)
    except Exception as e:
        st.warning(f"Could not write {LOG_PATH.name}: {e}")


def resolve_outcomes(log: pd.DataFrame, gold: pd.DataFrame, now_utc: datetime) -> pd.DataFrame:
    for idx in log.index:
        row = log.loc[idx]
        entry_bar = pd.to_datetime(row["entry_bar_utc"], utc=True)
        bars = gold[gold.index > entry_bar]

        # Price N minutes after the signal (bar open at that time)
        for mins, col in ((15, "px_15m"), (30, "px_30m"), (60, "px_60m")):
            if pd.isna(row[col]):
                t_target = entry_bar + timedelta(minutes=mins)
                cand = bars[bars.index >= t_target]
                if not cand.empty and cand.index[0] <= t_target + timedelta(minutes=10):
                    log.at[idx, col] = float(cand.iloc[0]["Open"])

        if row["status"] != "OPEN":
            continue

        side, entry = row["side"], float(row["entry"])
        stop, target = float(row["stop"]), float(row["target"])
        risk = abs(entry - stop) or 1e-9
        rr = abs(target - entry) / risk
        resolved = False

        for t, bar in bars.iterrows():
            if side == "LONG":
                hit_stop, hit_tgt = bar["Low"] <= stop, bar["High"] >= target
            else:
                hit_stop, hit_tgt = bar["High"] >= stop, bar["Low"] <= target
            if hit_stop:  # conservative: stop wins if both hit in the same bar
                log.at[idx, "status"], log.at[idx, "exit_price"] = "LOSS", stop
                log.at[idx, "r_result"], log.at[idx, "closed_utc"] = -1.0, t.isoformat()
                resolved = True
                break
            if hit_tgt:
                log.at[idx, "status"], log.at[idx, "exit_price"] = "WIN", target
                log.at[idx, "r_result"], log.at[idx, "closed_utc"] = rr, t.isoformat()
                resolved = True
                break

        if not resolved:
            opened = pd.to_datetime(row["time_utc"], utc=True)
            if now_utc - opened >= timedelta(hours=EXPIRY_HOURS):
                exit_px = float(bars["Close"].iloc[-1]) if not bars.empty else entry
                sign = 1 if side == "LONG" else -1
                log.at[idx, "status"], log.at[idx, "exit_price"] = "EXPIRED", exit_px
                log.at[idx, "r_result"] = sign * (exit_px - entry) / risk
                log.at[idx, "closed_utc"] = now_utc.isoformat()
    return log


def maybe_log_new_signal(raw: dict, score: int, now_utc: datetime) -> bool:
    ss = st.session_state
    log = ss.log
    if (log["status"] == "OPEN").any():
        return False
    if not log.empty:
        last_t = pd.to_datetime(log["time_utc"], utc=True).max()
        if now_utc - last_t < timedelta(minutes=COOLDOWN_MIN):
            return False
    new_id = int(log["id"].max()) + 1 if not log.empty and log["id"].notna().any() else 1
    row = {
        "id": new_id, "time_utc": now_utc.isoformat(), "side": raw["side"],
        "entry": raw["entry"], "stop": raw["stop"], "target": raw["target"],
        "net_score": score, "atr": raw["atr"], "entry_bar_utc": raw["entry_bar_utc"],
        "status": "OPEN", "exit_price": np.nan, "r_result": np.nan, "closed_utc": np.nan,
        "px_15m": np.nan, "px_30m": np.nan, "px_60m": np.nan,
    }
    log.loc[len(log)] = [row[c] for c in LOG_COLS]
    ss.log = log
    save_log(log)
    return True


def process_signals(market_open: bool, now_et: datetime, counts_for_confirmation: bool):
    ss = st.session_state
    now_utc = datetime.now(UTC)
    _, score, live = build_scorecard(ss.data)
    ss.score, ss.live = score, live
    gold = ss.gold_df

    if gold is not None:
        ss.log = resolve_outcomes(ss.log, gold, now_utc)
        save_log(ss.log)

    raw = evaluate_signal(score, live, gold, market_open, now_et, now_utc, ss.events)
    ss.raw = raw

    if counts_for_confirmation:
        ss.raw_history = (ss.raw_history + [raw["side"]])[-10:]
        ss.last_confirm_ts = time.time()
    conf = confirmed_side(ss.raw_history, ss.confirm_n)
    ss.confirmed = conf

    if counts_for_confirmation and conf != "NO TRADE" and raw["side"] == conf:
        if maybe_log_new_signal(raw, score, now_utc):
            msg = (f"{raw['side']} Gold @ {raw['entry']:.2f} | "
                   f"SL {raw['stop']:.2f} | TP {raw['target']:.2f}")
            st.toast(f"🚨 {msg}", icon="🥇")
            if ss.notify_on:
                if not desktop_notify(f"Gold {raw['side']} signal", msg):
                    st.toast("Desktop notification failed - is plyer installed?", icon="⚠️")


# ----------------------------------------------------------------------------
# Sidebar settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Signal Settings")
    st.slider("Min |net score| for a signal", 2, 4, 3, key="min_score")
    st.slider("Confirmations (consecutive refreshes)", 1, 3, 2, key="confirm_n")
    st.slider("Stop distance (× ATR)", 0.5, 3.0, 1.5, 0.1, key="atr_mult")
    st.slider("Reward : Risk", 1.0, 4.0, 2.0, 0.5, key="rr")
    st.toggle("Only signal 03:00-16:00 ET (London + NY)", True, key="session_filter")
    st.toggle("🔔 Desktop notifications", True, key="notify_on")
    if st.button("Send test notification"):
        ok = desktop_notify("Gold Macro Signals", "Test notification - it works!")
        st.success("Sent!") if ok else st.error("Failed. Run: pip install plyer")

    st.subheader("📅 News blackout (ET)")
    st.text_area(
        "One event per line: YYYY-MM-DD HH:MM",
        key="events_text", height=110,
        placeholder="2026-11-06 08:30\n2026-12-16 14:00",
        help=f"Signals are blocked {BLACKOUT_BEFORE_MIN} min before to "
             f"{BLACKOUT_AFTER_MIN} min after each event (CPI, NFP, FOMC...).",
    )
    _events, _bad = parse_events(st.session_state.get("events_text", ""))
    st.session_state.events = _events
    st.caption(f"{len(_events)} blackout event(s) loaded")
    if _bad:
        st.warning(f"{_bad} line(s) couldn't be parsed")


# ----------------------------------------------------------------------------
# Dashboard fragment
# ----------------------------------------------------------------------------
@st.fragment(run_every="1s")
def dashboard():
    ss = st.session_state
    now_et = datetime.now(ET)
    market_open = is_market_open(now_et)
    now_ts = time.time()

    ss.setdefault("data", None)
    ss.setdefault("gold_df", None)
    ss.setdefault("last_fetch_ts", 0.0)
    ss.setdefault("last_update", None)
    ss.setdefault("last_confirm_ts", 0.0)
    ss.setdefault("raw_history", [])
    ss.setdefault("raw", {"side": "NO TRADE", "reason": "Waiting for data"})
    ss.setdefault("confirmed", "NO TRADE")
    ss.setdefault("score", 0)
    ss.setdefault("live", 0)
    ss.setdefault("events", [])
    if "log" not in ss:
        ss.log = load_log()

    top_l, top_r = st.columns([5, 1])
    with top_r:
        manual = st.button("🔄 Refresh now", use_container_width=True)

    first_load = ss.data is None
    due = (now_ts - ss.last_fetch_ts) >= REFRESH_SECONDS
    if first_load or manual or (market_open and due):
        with st.spinner("Fetching market data..."):
            data = fetch_all()
        ss.data = data
        ss.gold_df = add_indicators(data["GOLD"]["hist"]) if data.get("GOLD") else None
        ss.last_fetch_ts = time.time()
        ss.last_update = datetime.now().astimezone()
        now_ts = ss.last_fetch_ts
        # Manual clicks shouldn't fast-forward the confirmation counter
        counts = first_load or (time.time() - ss.last_confirm_ts) >= 0.8 * REFRESH_SECONDS
        process_signals(market_open, now_et, counts)

    # ---- status bar
    remaining = max(0, int(REFRESH_SECONDS - (now_ts - ss.last_fetch_ts)))
    mins, secs = divmod(remaining, 60)
    if market_open:
        countdown = f"⏱ Next refresh in <b>{mins:02d}:{secs:02d}</b>"
        session = "🟢 Futures session OPEN"
    else:
        countdown = "⏸ Auto-refresh paused (market closed)"
        session = "🔴 Futures session CLOSED"
    with top_l:
        st.markdown(
            f'<div class="status-bar">🕒 Last update: '
            f'<b>{ss.last_update.strftime("%Y-%m-%d %H:%M:%S %Z")}</b> &nbsp;|&nbsp; '
            f'{countdown} &nbsp;|&nbsp; {session} &nbsp;|&nbsp; '
            f'ET: {now_et.strftime("%a %H:%M")}</div>',
            unsafe_allow_html=True)

    data = ss.data
    df_sc, score, live = build_scorecard(data)
    bias, conf_pct, tier = classify(score)
    raw, confirmed = ss.raw, ss.confirmed

    # ---- headline metrics
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro Bias", bias, f"Net score {score:+d} / {MAX_SCORE}", delta_color="off")
    c2.metric("Confidence", f"{conf_pct:.0f}%")
    c3.metric("Rating Tier", tier)
    if confirmed != "NO TRADE":
        sig_label = ("🟢 " if confirmed == "LONG" else "🔴 ") + confirmed
        sig_sub = "Confirmed"
    elif raw["side"] != "NO TRADE":
        sig_label, sig_sub = "⏳ " + raw["side"], f"Pending {ss.confirm_n}x confirmation"
    else:
        sig_label, sig_sub = "⚪ NO TRADE", None
    c4.metric("Signal", sig_label, sig_sub, delta_color="off")

    # ---- active signal / reason
    st.subheader("Active Signal")
    open_sigs = ss.log[ss.log["status"] == "OPEN"]
    if not open_sigs.empty:
        s = open_sigs.iloc[-1]
        opened = pd.to_datetime(s["time_utc"], utc=True)
        age_min = int((datetime.now(UTC) - opened).total_seconds() // 60)
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Side", ("🟢 " if s["side"] == "LONG" else "🔴 ") + s["side"])
        a2.metric("Entry", f"{s['entry']:,.2f}")
        a3.metric("Stop", f"{s['stop']:,.2f}")
        a4.metric("Target", f"{s['target']:,.2f}")
        a5.metric("Open for", f"{age_min} min")
    else:
        st.info(f"No open signal. {raw.get('reason', '')}")

    # ---- price strip
    st.subheader("Live Prices")
    cols = st.columns(len(ASSETS))
    for col, (key, (symbol, name, dec)) in zip(cols, ASSETS.items()):
        d = data.get(key)
        if d is None:
            col.metric(name, "N/A")
        else:
            col.metric(name, f"{d['last']:,.{dec}f}",
                       f"{d['change']:+.{dec}f} ({d['pct']:+.2f}%)")

    # ---- gold chart
    gold = ss.gold_df
    if gold is not None and len(gold) > 20:
        st.subheader("Gold 5m — Price vs EMA20 / EMA50")
        chart = gold[["Close", "ema20", "ema50"]].tail(150).copy()
        chart.index = chart.index.tz_convert(ET).tz_localize(None)
        st.line_chart(chart, height=280)

    # ---- driver table
    st.subheader("Driver Breakdown")
    st.dataframe(
        df_sc, hide_index=True, use_container_width=True,
        column_config={"Score": st.column_config.NumberColumn("Score", format="%+d")},
    )
    if any(v is None for v in data.values()):
        st.warning("Some feeds returned no data (Yahoo may be rate-limiting). "
                   "Missing/stale drivers score 0 and are excluded from the live count.")

    # ---- history
    st.subheader("Signal History")
    log = ss.log
    closed = log[log["status"].isin(["WIN", "LOSS", "EXPIRED"])]
    wins = int((closed["status"] == "WIN").sum())
    losses = int((closed["status"] == "LOSS").sum())
    win_rate = f"{wins / (wins + losses) * 100:.0f}%" if (wins + losses) else "—"
    avg_r = f"{closed['r_result'].mean():+.2f}R" if len(closed) else "—"
    total_r = f"{closed['r_result'].sum():+.2f}R" if len(closed) else "—"
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Total signals", len(log))
    h2.metric("Closed", len(closed))
    h3.metric("Win rate (TP vs SL)", win_rate)
    h4.metric("Avg result", avg_r)
    h5.metric("Total result", total_r)

    if log.empty:
        st.caption("No signals yet. Confirmed signals will appear here and in "
                   f"{LOG_PATH.name}.")
    else:
        v = log.copy().iloc[::-1]
        sign = np.where(v["side"] == "LONG", 1, -1)
        out = pd.DataFrame({
            "Time (ET)": pd.to_datetime(v["time_utc"], utc=True).dt.tz_convert(ET).dt.strftime("%m-%d %H:%M"),
            "Side": v["side"],
            "Entry": v["entry"].round(2),
            "Stop": v["stop"].round(2),
            "Target": v["target"].round(2),
            "Net Score": v["net_score"].astype("Int64"),
            "Status": v["status"].map({"OPEN": "🔵 OPEN", "WIN": "✅ WIN",
                                       "LOSS": "❌ LOSS", "EXPIRED": "⌛ EXPIRED"}),
            "R": v["r_result"].round(2),
            "15m P/L $": ((v["px_15m"] - v["entry"]) * sign).round(2),
            "30m P/L $": ((v["px_30m"] - v["entry"]) * sign).round(2),
            "60m P/L $": ((v["px_60m"] - v["entry"]) * sign).round(2),
        })
        st.dataframe(out, hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("🥇 Gold (XAUUSD) Macro Signals")
st.caption("Macro driver score + gold trend confirmation + ATR risk levels. "
           "Data via Yahoo Finance (delayed/unofficial). Paper trade first — not financial advice.")
dashboard()
