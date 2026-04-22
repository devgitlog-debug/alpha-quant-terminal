import os
import json
import math
from datetime import datetime
from collections import Counter, defaultdict

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    from google import genai
except Exception:
    genai = None


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Alpha Quant Terminal", page_icon="📈", layout="wide")
st.title("📈 Alpha Quant Terminal")
st.caption("Short-term + Long-term Investment Dashboard | News + Sentiment + Portfolio + Risk")
st.markdown(
    """
    <div style="
        padding: 14px 18px;
        border-radius: 16px;
        background: #0f172a;
        border-left: 4px solid #22c55e;
        margin-bottom: 18px;
    ">
        <div style="font-size: 13px; color: #94a3b8; letter-spacing: 0.5px;">
            BUILT BY
        </div>
        <div style="font-size: 20px; font-weight: 800; color: #ffffff;">
            Divyansh Parashar
        </div>
        <div style="font-size: 13px; color: #cbd5e1;">
            Alpha Quant Terminal
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY"))
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "models/gemini-flash-latest"))

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("⚙️ Settings")
NEWS_LIMIT = st.sidebar.slider("News items per bucket", 3, 10, 5)
CAPITAL = st.sidebar.number_input("Total Capital (₹)", min_value=0.0, value=100000.0, step=10000.0)
RISK_PER_TRADE_PCT = st.sidebar.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25)
MAX_ALLOC_PER_STOCK_PCT = st.sidebar.slider("Max allocation per stock (%)", 5, 30, 12, 1)
MAX_PORTFOLIO_DEPLOY_PCT = st.sidebar.slider("Max deployed capital (%)", 25, 100, 75, 5)
SHOW_AI = st.sidebar.checkbox("Show optional Gemini summary", value=False)

# =========================================================
# HELPERS
# =========================================================
def safe_get_json(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def text_blob(item):
    return f"{item.get('title', '')} {item.get('description', '')}".lower()


def fmt_num(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "N/A"


def fmt_pct(x):
    try:
        return f"{float(x):+.2f}%"
    except Exception:
        return "N/A"


def fmt_price(x):
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return "N/A"


def change_icon(value):
    if value is None:
        return "⚪"
    if value > 0:
        return "🟢⬆️"
    if value < 0:
        return "🔴⬇️"
    return "🟡→"


def action_badge(action):
    a = (action or "HOLD").upper()
    if a == "BUY":
        return "🟢 BUY"
    if a == "ACCUMULATE":
        return "🟢 ACCUMULATE"
    if a == "EXIT":
        return "🔴 EXIT"
    return "🟡 HOLD"

def source_badge(src):
    src = (src or "FALLBACK").upper()
    if src == "NEWS":
        return "🟣 NEWS BASED"
    return "⚪ FALLBACK"


def normalize_label(s):
    return (s or "").strip().lower()


def sentiment_analysis(text):
    t = normalize_label(text)
    positive = [
        "growth", "profit", "surge", "strong", "bull", "upgrade", "beats",
        "record", "expansion", "order", "wins", "recover", "outperform"
    ]
    negative = [
        "loss", "fall", "crash", "weak", "bear", "downgrade", "miss",
        "cut", "delay", "slump", "risk", "selloff", "war", "sanction"
    ]

    score = 0
    for w in positive:
        if w in t:
            score += 1
    for w in negative:
        if w in t:
            score -= 1

    if score > 0:
        return "🟢 Positive", score
    if score < 0:
        return "🔴 Negative", score
    return "🟡 Neutral", score


def impact_label(text):
    t = normalize_label(text)
    if any(k in t for k in ["rbi", "interest rate", "repo", "inflation", "cpi", "fiscal", "bond", "yield"]):
        return "🏦 Rates / Banking"
    if any(k in t for k in ["oil", "crude", "brent", "opec", "energy"]):
        return "🛢 Energy"
    if any(k in t for k in ["war", "sanction", "tariff", "china", "russia", "israel", "middle east", "geopolitic"]):
        return "⚠️ Geo-Risk"
    if any(k in t for k in ["it", "software", "tech", "ai", "digital", "cloud"]):
        return "💻 IT"
    if any(k in t for k in ["defence", "defense", "missile", "navy", "aircraft", "contract", "shipyard", "aerospace"]):
        return "🪖 Defence"
    if any(k in t for k in ["infra", "infrastructure", "order book", "capex", "construction", "engineering"]):
        return "🏗 Infra"
    if any(k in t for k in ["bank", "loan", "deposit", "npa", "credit", "psu bank"]):
        return "🏦 Banking"
    if any(k in t for k in ["pharma", "drug", "medicine", "healthcare", "hospital"]):
        return "💊 Pharma"
    if any(k in t for k in ["auto", "vehicle", "car", "mobility"]):
        return "🚗 Auto"
    return "🌐 General"


def most_common_label(labels, default="🌐 General"):
    if not labels:
        return default
    return Counter(labels).most_common(1)[0][0]


def safe_round(x, digits=2):
    try:
        return round(float(x), digits)
    except Exception:
        return None


# =========================================================
# NEWS ENGINE
# =========================================================
NEWS_BUCKETS = {
    "🇮🇳 Indian Market": [
        "Nifty OR Sensex OR RBI OR Indian stocks OR inflation OR earnings",
        "banking stocks India OR PSU banks OR auto stocks India OR market outlook India",
        "Nifty",
        "Sensex",
        "RBI",
        "Indian stocks",
        "market outlook India",
    ],
    "🌍 Global Market": [
        "US market OR Fed OR Nasdaq OR recession OR treasury yields OR dollar index",
        "Europe markets OR Asia markets OR global growth OR crude oil OR commodities",
        "US stocks",
        "Fed",
        "Nasdaq",
        "crude oil",
        "dollar index",
        "recession",
    ],
    "⚔️ Geo Politics": [
        "war OR sanctions OR tariffs OR China OR Russia OR Israel OR Middle East",
        "shipping route OR supply chain OR trade war OR oil shock",
        "war",
        "sanctions",
        "tariffs",
        "China",
        "Russia",
        "Middle East",
    ],
    "🏭 Sector News": [
        "banking stocks OR IT stocks OR defence stocks OR pharma stocks OR infra stocks",
        "RBI OR order book OR earnings OR guidance OR capex",
        "banking stocks india",
        "IT sector India",
        "metal stocks India",
        "auto stocks",
        "HDFC Bank",
        "SBI",
        "ICICI Bank",
        "TCS",
        "Infosys",
        "HAL",
        "Mazagon Dock",
        "BEL",
        "infra stocks India",
        "banking stocks India",
        "defence stocks India",
        "pharma stocks India",
        "infra stocks India",
    ],
}


@st.cache_data(ttl=600)
def fetch_news(query, max_items=5):
    if not NEWS_API_KEY:
        return []
    q = requests.utils.quote(query)
    url = f"https://gnews.io/api/v4/search?q={q}&lang=en&max={max_items}&apikey={NEWS_API_KEY}"
    data = safe_get_json(url, timeout=20)
    if isinstance(data, dict):
        return data.get("articles", [])
    return []


def dedupe_articles(items):
    seen = set()
    out = []
    for item in items:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


@st.cache_data(ttl=600)
def get_all_news(limit=5):
    news_data = {}
    all_news = []
    for bucket, queries in NEWS_BUCKETS.items():
        bucket_items = []
        for q in queries:
            bucket_items.extend(fetch_news(q, max_items=limit))
        bucket_items = dedupe_articles(bucket_items)[:limit]
        news_data[bucket] = bucket_items
        all_news.extend(bucket_items)
    return news_data, dedupe_articles(all_news)


def analyze_news_items(news_items):
    total_score = 0
    impacts = []
    analyzed = []

    for item in news_items:
        title = item.get("title", "Untitled")
        desc = item.get("description", "")
        src = item.get("source", {}).get("name", "")
        url = item.get("url", "")
        blob = f"{title} {desc}"

        sent_label, sent_score = sentiment_analysis(blob)
        imp = impact_label(blob)
        impacts.append(imp)
        total_score += sent_score

        analyzed.append({
            "title": title,
            "description": desc,
            "source": src,
            "url": url,
            "sentiment_label": sent_label,
            "sentiment_score": sent_score,
            "impact": imp,
        })

    market_impact = most_common_label(impacts)
    return analyzed, total_score, market_impact


# =========================================================
# STOCK LIBRARY
# =========================================================
STOCK_LIBRARY = {
    "reliance": {
        "symbol": "RELIANCE.NS",
        "name": "Reliance",
        "sector": "Energy / Telecom",
        "keywords": ["reliance", "jio", "oil", "petchem", "telecom", "refinery", "petroleum"]
    },
    "hdfcbank": {
        "symbol": "HDFCBANK.NS",
        "name": "HDFC Bank",
        "sector": "Banking",
        "keywords": ["hdfc", "hdfc bank", "banking", "bank", "loan", "deposit", "credit"]
    },
    "icicibank": {
        "symbol": "ICICIBANK.NS",
        "name": "ICICI Bank",
        "sector": "Banking",
        "keywords": ["icici", "icici bank", "banking", "bank", "loan", "deposit", "credit"]
    },
    "sbi": {
        "symbol": "SBIN.NS",
        "name": "SBI",
        "sector": "Banking",
        "keywords": ["sbi", "state bank", "psu bank", "bank", "loan", "deposit", "credit"]
    },
    "tcs": {
        "symbol": "TCS.NS",
        "name": "TCS",
        "sector": "IT",
        "keywords": ["tcs", "it stocks", "software", "digital", "technology", "cloud", "it"]
    },
    "infosys": {
        "symbol": "INFY.NS",
        "name": "Infosys",
        "sector": "IT",
        "keywords": ["infosys", "it stocks", "software", "digital", "technology", "cloud", "it"]
    },
    "lt": {
        "symbol": "LT.NS",
        "name": "Larsen & Toubro",
        "sector": "Infra / Engineering",
        "keywords": ["l&t", "larsen", "infra", "capex", "order book", "construction", "engineering"]
    },
    "itc": {
        "symbol": "ITC.NS",
        "name": "ITC",
        "sector": "FMCG",
        "keywords": ["itc", "fmcg", "cigarette", "hotel", "packaging", "consumer"]
    },
    "sunpharma": {
        "symbol": "SUNPHARMA.NS",
        "name": "Sun Pharma",
        "sector": "Pharma",
        "keywords": ["sun pharma", "pharma", "drug", "medicine", "healthcare", "pharmaceutical"]
    },
    "hal": {
        "symbol": "HAL.NS",
        "name": "HAL",
        "sector": "Defence",
        "keywords": ["hal", "defence", "defense", "aircraft", "aerospace", "fighter", "contract"]
    },
    "mazdock": {
        "symbol": "MAZDOCK.NS",
        "name": "Mazagon Dock",
        "sector": "Defence",
        "keywords": ["mazagon", "mazdock", "defence", "defense", "shipyard", "navy", "submarine", "warship"]
    },
    "bel": {
        "symbol": "BEL.NS",
        "name": "BEL",
        "sector": "Defence Electronics",
        "keywords": ["bel", "defence", "defense", "electronics", "radar", "missile", "contract"]
    },
    "maruti": {
        "symbol": "MARUTI.NS",
        "name": "Maruti Suzuki",
        "sector": "Auto",
        "keywords": ["maruti", "auto", "vehicle", "car", "mobility", "passenger"]
    },
    "m&m": {
        "symbol": "M&M.NS",
        "name": "M&M",
        "sector": "Auto",
        "keywords": ["mahindra", "m&m", "auto", "vehicle", "tractor", "mobility"]
    },
}


def get_fallback_stock_keys():
    # Major diversified names if news mapping returns nothing
    return ["reliance", "hdfcbank", "icicibank", "sbi", "tcs", "infosys", "lt", "hal", "mazdock", "sunpharma", "bel", "itc"]


def extract_stocks_from_news(news_items):
    selected = []
    matched_news = defaultdict(list)
    seen_symbols = set()

    for key, cfg in STOCK_LIBRARY.items():
        for item in news_items:
            blob = text_blob(item)
            if any(k in blob for k in cfg["keywords"]):
                matched_news[key].append(item)

        if matched_news[key]:
            if cfg["symbol"] not in seen_symbols:
                selected.append(key)
                seen_symbols.add(cfg["symbol"])

    if not selected:
        selected = get_fallback_stock_keys()

    # trim duplicate news titles inside each stock bucket
    for key in list(matched_news.keys()):
        matched_news[key] = dedupe_articles(matched_news[key])[:6]

    return selected, matched_news


# =========================================================
# MARKET DATA + TECHNICALS
# =========================================================
@st.cache_data(ttl=300)
def get_market_pulse():
    tickers = {
        "Nifty 50": "^NSEI",
        "Sensex": "^BSESN",
        "India VIX": "^INDIAVIX",
        "Gold": "GC=F",
        "Crude Oil": "CL=F",
        "USDINR": "INR=X",
    }

    out = {}
    for name, sym in tickers.items():
        try:
            hist = yf.Ticker(sym).history(period="5d")
            if hist is None or hist.empty or len(hist["Close"]) < 2:
                out[name] = {"val": None, "delta": None, "delta_pct": None}
                continue

            last = safe_round(hist["Close"].iloc[-1])
            prev = safe_round(hist["Close"].iloc[-2])
            delta = last - prev if last is not None and prev is not None else None
            delta_pct = (delta / prev * 100) if delta is not None and prev not in (None, 0) else None

            out[name] = {
                "val": last,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        except Exception:
            out[name] = {"val": None, "delta": None, "delta_pct": None}

    return out


@st.cache_data(ttl=600)
def get_stock_history(symbol):
    try:
        hist = yf.Ticker(symbol).history(period="6mo")
        return hist if hist is not None else None
    except Exception:
        return None


def calc_indicators(hist):
    if hist is None or hist.empty:
        return None

    close = hist["Close"].dropna()
    high = hist["High"].dropna()
    low = hist["Low"].dropna()

    if len(close) < 20:
        return None

    last = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma20

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    if len(gain.dropna()) == 0 or len(loss.dropna()) == 0:
        rsi = 50.0
    else:
        last_gain = float(gain.iloc[-1]) if pd.notna(gain.iloc[-1]) else 0.0
        last_loss = float(loss.iloc[-1]) if pd.notna(loss.iloc[-1]) else 0.0
        rs = last_gain / last_loss if last_loss not in (0, None) else 999.0
        rsi = 100 - (100 / (1 + rs))

    prev_5 = float(close.iloc[-6]) if len(close) >= 6 else last
    prev_20 = float(close.iloc[-21]) if len(close) >= 21 else last

    trend_5 = (last / prev_5 - 1) * 100 if prev_5 else 0.0
    trend_20 = (last / prev_20 - 1) * 100 if prev_20 else 0.0

    prev_close = close.shift(1)
    tr1 = (high - low)
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr.dropna()) >= 14 else last * 0.015

    recent_high = float(high.tail(20).max()) if len(high) else last
    recent_low = float(low.tail(20).min()) if len(low) else last

    return {
        "last": last,
        "sma20": sma20,
        "sma50": sma50,
        "rsi": float(rsi),
        "atr": float(atr),
        "trend_5": float(trend_5),
        "trend_20": float(trend_20),
        "recent_high": recent_high,
        "recent_low": recent_low,
    }


def cap_category(market_cap):
    try:
        mc = float(market_cap or 0)
    except Exception:
        mc = 0

    # Yahoo market cap is usually in absolute currency units
    if mc >= 2_000_000_000_000:
        return "🟢 Large Cap"
    if mc >= 500_000_000_000:
        return "🟡 Mid Cap"
    return "🔴 Small Cap"


def fundamental_score(symbol):
    try:
        info = yf.Ticker(symbol).info or {}
        roe = info.get("returnOnEquity", 0) or 0
        growth = info.get("revenueGrowth", 0) or 0
        debt = info.get("debtToEquity", 1) or 1
        pm = info.get("profitMargins", 0) or 0

        score = 0
        if roe > 0.15:
            score += 1
        if growth > 0.10:
            score += 1
        if debt < 0.60:
            score += 1
        if pm > 0.08:
            score += 1

        return score, info.get("marketCap", 0), info
    except Exception:
        return 0, 0, {}


# =========================================================
# STOCK DECISION ENGINE
# =========================================================
def related_news_for_stock(stock_key, stock_cfg, news_items):
    hits = []
    for item in news_items:
        blob = text_blob(item)
        if any(k in blob for k in stock_cfg["keywords"]):
            hits.append(item)
    return dedupe_articles(hits)


def stock_news_signal(related_news):
    score = 0
    impacts = []
    for item in related_news:
        blob = f"{item.get('title', '')} {item.get('description', '')}"
        sent_label, sent_score = sentiment_analysis(blob)
        score += sent_score
        impacts.append(impact_label(blob))
    return score, most_common_label(impacts), score


def build_decision(stock_key, stock_cfg, related_news, market_bias):
    symbol = stock_cfg["symbol"]
    name = stock_cfg["name"]

    hist = get_stock_history(symbol)
    ind = calc_indicators(hist)
    fund_score, market_cap, info = fundamental_score(symbol)

    if ind is None:
        return {
            "key": stock_key,
            "symbol": symbol,
            "name": name,
            "sector": stock_cfg["sector"],
            "cap": "N/A",
            "news_category": "🌐 General",
            "news_sentiment_label": "🟡 Neutral",
            "news_sentiment_score": 0,
            "action": "HOLD",
            "horizon": "Wait",
            "entry": "N/A",
            "target": "N/A",
            "stop": "N/A",
            "last": None,
            "rsi": None,
            "sma20": None,
            "sma50": None,
            "trend_5": None,
            "trend_20": None,
            "fund_score": fund_score,
            "risk_score": 0,
            "rr": None,
            "suggest_qty": 0,
            "suggest_invest": 0,
            "reason": "Not enough price history",
            "related_news": related_news[:5],
        }

    last = ind["last"]
    sma20 = ind["sma20"]
    sma50 = ind["sma50"]
    rsi = ind["rsi"]
    atr = ind["atr"]
    trend_5 = ind["trend_5"]
    trend_20 = ind["trend_20"]
    recent_high = ind["recent_high"]
    recent_low = ind["recent_low"]

    news_score, news_category, _ = stock_news_signal(related_news)

    # Dynamic decision logic
    action = "HOLD"
    horizon = "Watch"

    bullish_trend = last > sma20 and sma20 >= sma50 and trend_20 > 0
    strong_long_term = fund_score >= 2 and last > sma50 and trend_20 > 0
    weak_trend = last < sma20 or trend_20 < -3 or rsi > 75 or rsi < 28
    negative_news = news_score < 0

    if strong_long_term and not negative_news and market_bias >= -1:
        action = "ACCUMULATE"
        horizon = "Long Term"
    elif bullish_trend and news_score >= 0 and market_bias >= 0:
        action = "BUY"
        horizon = "Short Term"
    elif (weak_trend and negative_news) or (market_bias <= -2 and trend_5 < 0):
        action = "EXIT"
        horizon = "Risk Off"
    else:
        action = "HOLD"
        horizon = "Wait"

    # Entry / Target / Stop
    if action in ["BUY", "ACCUMULATE"]:
        entry_low = last * 0.985
        entry_high = last * 1.015
        target = last + (atr * 2.2)
        stop = last - (atr * 1.1)
    elif action == "EXIT":
        entry_low = last * 0.99
        entry_high = last * 1.01
        target = max(last - (atr * 1.6), 0.01)
        stop = last + (atr * 1.0)
    else:
        entry_low = last * 0.99
        entry_high = last * 1.01
        target = last + (atr * 1.2) if trend_20 >= 0 else max(last - (atr * 1.2), 0.01)
        stop = last - atr if trend_20 >= 0 else last + atr

    entry_mid = (entry_low + entry_high) / 2
    risk_per_share = abs(entry_mid - stop)
    rr = None
    try:
        rr = abs(target - entry_mid) / risk_per_share if risk_per_share else None
    except Exception:
        rr = None

    cap = cap_category(market_cap)

    # Risk system suggestion
    risk_budget = CAPITAL * (RISK_PER_TRADE_PCT / 100.0)
    alloc_cap = CAPITAL * (MAX_ALLOC_PER_STOCK_PCT / 100.0)
    if action in ["BUY", "ACCUMULATE"] and risk_per_share > 0:
        raw_qty = math.floor(risk_budget / risk_per_share)
        max_qty_by_alloc = math.floor(alloc_cap / entry_mid) if entry_mid > 0 else 0
        suggest_qty = max(0, min(raw_qty, max_qty_by_alloc))
    else:
        suggest_qty = 0

    suggest_invest = suggest_qty * entry_mid

    # Simple risk score
    risk_score = 0
    if negative_news:
        risk_score += 2
    if trend_20 < 0:
        risk_score += 1
    if rsi > 75 or rsi < 28:
        risk_score += 1
    if cap == "🔴 Small Cap":
        risk_score += 1
    if fund_score <= 1:
        risk_score += 1

    reason_parts = [
        f"Fund:{fund_score}/4",
        f"RSI:{rsi:.1f}",
        f"Trend5:{trend_5:+.1f}%",
        f"Trend20:{trend_20:+.1f}%",
        f"News:{news_score:+d}",
        f"Impact:{news_category}",
    ]

    return {
        "key": stock_key,
        "symbol": symbol,
        "name": name,
        "sector": stock_cfg["sector"],
        "cap": cap,
        "news_category": news_category,
        "news_sentiment_label": "🟢 Positive" if news_score > 0 else "🔴 Negative" if news_score < 0 else "🟡 Neutral",
        "news_sentiment_score": news_score,
        "action": action,
        "horizon": horizon,
        "entry": f"{entry_low:.2f} - {entry_high:.2f}",
        "target": f"{target:.2f}",
        "stop": f"{stop:.2f}",
        "last": last,
        "rsi": rsi,
        "sma20": sma20,
        "sma50": sma50,
        "trend_5": trend_5,
        "trend_20": trend_20,
        "fund_score": fund_score,
        "risk_score": risk_score,
        "rr": rr,
        "suggest_qty": suggest_qty,
        "suggest_invest": suggest_invest,
        "reason": " | ".join(reason_parts),
        "related_news": related_news[:5],
        "recent_high": recent_high,
        "recent_low": recent_low,
    }


def market_view_from(pulse, total_news_score):
    bias = 0

    nifty = pulse.get("Nifty 50", {})
    vix = pulse.get("India VIX", {})
    sensex = pulse.get("Sensex", {})

    nifty_delta = nifty.get("delta_pct")
    vix_delta = vix.get("delta_pct")
    sensex_delta = sensex.get("delta_pct")

    if nifty_delta is not None:
        bias += 1 if nifty_delta > 0 else -1
    if sensex_delta is not None:
        bias += 1 if sensex_delta > 0 else -1
    if vix_delta is not None:
        bias += 1 if vix_delta < 0 else -1

    if total_news_score > 3:
        bias += 2
    elif total_news_score < -3:
        bias -= 2

    if bias >= 2:
        return "🟢 Bullish", bias
    if bias <= -2:
        return "🔴 Bearish", bias
    return "🟡 Neutral", bias


def build_action_plan(market_view, stock_rows):
    buy_count = sum(1 for r in stock_rows if r["action"] in ["BUY", "ACCUMULATE"])
    exit_count = sum(1 for r in stock_rows if r["action"] == "EXIT")
    long_count = sum(1 for r in stock_rows if r["horizon"] == "Long Term")
    short_count = sum(1 for r in stock_rows if r["horizon"] == "Short Term")

    if market_view.startswith("🟢"):
        return (
            f"Market supportive. Focus on {long_count} long-term candidates first, then selective {short_count} short-term buys. "
            f"Prefer large caps and high-fundamental names. Keep cash reserve for weak setups."
        )
    if market_view.startswith("🔴"):
        return (
            f"Market weak. Reduce exposure. Avoid small caps. Prefer only strong long-term holds with good fundamentals. "
            f"Exit weak names showing negative news + poor trend."
        )
    return (
        f"Market mixed. Use selective buying only in strong large/mid caps. "
        f"Hold quality names, avoid aggressive averaging, and do not chase weak breakouts."
    )


# =========================================================
# MAIN DATA FLOW
# =========================================================
pulse = get_market_pulse()
news_data, all_news = get_all_news(limit=NEWS_LIMIT)
analyzed_news, total_news_score, dominant_market_impact = analyze_news_items(all_news)
market_view, market_bias = market_view_from(pulse, total_news_score)
selected_stock_keys, matched_news_map = extract_stocks_from_news(all_news)

stock_rows = []

for key in selected_stock_keys:
    cfg = STOCK_LIBRARY[key]
    related_news = matched_news_map.get(key, [])
    row = build_decision(key, cfg, related_news, market_bias)
    row["source"] = "NEWS"
    stock_rows.append(row)

if not stock_rows:
    # Fallback if no news-mapped stocks
    for key in get_fallback_stock_keys():
        cfg = STOCK_LIBRARY[key]
        related_news = matched_news_map.get(key, [])
        row = build_decision(key, cfg, related_news, market_bias)
        row["source"] = "FALLBACK"
        stock_rows.append(row)

# =========================================================
# TOP SUMMARY
# =========================================================
st.subheader("📊 Market Pulse")
pulse_cols = st.columns(6)

for idx, (name, item) in enumerate(pulse.items()):
    with pulse_cols[idx]:
        val = item.get("val")
        delta = item.get("delta")
        delta_pct = item.get("delta_pct")
        icon = change_icon(delta)
        st.metric(
            label=f"{icon} {name}",
            value=fmt_num(val),
            delta=fmt_pct(delta_pct),
            delta_color="inverse" if name == "India VIX" else "normal",
        )

st.divider()

sum_cols = st.columns([1.3, 1, 1])
with sum_cols[0]:
    st.markdown(f"### 🧭 Market View: {market_view.upper()}")
    st.write(build_action_plan(market_view, stock_rows))
with sum_cols[1]:
    st.markdown("### 🧠 News Read")
    st.write(f"Dominant impact: {dominant_market_impact}")
    st.write(f"News score: {total_news_score:+d}")
with sum_cols[2]:
    st.markdown("### ⚙️ Risk Settings")
    st.write(f"Risk/trade: {RISK_PER_TRADE_PCT:.2f}%")
    st.write(f"Max/stock: {MAX_ALLOC_PER_STOCK_PCT:.0f}%")
    st.write(f"Max deploy: {MAX_PORTFOLIO_DEPLOY_PCT:.0f}%")

st.divider()

# =========================================================
# NEWS SECTION
# =========================================================
st.subheader("📰 News Feed")
tabs = st.tabs(list(news_data.keys()))

for tab, bucket_name in zip(tabs, news_data.keys()):
    with tab:
        items = news_data[bucket_name]
        if not items:
            st.info("No news found in this bucket.")
        else:
            for item in items:
                title = item.get("title", "Untitled")
                desc = item.get("description", "")
                source = item.get("source", {}).get("name", "")
                url = item.get("url", "")
                sent_label, sent_score = sentiment_analysis(f"{title} {desc}")
                imp = impact_label(f"{title} {desc}")

                st.markdown(f"**{title}**")
                st.caption(f"{sent_label} | {imp} | Score {sent_score:+d}")
                if desc:
                    st.write(desc)
                if source or url:
                    st.caption(f"{source} | {url}")
                st.write("")

st.divider()

# =========================================================
# STOCK BOARD
# =========================================================
st.subheader("📌 Stock Action Board")

portfolio_inputs = []
holdings_rows = []

with st.expander("💼 Portfolio / Holdings / Risk Manager", expanded=True):
    st.caption("Enter your existing holdings here. The dashboard will calculate invested value, current value, P&L, and risk allocation suggestions.")

    for row in stock_rows:
        with st.container(border=True):
            left, mid, right = st.columns([1.2, 1, 1])

            signal_icon = "🟢⬆️" if row["action"] in ["BUY", "ACCUMULATE"] else "🔴⬇️" if row["action"] == "EXIT" else "🟡→"
            left.markdown(f"**{signal_icon} {row['name']}**")
            left.caption(f"{row['sector']} | {row['cap']} | {row['horizon']} | {source_badge(row.get('source'))}")
            left.caption(f"News: {row['news_sentiment_label']} | {row['news_category']}")
            left.caption(f"Fund score: {row['fund_score']}/4 | Risk score: {row['risk_score']}/5")

            mid.markdown(f"{action_badge(row['action'])}")
            mid.write(f"Entry: **{row['entry']}**")
            mid.write(f"Target: **{row['target']}**")
            mid.write(f"Stop: **{row['stop']}**")

            rr_text = f"{row['rr']:.2f}" if isinstance(row["rr"], (int, float)) and row["rr"] is not None else "N/A"
            right.write(f"Last: **{fmt_price(row['last'])}**")
            right.write(f"RSI: **{row['rsi']:.1f}**")
            right.write(f"R:R: **{rr_text}**")
            right.write(f"Suggested Qty: **{row['suggest_qty']}**")
            right.write(f"Suggested Invest: **{fmt_price(row['suggest_invest'])}**")

            st.caption(row["reason"])

            c1, c2 = st.columns(2)
            qty = c1.number_input(
                f"Owned Qty - {row['name']}",
                min_value=0,
                step=1,
                value=0,
                key=f"qty_{row['key']}"
            )
            avg_buy = c2.number_input(
                f"Avg Buy Price - {row['name']}",
                min_value=0.0,
                step=1.0,
                value=0.0,
                key=f"avg_{row['key']}"
            )

            current_price = row["last"] or 0.0
            invested = qty * avg_buy if qty and avg_buy else 0.0
            current_value = qty * current_price if qty else 0.0
            pnl = current_value - invested if qty else 0.0
            alloc_pct = (current_value / CAPITAL * 100) if CAPITAL > 0 else 0.0

            holdings_rows.append({
                "Stock": row["name"],
                "Action": row["action"],
                "Type": row["horizon"],
                "Category": row["cap"],
                "Qty": qty,
                "Avg Buy": avg_buy,
                "Current": current_price,
                "Invested": invested,
                "Current Value": current_value,
                "P&L": pnl,
                "% of Capital": alloc_pct,
                "Suggested Qty": row["suggest_qty"],
                "Suggested Add ₹": row["suggest_invest"],
            })

            portfolio_inputs.append({
                "key": row["key"],
                "name": row["name"],
                "qty": qty,
                "avg_buy": avg_buy,
                "current": current_price,
                "invested": invested,
                "current_value": current_value,
                "pnl": pnl,
                "action": row["action"],
                "suggested_qty": row["suggest_qty"],
                "suggested_invest": row["suggest_invest"],
            })

st.divider()

# =========================================================
# PORTFOLIO SUMMARY
# =========================================================
total_invested = sum(x["invested"] for x in portfolio_inputs)
total_current_value = sum(x["current_value"] for x in portfolio_inputs)
total_pnl = total_current_value - total_invested
deployed_pct = (total_invested / CAPITAL * 100) if CAPITAL > 0 else 0.0
cash_left = CAPITAL - total_invested

portfolio_cols = st.columns(5)
portfolio_cols[0].metric("Total Capital", fmt_price(CAPITAL))
portfolio_cols[1].metric("Invested", fmt_price(total_invested))
portfolio_cols[2].metric("Current Value", fmt_price(total_current_value))
portfolio_cols[3].metric("P&L", fmt_price(total_pnl), delta=fmt_pct((total_pnl / total_invested * 100) if total_invested > 0 else 0))
portfolio_cols[4].metric("Deployed", fmt_pct(deployed_pct))

st.write(f"Cash left: **{fmt_price(cash_left)}**")

if deployed_pct > MAX_PORTFOLIO_DEPLOY_PCT:
    st.warning("Portfolio deployment is above your limit. Reduce exposure or avoid fresh additions.")

if portfolio_inputs:
    st.dataframe(pd.DataFrame(holdings_rows), width="stretch")

st.divider()

# =========================================================
# SMART PORTFOLIO / RISK RECOMMENDATIONS
# =========================================================
st.subheader("🧠 Portfolio Suggestions")

for row in stock_rows:
    if row["action"] in ["ACCUMULATE", "BUY"]:
        st.success(
            f"{row['name']}: {row['action']} | Cap: {row['cap']} | "
            f"Suggested Add: {row['suggest_qty']} shares | {fmt_price(row['suggest_invest'])}"
        )
    elif row["action"] == "EXIT":
        st.error(f"{row['name']}: EXIT / Reduce exposure")
    else:
        st.info(f"{row['name']}: HOLD / Wait")

st.divider()

# =========================================================
# OPTIONAL GEMINI SUMMARY (ADVANCED AI ENGINE)
# =========================================================
if SHOW_AI:
    st.subheader("✨ Expert Gemini AI Insights")
    if not GEMINI_API_KEY or genai is None:
        st.warning("Gemini is not available right now. Dashboard is running in rule-based mode.")
    else:
        if st.button("🧠 Run Deep AI Summary"):
            with st.spinner("AI is analyzing cross-asset correlations & finding top stocks..."):
                try:
                    client = genai.Client(api_key=GEMINI_API_KEY)
                    
                    # ---------------------------------------------------------
                    # SNIPPET 3: TOP 5 STOCKS SORTING LOGIC
                    # (Priority: BUY/ACCUMULATE > Fund Score > Less Risk)
                    # ---------------------------------------------------------
                    sorted_stocks = sorted(
                        stock_rows, 
                        key=lambda r: (r["action"] in ["BUY", "ACCUMULATE"], r["fund_score"], -r["risk_score"]), 
                        reverse=True
                    )
                    best_5_stocks = sorted_stocks[:5]

                    # ---------------------------------------------------------
                    # SNIPPET 2: TOP 5 IMPACT NEWS SORTING LOGIC
                    # (Absolute Score के आधार पर सबसे बड़े धमाके वाली न्यूज़)
                    # ---------------------------------------------------------
                    impactful_news = sorted(
                        analyzed_news, 
                        key=lambda n: abs(n['sentiment_score']), 
                        reverse=True
                    )
                    top_headlines_for_ai = [
                        f"[{n['impact']}] {n['title']} (Score: {n['sentiment_score']})" 
                        for n in impactful_news[:5]
                    ]

                    # ---------------------------------------------------------
                    # SNIPPET 4: NEW AI PAYLOAD & PROMPT
                    # ---------------------------------------------------------
                    payload = {
                        "market_context": {
                            "view": market_view,
                            "sentiment_score": total_news_score,
                            "top_impact_headlines": top_headlines_for_ai
                        },
                        "global_commodities_USD": {
                            "Gold_Global_USD": pulse.get("Gold (Global)", {}).get("val", "N/A"),
                            "Crude_Oil_USD": pulse.get("Crude Oil", {}).get("val", "N/A")
                        },
                        "indian_indices_INR": {
                            "Nifty_50": pulse.get("Nifty 50", {}).get("val", "N/A"),
                            "India_VIX": pulse.get("India VIX", {}).get("val", "N/A"),
                            "Gold_ETF_INR": pulse.get("Gold ETF (India)", {}).get("val", "N/A")
                        },
                        "portfolio_rules": {
                            "total_capital_INR": CAPITAL,
                            "risk_per_trade_pct": RISK_PER_TRADE_PCT
                        },
                        "top_opportunities": [{"stock": r["name"], "action": r["action"], "reason": r["reason"]} for r in best_5_stocks]
                    }

                    prompt = f"""
                    You are a highly experienced Quantitative Fund Manager in India.
                    I am providing you with live market data as a JSON payload.

                    CRITICAL RULES:
                    - 'global_commodities_USD' values are in US Dollars ($).
                    - 'indian_indices_INR' and stock prices are in Indian Rupees (₹) or Points.
                    - Analyze the 'top_impact_headlines' carefully to understand the market drivers.

                    Based on the payload, provide a highly professional 3-point summary in JSON format EXACTLY with these keys:
                    1. "macro_view": (Analyze the specific top headlines and commodity prices to explain current market conditions).
                    2. "capital_strategy": (What to do with the {CAPITAL} INR capital right now based on the risk and macro view).
                    3. "stock_focus": (Which 1 or 2 stocks from the payload have the best setup and why).

                    DATA PAYLOAD:
                    {json.dumps(payload, ensure_ascii=False)}
                    """

                    # यहाँ तुम्हारा st.secrets वाला मॉडल नेम इस्तेमाल हो रहा है
                    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                    
                    ai_text = resp.text.replace('```json', '').replace('```', '')
                    st.json(json.loads(ai_text))
                    
                except Exception as e:
                    st.error(f"Gemini API Error: {e}")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
