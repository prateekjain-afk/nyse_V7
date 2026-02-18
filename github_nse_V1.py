# @title NSE Intraday Scanner - Robust ML + Consistent Stops + Time-Exit EV V5
from __future__ import annotations

import os, re, math, time, gzip, pickle, logging, warnings, io, random
from dataclasses import dataclass
from datetime import datetime as dt, timedelta, date, time as dtime
from typing import Any, Dict, List, Optional, Tuple
from bisect import bisect_right

import pandas as pd
import numpy as np
import requests
from tqdm.auto import tqdm
import yfinance as yf
import pytz

warnings.filterwarnings("ignore")

# =========================
# Optional ML libs
# =========================
SKLEARN_OK = True
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import log_loss
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    SKLEARN_OK = False


# ============================================================
# USER CONFIG - EDIT THESE
# ============================================================
API_PROVIDER = "YFINANCE"
API_KEY = ""

MAX_UNIVERSE = 500
MAX_WORKERS = 5
PICK_BUFFER_MULT = 4

CACHE_DIR = "cache_nse_scan"
os.makedirs(CACHE_DIR, exist_ok=True)

REGIME_TICKER = "NIFTYBEES.NS"
RELAX_PRESET = "VWAP_TREND"

UNIVERSE_SOURCE = "NSE"
UNIVERSE_CACHE_SIZE = 900

RVOL_INTERVAL = "5m"
RVOL_PERIOD = "15d"   # enough for 10 sessions + buffer


# ============================================================
# RUN MODE TOGGLE (LIVE vs SIMULATION)
# ============================================================
SIMULATION_MODE = True
SIM_TEST_DT = dt(2026, 2, 18, 11, 00)  # naive treated as IST

# ============================================================
# OVERLAYS / FILTERS
# ============================================================
ENABLE_SECTOR_GUARDRAILS = True

ENABLE_NEWS_FILTER = False
NEWS_LOOKBACK_HOURS = 48
NEWS_TOP_K = 50
NEWS_NEGATIVE_HARD_BLOCK = False
NEWS_SCORE_PENALTY = 6.0
NEWS_SCORE_BONUS = 2.0

ENABLE_EARNINGS_BLACKOUT = False
EARNINGS_BLACKOUT_DAYS = 2

# ============================================================
# PROBABILITY MODEL SETTINGS (RR-AWARE)
# ============================================================
ENABLE_PROB_MODEL = True

PROB_INTERVAL = "5m"
PROB_PERIOD = "60d"

PROB_TRAIN_LOOKBACK_SESSIONS = 45
PROB_TRAIN_TICKERS_MAX = 220
PROB_PREFETCH_BATCH = 60

PROB_USE_ASOF_TIME = True
PROB_TIE_POLICY = "STOP"       # STOP: treat TIE as SL; HALF: treat as NONE-ish
PROB_PRIOR_BLEND_K = 20

PROB_CACHE_TTL = 8 * 3600
INTRADAY_CACHE_TTL_1M = 10 * 60
INTRADAY_CACHE_TTL_5M = 60 * 60

# RR choices: include 0.75 to reduce "NONE all day" when stops are wider (safer sizing).
RR_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# ============================================================
# TAKE / PASS decision thresholds (base; will be regime-adjusted)
# ============================================================
ENABLE_TAKE_COLUMN = True
TAKE_MIN_EV_R = 0.03
TAKE_MAX_P_SL = 0.48
TAKE_MIN_P_TP_GIVEN_HIT = 0.38
TAKE_MAX_UNCERTAINTY = 1.08
TAKE_MIN_EMP_N = 10

# Robustness additions: prevent "high NONE probability" trades masquerading as good EV
TAKE_MIN_P_HIT = 0.45          # P(TP or SL) must be at least this (else likely no exit)
TAKE_MAX_P_NONE = 0.65         # cap on P_NONE

# Robustness additions: avoid targets that are very unlikely to be reached intraday
ENABLE_TARGET_REACHABILITY_FILTER = True
TARGET_MAX_ATR_MULT = 2.5      # require (Target-Entry) <= TARGET_MAX_ATR_MULT * ATR14 when ATR available
TARGET_MAX_PCT = 0.12          # fallback cap if ATR missing

# ============================================================
# Dynamic TP/SL barrier modeling (safe, data-available)
# ============================================================
# These settings improve realism: stops adapt to *known* intraday volatility at entry,
# and targets (RR choices) are capped by what has been historically achievable for that
# ticker at the same entry time (no peeking into the current day).
ENABLE_DYNAMIC_STOP_FROM_INTRA_RANGE = True
STOP_INTRA_RANGE_MULT = 0.60   # stop distance floor = 0.60 * (high-low from open->entry)

ENABLE_DYNAMIC_RR_CAP_FROM_MFE = True
MFE_LOOKBACK_SESSIONS = 45     # uses prior sessions only
MFE_QUANTILE = 0.60            # typical "reachable" upside
MFE_MIN_SAMPLES = 12
MFE_RR_SLACK = 1.10            # small slack over historical quantile

# ============================================================
# Dynamic Risk Management (ATR-based stop fallback)
# ============================================================
ATR_LEN = 14
ATR_STOP_MULT = 1.5

STOP_FALLBACK_PCT = 0.0125
STOP_MIN_PCT      = 0.0040
STOP_MAX_PCT      = 0.0600

# ============================================================
# Market breadth regime thresholds
# ============================================================
BREADTH_NEGATIVE_RATIO = 0.45
BREADTH_STRICTEN_FACTOR = 0.20  # 20%

# ============================================================
# Intraday ATR + Vol-clustering (time-of-day + volatility shock)
# ============================================================
# Why:
# - Daily ATR (ATR14) is too slow for sudden intraday spikes.
# - NSE's first ~45 minutes often has "volatility clustering" (bigger swings),
#   while mid-day is usually calmer. We widen stops in volatile windows and
#   tighten them in calmer windows to reduce random stop-outs and reduce
#   unrealistic TP/SL expectations.
ENABLE_INTRADAY_ATR = True
INTRA_ATR_BARS = 12            # last ~60 minutes on 5m bars (12 * 5m)
INTRA_ATR_LEN  = 12            # ATR length on those bars
INTRA_ATR_STOP_MULT = 1.00     # stop distance floor = atr_intra * mult * vol_mult

ENABLE_VOL_CLUSTERING_TOD = True
# Time-of-day volatility multipliers (IST). Calibrated for "open chaos" vs mid-day lull.
TOD_VOL_SCHEDULE = [
    (dtime(9, 15), dtime(10, 0), 1.35),
    (dtime(10, 0), dtime(11, 30), 1.15),
    (dtime(11, 30), dtime(14, 0), 1.00),
    (dtime(14, 0), dtime(15, 0), 1.10),
    (dtime(15, 0), dtime(15, 30), 1.25),
]

# "Vol shock" multiplier: compares intraday ATR to a rough expected 5m ATR implied by daily ATR.
# This captures volatility clustering beyond time-of-day (e.g., news spikes).
BARS_PER_SESSION_5M = 75
VOL_SHOCK_CLAMP_LO, VOL_SHOCK_CLAMP_HI = 0.80, 1.80

def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        if not np.isfinite(x):
            return lo
    except Exception:
        return lo
    return float(max(lo, min(hi, x)))

def tod_vol_multiplier(clock: dtime) -> float:
    if not ENABLE_VOL_CLUSTERING_TOD:
        return 1.0
    for a, b, mult in TOD_VOL_SCHEDULE:
        if a <= clock < b:
            return float(mult)
    return 1.0

def intraday_atr_from_5m(df_5m_pre: pd.DataFrame, bars: int = INTRA_ATR_BARS, atr_len: int = INTRA_ATR_LEN) -> float:
    """ATR computed from the most recent pre-entry 5m candles."""
    if df_5m_pre is None or len(df_5m_pre) < 3:
        return float("nan")
    g = df_5m_pre.tail(max(bars, atr_len + 1)).copy()
    for col in ("high", "low", "close"):
        if col not in g.columns:
            return float("nan")
    prev_close = g["close"].shift(1)
    tr1 = (g["high"] - g["low"]).abs()
    tr2 = (g["high"] - prev_close).abs()
    tr3 = (g["low"] - prev_close).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = tr.rolling(atr_len, min_periods=max(3, atr_len // 2)).mean()
    return float(atr.iloc[-1])

def vol_cluster_multiplier(clock: dtime, atr_intra: float, atr14: float) -> float:
    """Combines time-of-day multiplier with a volatility-shock multiplier."""
    base = tod_vol_multiplier(clock)
    shock = 1.0
    if ENABLE_INTRADAY_ATR and np.isfinite(atr_intra) and np.isfinite(atr14) and atr14 > 0:
        # Expected 5m ATR implied by daily ATR (rough random-walk scaling).
        expected_5m_atr = atr14 / np.sqrt(BARS_PER_SESSION_5M)
        if expected_5m_atr > 0:
            shock = _clamp(atr_intra / expected_5m_atr, VOL_SHOCK_CLAMP_LO, VOL_SHOCK_CLAMP_HI)
    return float(base * shock)

# Timezone
ET = pytz.timezone("Asia/Kolkata")   # keep variable name ET for minimal changes in code paths
IST = ET
UTC = pytz.UTC


# ============================================================
# LOGGING
# ============================================================
logger = logging.getLogger("nse_scanner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)

logger.info(f"🚀 NSE Scanner | API={API_PROVIDER} | MAX_UNIVERSE={MAX_UNIVERSE} | ML={SKLEARN_OK}")


# ============================================================
# CONFIG
# ============================================================
@dataclass
class ScanConfig:
    price_min: float = 5.0
    avg10_dollar_vol_min: float = 1.0e7   # treated as INR notional now (label only)
    max_universe: int = MAX_UNIVERSE

    opening_range_start: dtime = dtime(9, 15)
    opening_range_end: dtime = dtime(9, 45)    # 30-min OR for NSE
    eval_time_default: dtime = dtime(10, 0)

    rvol_min: float = 0.8
    rvol_lookback_sessions: int = 10

    require_macd_hist_rising: bool = False
    macd_min_bars_5m: int = 20
    macd_min_bars_15m_soft: int = 15

    stage1_require_vwap: bool = False
    stage1_require_macd: bool = False
    stage1_require_orh: bool = False
    vwap_tol_pct: float = 0.05
    orh_tol_pct: float = 0.05

    score_min: float = 40.0
    max_per_sector: int = 8
    target_positions: int = 40

    min_pos_dollars: float = 1500.0  # treated as INR now (label only)
    stop_buffer_below_15m_low: float = 0.005
    rr_min: float = 1.0

cfg = ScanConfig()

logger.info(
    f"Preset={RELAX_PRESET} | rvol_min={cfg.rvol_min} | score_min={cfg.score_min} | "
    f"liq_min(INR notional)={cfg.avg10_dollar_vol_min:,.0f}"
)

# ============================================================
# UTILITIES
# ============================================================
class SimpleFileCache:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", key)
        return os.path.join(self.base_dir, f"{safe}.pkl.gz")

    def get(self, key: str, ttl_seconds: Optional[int] = None) -> Any:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        if ttl_seconds is not None:
            age = time.time() - os.path.getmtime(path)
            if age > ttl_seconds:
                return None
        try:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        except:
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        tmp = path + ".tmp"
        with gzip.open(tmp, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

cache = SimpleFileCache(CACHE_DIR)

class RateLimiter:
    def __init__(self, max_calls_per_minute: int):
        self.max_calls_per_minute = max(1, int(max_calls_per_minute))
        self.min_interval = 60.0 / float(self.max_calls_per_minute)
        self._last = 0.0
        import threading
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0 or np.isnan(b):
            return default
        return a / b
    except:
        return default

def dt_combine(d: date, t: dtime, tz=ET) -> dt:
    return tz.localize(dt(d.year, d.month, d.day, t.hour, t.minute))

def floor_time_to_minute(x: dt) -> dt:
    return x.replace(second=0, microsecond=0)

def minutes_between(a: dt, b: dt) -> float:
    return float((b - a).total_seconds() / 60.0)

# ============================================================
# NSE TRADING CALENDAR (best effort)
# ============================================================
def _get_nse_calendar():
    try:
        import exchange_calendars as xcals
        for name in ("XNSE", "NSE", "XBOM"):
            try:
                cal = xcals.get_calendar(name)
                return ("exchange_calendars", cal)
            except Exception:
                continue
    except Exception:
        pass

    try:
        import pandas_market_calendars as mcal
        for name in ("NSE", "XNSE", "BSE", "XBOM"):
            try:
                cal = mcal.get_calendar(name)
                return ("pandas_market_calendars", cal)
            except Exception:
                continue
    except Exception:
        pass

    return ("weekday_fallback", None)

_CAL_MODE, _EXCH = _get_nse_calendar()
if _CAL_MODE == "weekday_fallback":
    logger.warning("⚠️ Trading calendar fallback is weekday-only (won't know NSE holidays). Install exchange_calendars for best results.")
else:
    logger.info(f"✅ Trading calendar mode: {_CAL_MODE}")

def is_trading_day(d: date) -> bool:
    if _EXCH is None:
        return d.weekday() < 5
    try:
        if _CAL_MODE == "exchange_calendars":
            sess = _EXCH.sessions_in_range(pd.Timestamp(d), pd.Timestamp(d))
            return len(sess) > 0
        else:
            days = _EXCH.valid_days(start_date=d, end_date=d)
            return len(days) > 0
    except Exception:
        return d.weekday() < 5

def get_trading_days(end_d: date, n: int) -> List[date]:
    if _EXCH is not None:
        try:
            start = end_d - timedelta(days=900)
            if _CAL_MODE == "exchange_calendars":
                sess = _EXCH.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end_d - timedelta(days=1)))
                out = [pd.Timestamp(x).date() for x in sess]
                return out[-n:] if len(out) >= n else out
            else:
                valid = _EXCH.valid_days(start_date=start, end_date=end_d - timedelta(days=1))
                valid = pd.to_datetime(valid)
                if getattr(valid, "tz", None) is not None:
                    valid = valid.tz_convert(None)
                out = [x.date() for x in valid]
                return out[-n:] if len(out) >= n else out
        except Exception:
            pass

    out = []
    d = end_d - timedelta(days=1)
    while len(out) < n and (end_d - d).days < 1200:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)

def get_last_trading_day(now_et: dt) -> date:
    d0 = now_et.date()
    if is_trading_day(d0) and now_et.time() < cfg.opening_range_start:
        d0 = d0 - timedelta(days=1)
    for _ in range(60):
        if is_trading_day(d0):
            return d0
        d0 = d0 - timedelta(days=1)
    return now_et.date()

def get_session_open_close(d: date) -> Tuple[dt, dt]:
    return dt_combine(d, cfg.opening_range_start, ET), dt_combine(d, dtime(15, 30), ET)

# ============================================================
# UNIVERSE HELPERS (NIFTY500 preferred; fallback to EQUITY_L)
# ============================================================
_http_rl = RateLimiter(60)

def _http_get_text(url: str, timeout: int = 45, retries: int = 3, backoff: float = 1.6) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/plain,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com/",
    }
    last_err = None
    for i in range(max(1, retries)):
        try:
            _http_rl.wait()
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep((backoff ** i) + random.random() * 0.25)
    raise RuntimeError(f"GET failed for {url}: {last_err}")

def _clean_nse_symbol(sym: str) -> Optional[str]:
    if not isinstance(sym, str):
        return None
    s = sym.strip().upper()
    if not s:
        return None
    if any(ch in s for ch in ["$", "^", "/", "\\", " "]):
        return None
    if not re.match(r"^[A-Z0-9\-\&]{1,20}$", s):
        return None
    return s

def fetch_nse_universe() -> pd.DataFrame:
    cache_key = "nse_universe_v3"
    cached = cache.get(cache_key, ttl_seconds=24 * 3600)
    if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached.copy()

    # Try NIFTY500 constituent list (archives)
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    ]

    df = None
    used = None
    for u in urls:
        try:
            txt = _http_get_text(u, timeout=60, retries=4)
            df0 = pd.read_csv(io.StringIO(txt))
            if df0 is not None and not df0.empty:
                df = df0
                used = u
                break
        except Exception:
            continue

    if df is None or df.empty:
        out = pd.DataFrame({"ticker": ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"],
                            "name": ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK"], "exchange": ["NSE"]*5})
        cache.set(cache_key, out)
        logger.info(f"✅ Fetched {len(out)} NSE tickers (fallback hardcoded)")
        return out.copy()

    # Parse depending on file
    sym_col = None
    name_col = None
    for c in ["Symbol", "SYMBOL"]:
        if c in df.columns:
            sym_col = c
            break
    for c in ["Company Name", "Security Name", "NAME OF COMPANY", "NAME"]:
        if c in df.columns:
            name_col = c
            break

    if sym_col is None:
        sym_col = df.columns[0]

    rows = []
    for _, r in df.iterrows():
        sym = _clean_nse_symbol(str(r.get(sym_col, "")))
        if not sym:
            continue
        tkr = f"{sym}.NS"
        nm = str(r.get(name_col, sym)).strip() if name_col else sym
        rows.append({"ticker": tkr, "name": nm, "exchange": "NSE"})

    out = pd.DataFrame(rows).drop_duplicates("ticker").reset_index(drop=True)
    cache.set(cache_key, out)
    if used and "nifty500" in used.lower():
        logger.info(f"✅ Fetched {len(out)} NSE tickers (NIFTY500 preferred; fallback: EQUITY_L)")
    else:
        logger.info(f"✅ Fetched {len(out)} NSE tickers (EQUITY_L)")
    return out.copy()

# ============================================================
# DATA CLIENT - YFINANCE + CACHE
# ============================================================
class DataClient:
    def __init__(self, cache_obj: Optional[SimpleFileCache] = None):
        self.cache = cache_obj
        # Cache validity controls (set by the main run block)
        self.require_end_dt = None  # tz-aware datetime; if cache ends before this, we refetch
        self.force_refresh = False  # if True, bypass cache reads entirely
        self.rl = RateLimiter(60)

    def list_stocks(self) -> pd.DataFrame:
        cache_key = f"nse_universe_list_v3_{UNIVERSE_CACHE_SIZE}"
        cached = self.cache.get(cache_key, ttl_seconds=24 * 3600) if self.cache else None
        if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
            logger.info(f"Using cached universe: {len(cached)} stocks")
            return cached.copy()

        uni = fetch_nse_universe()
        df = uni.copy()
        df["symbol"] = df["ticker"]
        df = df.head(UNIVERSE_CACHE_SIZE).reset_index(drop=True)

        if self.cache:
            self.cache.set(cache_key, df)

        logger.info(f"Final universe: {len(df)} stocks ({UNIVERSE_SOURCE})")
        return df.copy()

    def _daily_cache_key(self, symbols: List[str], start_date: date, end_date: date) -> str:
        syms = sorted([s for s in symbols if isinstance(s, str)])
        sig = str(hash(tuple(syms[:200])))
        return f"daily_nse_v6_{start_date}_{end_date}_{len(syms)}_{sig}"

    def get_daily_data(self, symbols: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        cache_key = self._daily_cache_key(symbols, start_date, end_date)
        cached = self.cache.get(cache_key, ttl_seconds=12 * 3600) if self.cache else None
        if cached is not None and isinstance(cached, pd.DataFrame):
            return cached.copy()

        all_rows = []
        end_plus = end_date + timedelta(days=1)
        batch_size = 80

        for i in tqdm(range(0, len(symbols), batch_size), desc="Daily data", leave=False):
            chunk = symbols[i:i+batch_size]
            try:
                self.rl.wait()
                data = yf.download(
                    chunk, start=start_date, end=end_plus, interval="1d",
                    group_by="ticker", auto_adjust=False, threads=True, progress=False
                )
                if data is None or data.empty:
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                    for tkr in chunk:
                        if tkr not in data.columns.get_level_values(0):
                            continue
                        dft = data[tkr].dropna(how="all").reset_index()
                        if not dft.empty:
                            dft["ticker"] = tkr
                            all_rows.append(dft[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]])
                else:
                    dft = data.dropna(how="all").reset_index()
                    if not dft.empty:
                        dft["ticker"] = chunk[0]
                        all_rows.append(dft[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]])
            except Exception:
                continue

        if not all_rows:
            return pd.DataFrame()

        out = pd.concat(all_rows, ignore_index=True)
        out.columns = ["date", "ticker", "open", "high", "low", "close", "volume"]
        out["date"] = pd.to_datetime(out["date"])
        if self.cache:
            self.cache.set(cache_key, out)
        return out.copy()

    def _intraday_cache_key(self, symbol: str, interval: str, period: str) -> str:
        return f"intraday_nse_v9_{symbol}_{interval}_{period}"

    def _fetch_intraday_raw(self, symbol: str, interval: str, period: str, ttl: int) -> pd.DataFrame:
        cache_key = self._intraday_cache_key(symbol, interval, period)
        cached = self.cache.get(cache_key, ttl_seconds=ttl) if self.cache else None
        if self.force_refresh:
            cached = None
        if cached is not None and isinstance(cached, pd.DataFrame):
            # If the cached file is too old to cover the requested simulation window, refetch.
            if self.require_end_dt is not None and not cached.empty:
                try:
                    last_ts = cached.index.max()
                    last_ts = pd.Timestamp(last_ts).to_pydatetime()
                    if last_ts.tzinfo is None:
                        last_ts = UTC.localize(last_ts).astimezone(ET)
                    if last_ts < self.require_end_dt:
                        cached = None
                except Exception:
                    cached = None
            if cached is not None:
                return cached.copy()

        try:
            self.rl.wait()
            df = yf.Ticker(symbol).history(period=period, interval=interval, prepost=False)
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            dtcol = "Datetime" if "Datetime" in df.columns else "Date"
            df[dtcol] = pd.to_datetime(df[dtcol])

            if df[dtcol].dt.tz is None:
                df[dtcol] = df[dtcol].dt.tz_localize(UTC)
            df[dtcol] = df[dtcol].dt.tz_convert(ET)

            df = df.set_index(dtcol)
            df.index.name = "datetime"

            out = pd.DataFrame({
                "open": df.get("Open", pd.Series(dtype=float)),
                "high": df.get("High", pd.Series(dtype=float)),
                "low": df.get("Low", pd.Series(dtype=float)),
                "close": df.get("Close", pd.Series(dtype=float)),
                "volume": df.get("Volume", pd.Series(dtype=float)),
            }).dropna(subset=["open", "high", "low", "close"])

            if self.cache:
                self.cache.set(cache_key, out)
            return out.copy()
        except Exception:
            return pd.DataFrame()

    def get_intraday_data(self, symbol: str, date_: date, interval: str = "1m", period: str = "8d") -> pd.DataFrame:
        if interval == "1m":
            period = "8d" if period not in ("1d","2d","5d","7d","8d") else period
            ttl = INTRADAY_CACHE_TTL_1M
        else:
            ttl = INTRADAY_CACHE_TTL_5M

        raw = self._fetch_intraday_raw(symbol, interval, period, ttl=ttl)
        if raw.empty:
            return pd.DataFrame()

        day = raw[raw.index.date == date_].copy()
        # If we're simulating and the cached window doesn't include this day (common when changing SIM_TEST_DT),
        # force a one-time refresh and try again.
        if day.empty and self.require_end_dt is not None and (not self.force_refresh):
            try:
                old_force = self.force_refresh
                self.force_refresh = True
                raw2 = self._fetch_intraday_raw(symbol, interval, period, ttl=0)
                day2 = raw2[raw2.index.date == date_].copy() if raw2 is not None and not raw2.empty else pd.DataFrame()
                self.force_refresh = old_force
                if not day2.empty:
                    return day2
            except Exception:
                self.force_refresh = False
        return day

client = DataClient(cache_obj=cache)

# --- Prefetch intraday (place ABOVE Stage-2 where you call it) ---
def prefetch_intraday_batch(
    symbols: List[str],
    interval: str,
    period: str,
    batch_size: int = 80,
    ttl_seconds: int = INTRADAY_CACHE_TTL_5M,
):
    symbols = [s for s in symbols if isinstance(s, str) and s]

    need = []
    for s in symbols:
        key = client._intraday_cache_key(s, interval, period)
        cached = None if client.force_refresh else cache.get(key, ttl_seconds=ttl_seconds)

        stale = False
        if cached is None or (not isinstance(cached, pd.DataFrame)) or cached.empty:
            stale = True
        elif client.require_end_dt is not None:
            try:
                last_ts = pd.Timestamp(cached.index.max())
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize(UTC)
                last_ts = last_ts.tz_convert(ET).to_pydatetime()
                if last_ts < client.require_end_dt:
                    stale = True
            except Exception:
                stale = True

        if stale:
            need.append(s)

    if not need:
        logger.info(f"Prefetch intraday: all cached (interval={interval}, period={period})")
        return

    logger.info(f"Prefetch intraday: downloading {len(need)}/{len(symbols)} uncached tickers...")

    for i in tqdm(range(0, len(need), batch_size), desc="Prefetch intraday", leave=False):
        chunk = need[i:i + batch_size]
        try:
            data = yf.download(
                tickers=chunk,
                interval=interval,
                period=period,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False
            )
            if data is None or data.empty:
                continue

            def _store_one(tkr: str, dft: pd.DataFrame):
                if dft is None or dft.empty:
                    return
                idx = pd.to_datetime(dft.index)
                if getattr(idx, "tz", None) is None:
                    idx = idx.tz_localize(UTC)
                idx = idx.tz_convert(ET)

                dft = dft.copy()
                dft.index = idx
                dft.columns = [c.lower() for c in dft.columns]

                out = pd.DataFrame({
                    "open": dft.get("open"),
                    "high": dft.get("high"),
                    "low": dft.get("low"),
                    "close": dft.get("close"),
                    "volume": dft.get("volume"),
                }).dropna(subset=["open", "high", "low", "close"])

                cache.set(client._intraday_cache_key(tkr, interval, period), out)

            if isinstance(data.columns, pd.MultiIndex):
                for tkr in chunk:
                    if tkr in data.columns.get_level_values(0):
                        _store_one(tkr, data[tkr].dropna(how="all"))
            else:
                _store_one(chunk[0], data.dropna(how="all"))

        except Exception:
            continue


# ============================================================
# INDICATORS
# ============================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def resample_ohlcv_intraday(df: pd.DataFrame, session_open: dt, rule: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    o = df["open"].resample(rule, origin=session_open, label="right", closed="left").first()
    h = df["high"].resample(rule, origin=session_open, label="right", closed="left").max()
    l = df["low"].resample(rule, origin=session_open, label="right", closed="left").min()
    c = df["close"].resample(rule, origin=session_open, label="right", closed="left").last()
    v = df["volume"].resample(rule, origin=session_open, label="right", closed="left").sum()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna(subset=["open"])

def compute_session_vwap(df: pd.DataFrame) -> Tuple[float, pd.Series]:
    if df.empty:
        return np.nan, pd.Series(dtype=float)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["volume"].cumsum()
    vwap_series = (cum_pv / cum_vol.replace(0, np.nan)).ffill()
    last_vwap = float(vwap_series.iloc[-1]) if len(vwap_series) else np.nan
    return last_vwap, vwap_series

def compute_vwap_bands(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {"vwap": np.nan, "sigma": np.nan, "vwap_p1": np.nan, "vwap_p2": np.nan}
    vwap_last, vwap_series = compute_session_vwap(df)
    if not np.isfinite(vwap_last) or vwap_series is None or vwap_series.empty:
        return {"vwap": vwap_last, "sigma": np.nan, "vwap_p1": np.nan, "vwap_p2": np.nan}
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vws = vwap_series.reindex(tp.index).ffill()
    resid = (tp - vws).dropna()
    sigma = float(resid.std(ddof=0)) if len(resid) >= 5 else np.nan
    vwap_p1 = float(vwap_last + sigma) if np.isfinite(sigma) else np.nan
    vwap_p2 = float(vwap_last + 2.0 * sigma) if np.isfinite(sigma) else np.nan
    return {"vwap": float(vwap_last), "sigma": sigma, "vwap_p1": vwap_p1, "vwap_p2": vwap_p2}

def vwap_zone(entry: float, vwap: float, vwap_p1: float, vwap_p2: float) -> str:
    if not (np.isfinite(entry) and np.isfinite(vwap)):
        return "UNKNOWN"
    if entry < vwap:
        return "BELOW_VWAP"
    if np.isfinite(vwap_p1) and entry <= vwap_p1:
        return "ACCUMULATION"
    if np.isfinite(vwap_p2) and entry > vwap_p2:
        return "OVEREXTENDED"
    return "TREND"

def linear_slope(y: pd.Series, lookback: int = 5) -> float:
    if y is None or len(y) < lookback:
        return 0.0
    yy = y.iloc[-lookback:].values
    x = np.arange(len(yy)) - np.mean(np.arange(len(yy)))
    denom = (x * x).sum()
    return float((x * (yy - yy.mean())).sum() / denom) if denom != 0 else 0.0

def compute_relative_strength_series(ticker_close: pd.Series, bench_close: pd.Series) -> pd.Series:
    if ticker_close.empty or bench_close.empty:
        return pd.Series(dtype=float)

    def _ret(s):
        s = s.dropna()
        if s.empty:
            return pd.Series(dtype=float)
        return (s / float(s.iloc[0])) - 1.0

    t_ret = _ret(ticker_close)
    b = bench_close.reindex(t_ret.index, method="ffill").reindex(t_ret.index, method="ffill")
    b_ret = _ret(b).reindex(t_ret.index, method="ffill").fillna(0)
    return (t_ret - b_ret).dropna()

def compute_rs_metrics(rs: pd.Series) -> Tuple[float, float]:
    if rs is None or rs.empty:
        return 0.0, 0.0
    return float(rs.iloc[-1]), linear_slope(rs, lookback=min(6, len(rs)))

def compute_macd_state(close_series: pd.Series, min_bars: int, require_hist_rising: bool) -> Dict[str, Any]:
    close_series = close_series.dropna()
    if len(close_series) < min_bars:
        return {"ok": False, "hist_rising": False, "bullish": False}
    macd_line, signal_line, hist = macd(close_series)
    bullish_now = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
    hist_rising = bool(hist.iloc[-1] > hist.iloc[-2]) if len(hist) >= 2 else False
    ok = bullish_now and (hist_rising if require_hist_rising else True)
    return {"ok": bool(ok), "hist_rising": bool(hist_rising), "bullish": bool(bullish_now)}

# ============================================================
# NEWS (Yahoo via yfinance) - optional
# ============================================================
news_rl = RateLimiter(60)
NEGATIVE_NEWS_KEYWORDS = [
    "offering", "secondary offering", "dilution", "bankruptcy", "chapter 11",
    "sec investigation", "investigation", "lawsuit", "fraud",
    "guidance cut", "cuts guidance", "misses", "downgrade",
    "going concern", "restatement", "halted", "delisting",
]
POSITIVE_NEWS_KEYWORDS = ["raises guidance", "beats", "upgrade", "contract", "partnership", "record revenue"]

def fetch_yfinance_news_risk(ticker: str) -> Dict[str, Any]:
    cache_key = f"news_yf_nse_v2_{ticker}_{NEWS_LOOKBACK_HOURS}"
    cached = cache.get(cache_key, ttl_seconds=30 * 60)
    if cached is not None:
        return cached

    try:
        news_rl.wait()
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []

    cutoff = dt.now(UTC) - timedelta(hours=NEWS_LOOKBACK_HOURS)
    titles: List[str] = []
    for it in items:
        title = (it.get("title") or "").strip()
        ts = it.get("providerPublishTime", None)
        pub = None
        if ts is not None:
            try:
                pub = dt.fromtimestamp(int(ts), tz=UTC)
            except:
                pub = None
        if title and (pub is None or pub >= cutoff):
            titles.append(title)

    neg_hits = sum(any(k in t.lower() for k in NEGATIVE_NEWS_KEYWORDS) for t in titles)
    pos_hits = sum(any(k in t.lower() for k in POSITIVE_NEWS_KEYWORDS) for t in titles)

    news_count = len(titles)
    if neg_hits > 0:
        flag = "NEGATIVE"
    elif news_count > 0:
        flag = "OK"
    else:
        flag = "NONE"

    score = 0.0
    if news_count >= 5:
        score += 0.25
    if pos_hits > 0 and neg_hits == 0:
        score += 0.25
    if neg_hits > 0:
        score -= 1.0

    out = {
        "news_score": float(score),
        "news_count": int(news_count),
        "news_flag": flag,
        "news_latest": (titles[0] if titles else "")[:180],
        "catalyst": "NEWS" if news_count > 0 else "NONE",
    }
    cache.set(cache_key, out)
    return out

# ============================================================
# EARNINGS BLACKOUT (best-effort)
# ============================================================
earnings_rl = RateLimiter(60)

def has_earnings_soon(ticker: str) -> bool:
    if not ENABLE_EARNINGS_BLACKOUT:
        return False
    cache_key = f"earnings_nse_v3_{ticker}_{EARNINGS_BLACKOUT_DAYS}"
    cached = cache.get(cache_key, ttl_seconds=6 * 3600)
    if cached is not None:
        return bool(cached)

    try:
        earnings_rl.wait()
        cal = yf.Ticker(ticker).calendar
        if cal is None or cal.empty:
            cache.set(cache_key, False)
            return False

        if "Earnings Date" in cal.index:
            val = cal.loc["Earnings Date"].values
        else:
            val = cal.values.flatten()

        if val is None or len(val) == 0:
            cache.set(cache_key, False)
            return False

        earn_dt = pd.to_datetime(val[0]).to_pydatetime()
        if earn_dt.tzinfo is not None:
            earn_dt = earn_dt.astimezone(UTC).replace(tzinfo=None)

        today = dt.now().replace(tzinfo=None)
        soon = abs((earn_dt.date() - today.date()).days) <= EARNINGS_BLACKOUT_DAYS
        cache.set(cache_key, soon)
        return bool(soon)
    except Exception:
        cache.set(cache_key, False)
        return False

# ============================================================
# SECTOR (yfinance info) - cached
# ============================================================
sector_rl = RateLimiter(60)

def get_sector_yf(ticker: str) -> str:
    cache_key = f"sector_nse_v3_{ticker}"
    cached = cache.get(cache_key, ttl_seconds=7 * 24 * 3600)
    if cached is not None:
        return str(cached)

    sec = "Unknown"
    try:
        sector_rl.wait()
        info = yf.Ticker(ticker).info or {}
        sec = info.get("sector") or info.get("industry") or "Unknown"
        if not isinstance(sec, str) or not sec.strip():
            sec = "Unknown"
        sec = sec.strip()
    except Exception:
        sec = "Unknown"

    cache.set(cache_key, sec)
    return sec

# ============================================================
# ATR (daily) helpers
# ============================================================
def compute_atr_table(daily_df: pd.DataFrame, atr_len: int = ATR_LEN) -> pd.DataFrame:
    if daily_df is None or daily_df.empty:
        return pd.DataFrame(columns=["ticker", "date", "atr"])
    df = daily_df.copy()
    df = df.dropna(subset=["high","low","close"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["ticker","date"])

    out_rows = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date")
        high = g["high"].astype(float).values
        low  = g["low"].astype(float).values
        close = g["close"].astype(float).values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr = pd.Series(tr).rolling(atr_len, min_periods=max(5, atr_len//2)).mean().values
        for d, a in zip(g["date"].values, atr):
            out_rows.append({"ticker": tkr, "date": d, "atr": float(a) if np.isfinite(a) else np.nan})
    return pd.DataFrame(out_rows)

def build_atr_lookup(atr_tbl: pd.DataFrame) -> Dict[str, Tuple[List[date], List[float]]]:
    out: Dict[str, Tuple[List[date], List[float]]] = {}
    if atr_tbl is None or atr_tbl.empty:
        return out
    atr_tbl = atr_tbl.dropna(subset=["atr"])
    for tkr, g in atr_tbl.groupby("ticker", sort=False):
        g = g.sort_values("date")
        out[tkr] = (g["date"].tolist(), g["atr"].astype(float).tolist())
    return out

def atr_asof(atr_lookup: Dict[str, Tuple[List[date], List[float]]], ticker: str, asof_date: date) -> float:
    if ticker not in atr_lookup:
        return np.nan
    ds, av = atr_lookup[ticker]
    if not ds:
        return np.nan
    i = bisect_right(ds, asof_date) - 1
    return float(av[i]) if i >= 0 else np.nan

# ============================================================
# CONSISTENT STOP FUNCTION (ROBUSTNESS FIX)
# - Used by BOTH allocation AND training simulation to avoid mismatch.
# ============================================================
def compute_stop_auto(
    entry: float,
    last15_low: float,
    session_low: float,
    buffer: float,
    atr14: float,
    range_pre: float = np.nan,
    atr_intra: float = np.nan,
    vol_mult: float = 1.0,
) -> float:
    """Compute a long stop price with practical intraday behavior.

    Components (then clamped):
      - Structural: below pre-entry lows (last-15m low and/or session low) minus buffer
      - ATR14: daily volatility-aware floor (prevents overly tight stops on large names)
      - Intraday floors:
          * pre-entry range floor (reacts to noisy opens)
          * intraday ATR floor (reacts to sudden spikes even mid-day)
      - Vol clustering multiplier: widens floors in known volatile windows (open/close) and
        during volatility shocks (intraday ATR >> expected).

    Note: Lower stop = wider risk. We start with the *tighter* of structural vs ATR14,
    then widen only if the intraday floors say the stop is unrealistically tight.
    """
    entry = float(entry)
    # --- Base candidates ---
    structural_low = last15_low
    if np.isfinite(session_low):
        structural_low = min(structural_low, session_low)
    structural = float(structural_low - buffer)

    atr_stop = float("-inf")
    if np.isfinite(atr14) and atr14 > 0:
        atr_stop = float(entry - ATR_STOP_MULT * atr14)

    # Pick a "reasonable tight" baseline (higher price = tighter stop)
    stop = float(max(structural, atr_stop))

    # --- Intraday widening floors ---
    vm = float(vol_mult) if np.isfinite(vol_mult) else 1.0
    vm = _clamp(vm, 0.75, 2.0)

    floors = []

    if ENABLE_DYNAMIC_STOP_FROM_INTRA_RANGE and np.isfinite(range_pre) and range_pre > 0:
        floors.append(float(STOP_INTRA_RANGE_MULT * range_pre * vm))

    if ENABLE_INTRADAY_ATR and np.isfinite(atr_intra) and atr_intra > 0:
        floors.append(float(INTRA_ATR_STOP_MULT * atr_intra * vm))

    if floors:
        floor_rps = float(max(floors))
        rps = float(entry - stop)
        if rps < floor_rps:
            stop = float(entry - floor_rps)

    # --- Clamp stop distance as % of entry ---
    if entry > 0:
        stop_pct = float((entry - stop) / entry)
        if stop_pct > STOP_MAX_PCT:
            stop = float(entry * (1.0 - STOP_MAX_PCT))
        if stop_pct < STOP_MIN_PCT:
            stop = float(entry * (1.0 - STOP_MIN_PCT))

    return float(stop)

def get_market_breadth_asof_intraday(
    universe: List[str],
    d: date,
    as_of_et: dt,
    interval: str = "5m",
    period: str = "60d",
    batch_size: int = 120,
) -> Dict[str, Any]:
    if not universe:
        return {"ad_ratio": np.nan, "adv": 0, "dec": 0, "n": 0, "note": "breadth: empty universe"}

    session_open, session_close = get_session_open_close(d)
    cutoff = min(as_of_et, session_close)

    adv = 0
    dec = 0
    n = 0

    def _fix_index_to_et(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        idx = pd.to_datetime(idx)
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize(UTC)
        return idx.tz_convert(ET)

    for i in range(0, len(universe), batch_size):
        chunk = universe[i:i + batch_size]
        try:
            _http_rl.wait()
            data = yf.download(
                tickers=chunk,
                interval=interval,
                period=period,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
            if data is None or data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                for tkr in chunk:
                    if tkr not in data.columns.get_level_values(0):
                        continue
                    dft = data[tkr].dropna(how="all")
                    if dft.empty:
                        continue

                    idx = _fix_index_to_et(dft.index)
                    dft = dft.copy()
                    dft.index = idx

                    day = dft[(dft.index >= session_open) & (dft.index <= cutoff)]
                    if day.empty:
                        continue

                    o = float(day["Open"].iloc[0])
                    c = float(day["Close"].iloc[-1])
                    if not (np.isfinite(o) and np.isfinite(c)):
                        continue

                    if c > o:
                        adv += 1
                    elif c < o:
                        dec += 1
                    n += 1
            else:
                dft = data.dropna(how="all")
                if not dft.empty:
                    idx = _fix_index_to_et(dft.index)
                    dft = dft.copy()
                    dft.index = idx
                    day = dft[(dft.index >= session_open) & (dft.index <= cutoff)]
                    if not day.empty:
                        o = float(day["Open"].iloc[0])
                        c = float(day["Close"].iloc[-1])
                        if np.isfinite(o) and np.isfinite(c):
                            adv += int(c > o)
                            dec += int(c < o)
                            n += 1

        except Exception:
            continue

    denom = adv + dec
    ratio = float(adv / denom) if denom > 0 else np.nan
    note = f"breadth(asof) A/D={ratio:.3f} (adv={adv}, dec={dec}, n={n})" if np.isfinite(ratio) else f"breadth(asof) A/D=NaN (adv={adv}, dec={dec}, n={n})"
    return {"ad_ratio": ratio, "adv": adv, "dec": dec, "n": n, "note": note}


# ============================================================
# Resolve analysis datetime (LIVE vs SIMULATION)
# ============================================================
if SIMULATION_MODE:
    _dt_raw = ET.localize(SIM_TEST_DT) if getattr(SIM_TEST_DT, 'tzinfo', None) is None else SIM_TEST_DT.astimezone(ET)
    logger.info(f"🧪 SIMULATION MODE: Testing as of {_dt_raw}")
else:
    _dt_raw = dt.now(tz=ET)

# If the requested date is not a trading day, roll back to the most recent trading session.
_requested_date = _dt_raw.date()
analysis_date = _requested_date if is_trading_day(_requested_date) else get_last_trading_day(_dt_raw)
if analysis_date != _requested_date:
    logger.warning(f"Requested date {_requested_date} is not an NSE trading day; using {analysis_date} instead.")

# as_of uses the requested clock-time on the resolved trading date.
as_of_et = dt_combine(analysis_date, _dt_raw.time())
session_open_et, session_close_et = get_session_open_close(analysis_date)

# Clamp as_of into the session so downstream slices always exist.
if as_of_et < session_open_et:
    logger.warning(f"as_of {as_of_et.time()} is before market open; clamping to {session_open_et.time()}.")
    as_of_et = session_open_et
if as_of_et > session_close_et:
    logger.warning(f"as_of {as_of_et.time()} is after market close; clamping to {session_close_et.time()}.")
    as_of_et = session_close_et

logger.info(f"Analysis date: {analysis_date} | as_of: {as_of_et.time()} | session_close: {session_close_et.time()}")
_prev_list = get_trading_days(analysis_date, 1)
prev_session = _prev_list[-1] if _prev_list else (analysis_date - timedelta(days=1))
prev_session_date = prev_session  # alias (date)
logger.info(f"Previous session: {prev_session_date} | RVOL lookback: {cfg.rvol_lookback_sessions} sessions")

# Cache freshness: if your intraday cache ends before the required date, refetch.
# This fixes the "old cached day" issue when you change SIM_TEST_DT.
client.force_refresh = False
client.require_end_dt = session_close_et if SIMULATION_MODE else None

# Load NSE universe (source) and select the scan universe.
logger.info("Loading NSE stock universe...")
universe_df = client.list_stocks()
universe_scan = universe_df['ticker'].astype(str).head(MAX_UNIVERSE).tolist()
logger.info(f"Final universe: {len(universe_scan)} stocks (NSE)")

# Map ticker -> name for display
name_map: Dict[str, str] = {}
try:
    if isinstance(universe_df, pd.DataFrame) and "ticker" in universe_df.columns:
        if "name" in universe_df.columns:
            name_map = dict(zip(universe_df["ticker"].astype(str), universe_df["name"].astype(str)))
        elif "Name" in universe_df.columns:
            name_map = dict(zip(universe_df["ticker"].astype(str), universe_df["Name"].astype(str)))
except Exception:
    name_map = {}

# Opening range end datetime (IST) used for ORH computations
or_end_et = dt_combine(analysis_date, cfg.opening_range_end, ET)

# Historical days for RVOL stats (prior sessions only)
hist_days = get_trading_days(analysis_date, cfg.rvol_lookback_sessions)

# Daily data for ATR + optional liquidity computations (best-effort)
daily_start = analysis_date - timedelta(days=max(120, cfg.rvol_lookback_sessions * 6))
daily_end = analysis_date
daily_symbols = sorted(set(universe_scan + [REGIME_TICKER]))
logger.info(f"Fetching daily data for ATR: symbols={len(daily_symbols)}, start={daily_start}, end={daily_end}")
daily_data = client.get_daily_data(daily_symbols, daily_start, daily_end)


breadth = get_market_breadth_asof_intraday(universe_scan, analysis_date, as_of_et, interval="5m", period="60d")
breadth_ratio = float(breadth["ad_ratio"]) if np.isfinite(breadth["ad_ratio"]) else np.nan
breadth_strict = bool(np.isfinite(breadth_ratio) and breadth_ratio < BREADTH_NEGATIVE_RATIO)
logger.info("📊 " + breadth["note"] + (f" | STRICT(+{int(BREADTH_STRICTEN_FACTOR*100)}%)=True" if breadth_strict else " | STRICT=False"))

def effective_take_thresholds(strict: bool) -> Dict[str, Any]:
    if not strict:
        return dict(
            min_ev=TAKE_MIN_EV_R,
            max_psl=TAKE_MAX_P_SL,
            min_ptp_hit=TAKE_MIN_P_TP_GIVEN_HIT,
            max_unc=TAKE_MAX_UNCERTAINTY,
            min_emp=TAKE_MIN_EMP_N
        )
    return dict(
        min_ev=TAKE_MIN_EV_R * (1.0 + BREADTH_STRICTEN_FACTOR),
        max_psl=TAKE_MAX_P_SL * (1.0 - BREADTH_STRICTEN_FACTOR),
        min_ptp_hit=TAKE_MIN_P_TP_GIVEN_HIT * (1.0 + BREADTH_STRICTEN_FACTOR),
        max_unc=TAKE_MAX_UNCERTAINTY * (1.0 - BREADTH_STRICTEN_FACTOR),
        min_emp=int(math.ceil(TAKE_MIN_EMP_N * (1.0 + BREADTH_STRICTEN_FACTOR)))
    )

TAKE_EFF = effective_take_thresholds(breadth_strict)

# ============================================================
# REGIME + BASELINES
# ============================================================
logger.info(f"Fetching regime data for {REGIME_TICKER}...")
regime_1m = client.get_intraday_data(REGIME_TICKER, analysis_date, interval="1m", period="8d")

def compute_session_vwap_last(df: pd.DataFrame) -> Tuple[float, float]:
    if df.empty:
        return np.nan, np.nan
    vwap_val, _ = compute_session_vwap(df)
    last_val = float(df["close"].iloc[-1]) if not df.empty else np.nan
    return vwap_val, last_val

if regime_1m.empty:
    regime_score = 0.5
    regime_note = "Regime=0.50 | No data"
else:
    cut = regime_1m[(regime_1m.index >= session_open_et) & (regime_1m.index < as_of_et)]
    vwap_val, last_val = compute_session_vwap_last(cut) if not cut.empty else (np.nan, np.nan)
    above = bool(last_val > vwap_val) if np.isfinite(last_val) and np.isfinite(vwap_val) else None
    regime_score = 0.70 if above else 0.30
    regime_note = f"Regime={regime_score:.2f} | {REGIME_TICKER}>VWAP={above}"
logger.info(regime_note)

bench_today_1m = regime_1m
bench_today_5m = resample_ohlcv_intraday(
    bench_today_1m[(bench_today_1m.index >= session_open_et) & (bench_today_1m.index < as_of_et)],
    session_open_et, "5min"
) if not bench_today_1m.empty else pd.DataFrame()

bench_today_15m = resample_ohlcv_intraday(
    bench_today_1m[(bench_today_1m.index >= session_open_et) & (bench_today_1m.index < as_of_et)],
    session_open_et, "15min"
) if not bench_today_1m.empty else pd.DataFrame()

# ============================================================
# EARLY BLOCK FILTERS (Earnings + News hard blocks)
# ============================================================
def prefilter_blocked_universe(universe: List[str]) -> Tuple[List[str], Dict[str, str]]:
    if not universe:
        return [], {}
    reason: Dict[str, str] = {}

    if ENABLE_EARNINGS_BLACKOUT:
        logger.info("Early filter: earnings blackout...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(has_earnings_soon, t): t for t in universe}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Earnings blk", leave=False):
                t = futs[fut]
                try:
                    if bool(fut.result()):
                        reason[t] = f"EARNINGS(+/-{EARNINGS_BLACKOUT_DAYS}d)"
                except Exception:
                    pass

    if ENABLE_NEWS_FILTER and NEWS_NEGATIVE_HARD_BLOCK:
        logger.info("Early filter: NEGATIVE news hard-block...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(fetch_yfinance_news_risk, t): t for t in universe if t not in reason}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="News blk", leave=False):
                t = futs[fut]
                try:
                    ns = fut.result()
                    if (ns or {}).get("news_flag") == "NEGATIVE":
                        reason[t] = "NEGATIVE_NEWS"
                except Exception:
                    pass

    filtered = [t for t in universe if t not in reason]
    if reason:
        logger.info(f"Early blocks: {len(reason)} removed | remaining={len(filtered)}")
    return filtered, reason

universe_scan, early_block_reasons = prefilter_blocked_universe(universe_scan)

# ============================================================
# STAGE 1 HELPERS
# ============================================================
def compute_opening_range_high(df_1m_cut: pd.DataFrame) -> float:
    w = df_1m_cut[(df_1m_cut.index >= session_open_et) & (df_1m_cut.index < or_end_et)]
    return float(w["high"].max()) if not w.empty else np.nan

def compute_last_closed_15m_low_today(df_1m: pd.DataFrame) -> float:
    cut = df_1m[(df_1m.index >= session_open_et) & (df_1m.index < as_of_et)]
    bars15 = resample_ohlcv_intraday(cut, session_open_et, "15min")
    return float(bars15["low"].iloc[-1]) if not bars15.empty else np.nan

# ============================================================
# STAGE 1: Momentum + VWAP σ-bands + RS + MACD
# ============================================================
def analyze_ticker_stage1(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        df_today = client.get_intraday_data(ticker, analysis_date, interval="1m", period="8d")
        if df_today.empty:
            return None

        df_cut = df_today[(df_today.index >= session_open_et) & (df_today.index < as_of_et)]
        if df_cut.empty:
            return None

        entry = float(df_cut["close"].iloc[-1])
        if not np.isfinite(entry) or entry < cfg.price_min:
            return None

        open_px = float(df_cut["open"].iloc[0]) if len(df_cut) else np.nan
        session_low = float(df_cut["low"].min())
        session_high = float(df_cut["high"].max())
        intraday_range = float(session_high - session_low) if np.isfinite(session_high) and np.isfinite(session_low) else np.nan

        orh = compute_opening_range_high(df_cut)

        vb = compute_vwap_bands(df_cut)
        day_vwap = float(vb["vwap"]) if np.isfinite(vb["vwap"]) else entry
        vwap_sigma = float(vb["sigma"]) if np.isfinite(vb["sigma"]) else np.nan
        vwap_p1 = float(vb["vwap_p1"]) if np.isfinite(vb["vwap_p1"]) else np.nan
        vwap_p2 = float(vb["vwap_p2"]) if np.isfinite(vb["vwap_p2"]) else np.nan
        zone = vwap_zone(entry, day_vwap, vwap_p1, vwap_p2)

        if cfg.stage1_require_vwap and not (entry >= day_vwap * (1.0 - cfg.vwap_tol_pct)):
            return None

        bars5_today = resample_ohlcv_intraday(df_cut, session_open_et, "5min")
        if bars5_today.empty:
            return None
        bars15_today = resample_ohlcv_intraday(df_cut, session_open_et, "15min")

        if cfg.stage1_require_orh and np.isfinite(orh):
            last5_close = float(bars5_today["close"].iloc[-1])
            if not (entry >= orh * (1.0 - cfg.orh_tol_pct) and last5_close >= orh * (1.0 - cfg.orh_tol_pct)):
                return None

        df_prev = client.get_intraday_data(ticker, prev_session_date, interval="1m", period="8d")
        session_open_prev, _ = get_session_open_close(prev_session_date)
        bars5_prev = resample_ohlcv_intraday(df_prev, session_open_prev, "5min") if not df_prev.empty else pd.DataFrame()
        bars15_prev = resample_ohlcv_intraday(df_prev, session_open_prev, "15min") if not df_prev.empty else pd.DataFrame()

        bars5 = pd.concat([bars5_prev, bars5_today]).sort_index()
        bars15 = pd.concat([bars15_prev, bars15_today]).sort_index()

        macd5 = compute_macd_state(bars5["close"], cfg.macd_min_bars_5m, cfg.require_macd_hist_rising)
        macd15 = compute_macd_state(bars15["close"], cfg.macd_min_bars_15m_soft, cfg.require_macd_hist_rising)

        macd_ok = bool(macd5["ok"] and (macd15["ok"] if len(bars15) >= cfg.macd_min_bars_15m_soft else True))
        if cfg.stage1_require_macd and not macd_ok:
            return None

        rs5_series = compute_relative_strength_series(bars5_today["close"], bench_today_5m["close"]) if not bench_today_5m.empty else pd.Series(dtype=float)
        rs15_series = compute_relative_strength_series(bars15_today["close"], bench_today_15m["close"]) if not bench_today_15m.empty else pd.Series(dtype=float)
        rs5_last, rs5_slope = compute_rs_metrics(rs5_series)
        rs15_last, rs15_slope = compute_rs_metrics(rs15_series)

        cum_vol_today = float(df_cut["volume"].sum())

        or_break_pct = safe_div(entry - orh, orh, 0.0) if np.isfinite(orh) else 0.0
        vwap_dist_pct = safe_div(entry - day_vwap, day_vwap, 0.0)
        vwap_dist_sig = safe_div(entry - day_vwap, vwap_sigma, 0.0) if np.isfinite(vwap_sigma) and vwap_sigma > 0 else np.nan

        last15_low = compute_last_closed_15m_low_today(df_today)
        ret_from_open = safe_div(entry - open_px, open_px, 0.0) if np.isfinite(open_px) and open_px > 0 else np.nan

        return {
            "ticker": ticker,
            "name": name_map.get(ticker, ticker),
            "entry": float(entry),
            "open_px": float(open_px) if np.isfinite(open_px) else np.nan,
            "ret_from_open": float(ret_from_open) if np.isfinite(ret_from_open) else np.nan,

            "orh": float(orh) if np.isfinite(orh) else np.nan,

            "day_vwap": float(day_vwap),
            "vwap_sigma": float(vwap_sigma) if np.isfinite(vwap_sigma) else np.nan,
            "vwap_p1": float(vwap_p1) if np.isfinite(vwap_p1) else np.nan,
            "vwap_p2": float(vwap_p2) if np.isfinite(vwap_p2) else np.nan,
            "vwap_zone": zone,

            "or_break_pct": float(or_break_pct),
            "vwap_dist_pct": float(vwap_dist_pct),
            "vwap_dist_sig": float(vwap_dist_sig) if np.isfinite(vwap_dist_sig) else np.nan,

            "cum_vol_today": float(cum_vol_today),
            "intraday_range": float(intraday_range) if np.isfinite(intraday_range) else np.nan,

            "macd_ok": bool(macd_ok),
            "macd_hist_rising_ok": bool(macd5["hist_rising"] and macd15["hist_rising"]),
            "rs5_last": float(rs5_last),
            "rs5_slope": float(rs5_slope),
            "rs15_last": float(rs15_last),
            "rs15_slope": float(rs15_slope),

            "last15_low": float(last15_low) if np.isfinite(last15_low) else np.nan,
            "session_low": float(float(df_cut["low"].min())) if not df_cut.empty else np.nan,
        }
    except Exception:
        return None

def run_stage1(universe: List[str]) -> List[Dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(analyze_ticker_stage1, t): t for t in universe}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage1", leave=False):
            r = fut.result()
            if r is not None:
                out.append(r)
    return out

logger.info("Stage 1: Momentum + VWAP σ-band tagging...")
stage1_results = run_stage1(universe_scan)
logger.info(f"Stage 1 pass: {len(stage1_results)} stocks")

if len(stage1_results) == 0:
    logger.warning("Stage-1 returned 0. Disabling hard gates and rerunning...")
    cfg.stage1_require_vwap = False
    cfg.stage1_require_macd = False
    cfg.stage1_require_orh = False
    stage1_results = run_stage1(universe_scan)
    logger.info(f"Stage 1 after fallback: {len(stage1_results)} stocks")


# Prefetch 5m once for all Stage-1 pass tickers (massive speedup)
stage1_tickers = [r["ticker"] for r in stage1_results]
prefetch_intraday_batch(stage1_tickers, interval=RVOL_INTERVAL, period=RVOL_PERIOD, batch_size=80)


# ============================================================
# STAGE 2: RVOL (+ Volume Z-score stats)
# ============================================================
def compute_rvol_and_volstats_for_ticker(ticker: str, numerator_vol: float) -> Dict[str, float]:
    try:
        cutoff_time = as_of_et.time()

        raw = client._fetch_intraday_raw(ticker, interval=RVOL_INTERVAL, period=RVOL_PERIOD, ttl=INTRADAY_CACHE_TTL_5M)
        if raw is None or raw.empty:
            return {"rvol": np.nan, "vol_mean": np.nan, "vol_std": np.nan, "vol_z": np.nan}

        # keep only bars up to cutoff time for each day
        df = raw.copy()
        df = df[df.index.time < cutoff_time]
        if df.empty:
            return {"rvol": np.nan, "vol_mean": np.nan, "vol_std": np.nan, "vol_z": np.nan}

        # cum volume per day up to cutoff
        vol_by_day = df.groupby(df.index.date)["volume"].sum()

        # only use the prior sessions you want
        series = vol_by_day.reindex(hist_days).dropna()
        min_obs = max(3, int(cfg.rvol_lookback_sessions * 0.4))
        if len(series) < min_obs:
            return {"rvol": np.nan, "vol_mean": np.nan, "vol_std": np.nan, "vol_z": np.nan}

        denom_mean = float(series.mean())
        denom_std = float(series.std(ddof=0))
        rvol = float(numerator_vol / denom_mean) if denom_mean > 0 else np.nan
        vol_z = float((numerator_vol - denom_mean) / denom_std) if denom_std > 1e-9 else np.nan

        return {"rvol": rvol, "vol_mean": denom_mean, "vol_std": denom_std, "vol_z": vol_z}
    except Exception:
        return {"rvol": np.nan, "vol_mean": np.nan, "vol_std": np.nan, "vol_z": np.nan}


logger.info("Stage 2: RVOL filter...")
stage2_rows = []
from concurrent.futures import ThreadPoolExecutor, as_completed
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(compute_rvol_and_volstats_for_ticker, r["ticker"], r["cum_vol_today"]): r for r in stage1_results}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="RVOL", leave=False):
        row = futures[fut]
        pack = fut.result()
        rvol = pack.get("rvol", np.nan)
        if np.isfinite(rvol) and rvol >= cfg.rvol_min:
            x = dict(row)
            x["rvol"] = float(rvol)
            x["vol_mean"] = float(pack.get("vol_mean", np.nan))
            x["vol_std"]  = float(pack.get("vol_std", np.nan))
            x["vol_z"]    = float(pack.get("vol_z", np.nan))
            stage2_rows.append(x)

logger.info(f"Stage 2 pass (RVOL >= {cfg.rvol_min}): {len(stage2_rows)}")

if len(stage2_rows) == 0:
    print("\n" + "=" * 80)
    print("NO CANDIDATES FOUND")
    print("=" * 80)
    print(f"Stage 1: {len(stage1_results)} → Stage 2: 0")
    print("Tip: lower rvol_min (e.g. 0.5), increase MAX_UNIVERSE, or reduce avg10_dollar_vol_min")
    print("=" * 80)
    raise SystemExit

# ============================================================
# ATR lookup for current allocation/inference
# ============================================================
atr_tbl_now = compute_atr_table(daily_data, atr_len=ATR_LEN)
atr_lookup_now = build_atr_lookup(atr_tbl_now)

session_incomplete = bool(as_of_et < (session_close_et - timedelta(minutes=1)))
atr_cutoff_date = prev_session_date if session_incomplete else analysis_date

def atr_for_stop(ticker: str) -> float:
    return atr_asof(atr_lookup_now, ticker, atr_cutoff_date)

# ============================================================
# SCORING (includes VWAP zone alpha)
# ============================================================
def enrich_with_score(r: Dict[str, Any]) -> Dict[str, Any]:
    def score_breakout_vwap(or_break_pct, vwap_dist_pct):
        return 0.6 * clamp(or_break_pct / 0.01, 0, 1) + 0.4 * clamp(vwap_dist_pct / 0.005, 0, 1)

    def score_rvol(v):
        return clamp((v - 1.0) / 2.5, 0, 1)

    def score_macd(macd_ok, hist_rising):
        if macd_ok and hist_rising:
            return 1.0
        if macd_ok:
            return 0.6
        return 0.2

    zone = str(r.get("vwap_zone", "UNKNOWN"))
    zone_bonus = {
        "ACCUMULATION": 6.0,
        "TREND": 2.0,
        "OVEREXTENDED": -8.0,
        "BELOW_VWAP": -10.0,
        "UNKNOWN": 0.0
    }.get(zone, 0.0)

    a_break = score_breakout_vwap(r.get("or_break_pct", 0), r.get("vwap_dist_pct", 0))
    a_rvol = score_rvol(r.get("rvol", 1.0))
    a_macd = score_macd(bool(r.get("macd_ok")), bool(r.get("macd_hist_rising_ok")))

    stack_a = 15 * a_break + 15 * a_rvol + 5 * a_macd
    stack_b = 10.0
    stack_c = 5.0

    d_rs5 = clamp(r.get("rs5_last", 0) / 0.005, 0, 1) * 0.65 + (1 if r.get("rs5_slope", 0) > 0 else 0) * 0.35
    d_rs15 = clamp(r.get("rs15_last", 0) / 0.005, 0, 1) * 0.65 + (1 if r.get("rs15_slope", 0) > 0 else 0) * 0.35
    stack_d = 12 * d_rs5 + 8 * d_rs15
    stack_e = 5.0

    raw_score = stack_a + stack_b + stack_c + stack_d + stack_e + zone_bonus
    final_score = clamp(raw_score, 0, 100)

    out = dict(r)
    out.update({"conviction_score": float(final_score)})
    return out

logger.info("Scoring candidates...")
df_cand = pd.DataFrame([enrich_with_score(r) for r in stage2_rows])

if not df_cand.empty:
    df_cand = df_cand.sort_values("conviction_score", ascending=False).reset_index(drop=True)
    top_for_sector = df_cand.head(max(60, cfg.target_positions * PICK_BUFFER_MULT * 3)).copy()
    if not top_for_sector.empty:
        secs = []
        logger.info(f"Fetching sectors for top {len(top_for_sector)} tickers...")
        for t in top_for_sector["ticker"].tolist():
            secs.append({"ticker": t, "sector": get_sector_yf(t)})
        df_sec = pd.DataFrame(secs)
        df_cand = df_cand.merge(df_sec, on="ticker", how="left")
    df_cand["sector"] = df_cand["sector"].fillna("Unknown")

# ============================================================
# FILTER BY SCORE
# ============================================================
df_ranked = df_cand[df_cand["conviction_score"] >= cfg.score_min].copy()
df_ranked = df_ranked.sort_values("conviction_score", ascending=False).reset_index(drop=True)

# ============================================================
# SECTOR CAPS
# ============================================================
def apply_sector_caps(df: pd.DataFrame, max_per_sector: int, target: int) -> pd.DataFrame:
    picks = []
    counts: Dict[str, int] = {}
    for _, row in df.iterrows():
        sec = (row.get("sector", "Unknown") or "Unknown").strip() or "Unknown"
        if ENABLE_SECTOR_GUARDRAILS and max_per_sector > 0 and sec != "Unknown":
            if counts.get(sec, 0) >= max_per_sector:
                continue
        picks.append(row)
        if sec != "Unknown":
            counts[sec] = counts.get(sec, 0) + 1
        if len(picks) >= target:
            break
    return pd.DataFrame(picks)

df_pick = apply_sector_caps(df_ranked, cfg.max_per_sector, cfg.target_positions * PICK_BUFFER_MULT)

# ============================================================
# ALLOCATION (uses consistent compute_stop_auto)
# ============================================================
stage2_map = {r["ticker"]: r for r in stage2_rows}

alloc_rows = []
for _, r in df_pick.iterrows():
    tkr = r["ticker"]
    entry = float(r["entry"])

    last15_low = float(r.get("last15_low", np.nan))
    session_low = float(r.get("session_low", np.nan))
    atr14 = float(atr_for_stop(tkr))

    # Intraday ATR + vol clustering (time-of-day + volatility shock)
    atr_intra = np.nan
    vol_mult = 1.0
    if ENABLE_INTRADAY_ATR or ENABLE_VOL_CLUSTERING_TOD:
        try:
            df5 = client.get_intraday_data(tkr, analysis_date, interval="5m", period=PROB_PERIOD)
            if df5 is not None and not df5.empty:
                df5 = df5.copy()
                if df5.index.tz is None:
                    df5.index = df5.index.tz_localize(IST)
                else:
                    df5.index = df5.index.tz_convert(IST)
                df5d = df5[df5.index.date == analysis_date]
                pre5 = df5d[df5d.index <= as_of_et]
                atr_intra = intraday_atr_from_5m(pre5)
        except Exception:
            atr_intra = np.nan
        vol_mult = vol_cluster_multiplier(as_of_et.time(), atr_intra, atr14)

    stop = compute_stop_auto(
        entry=entry,
        last15_low=last15_low,
        session_low=session_low,
        buffer=float(cfg.stop_buffer_below_15m_low),
        atr14=atr14,
        range_pre=float(r.get("intraday_range", np.nan)),
        atr_intra=atr_intra,
        vol_mult=vol_mult,
    )

    if stop <= 0 or stop >= entry:
        continue

    tp = entry + float(cfg.rr_min) * (entry - stop)

    score = float(r.get("conviction_score", 0))
    risk_budget = 260.0 if score >= 70 else (220.0 if score >= 60 else 180.0)

    rps = entry - stop
    shares = int(risk_budget / rps) if rps > 0 else 0
    if shares <= 0:
        continue

    max_pos = 12000.0
    max_shares = int(max_pos / entry)
    shares = min(shares, max_shares)

    pos_value = shares * entry
    actual_risk = shares * rps
    if pos_value < cfg.min_pos_dollars:
        continue

    alloc_rows.append({
        "Ticker": tkr,
        "Name": name_map.get(tkr, tkr),
        "Sector": r.get("sector", "Unknown"),
        "Catalyst": r.get("catalyst", "NONE"),

        "VWAP": float(r.get("day_vwap", np.nan)),
        "VWAP_+1s": float(r.get("vwap_p1", np.nan)),
        "VWAP_+2s": float(r.get("vwap_p2", np.nan)),
        "VWAP_Zone": str(r.get("vwap_zone", "UNKNOWN")),

        "Open": float(r.get("open_px", np.nan)),
        "Entry": entry,
        "Stop": stop,
        "StopPct": float((entry - stop) / entry) if entry > 0 else np.nan,
        "ATR14": float(atr14) if np.isfinite(atr14) else np.nan,
        "Target": tp,

        "Shares": shares,
        "Position (₹)": pos_value,
        "Risk (₹)": actual_risk,
        "Score": score,
        "RVOL": float(r.get("rvol", np.nan)),
        "VolZ": float(r.get("vol_z", np.nan)),
        "NewsFlag": r.get("news_flag", "NONE"),
        "NewsLatest": r.get("news_latest", ""),
    })

df_alloc = pd.DataFrame(alloc_rows).sort_values("Score", ascending=False).head(cfg.target_positions)

# ============================================================
# PROBABILITY MODEL (ROBUSTNESS UPGRADES)
# Key changes:
#   1) Training simulation uses the SAME compute_stop_auto as allocation.
#   2) Empirical stats store MeanR_NONE so EV can include P_NONE realistically.
#   3) BestRR selection can skip unreachable targets (ATR / % cap).
# ============================================================
def prefetch_intraday_batch(
    symbols: List[str],
    interval: str,
    period: str,
    batch_size: int = 50,
    ttl_seconds: int = INTRADAY_CACHE_TTL_5M,
):
    symbols = [s for s in symbols if isinstance(s, str) and len(s) > 0]

    need = []
    for s in symbols:
        key = client._intraday_cache_key(s, interval, period)
        cached = None if client.force_refresh else cache.get(key, ttl_seconds=ttl_seconds)

        stale = False
        if cached is None or (not isinstance(cached, pd.DataFrame)) or cached.empty:
            stale = True
        elif client.require_end_dt is not None:
            # Ensure cached intraday extends past required simulation window
            try:
                last_ts = pd.Timestamp(cached.index.max())
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize(UTC)
                last_ts = last_ts.tz_convert(ET).to_pydatetime()
                if last_ts < client.require_end_dt:
                    stale = True
            except Exception:
                stale = True

        if stale:
            need.append(s)
    if not need:
        logger.info(f"Prefetch intraday: all cached (interval={interval}, period={period})")
        return

    logger.info(f"Prefetch intraday: downloading {len(need)}/{len(symbols)} uncached tickers...")

    for i in tqdm(range(0, len(need), batch_size), desc="Prefetch intraday", leave=False):
        chunk = need[i:i+batch_size]
        try:
            data = yf.download(
                tickers=chunk, interval=interval, period=period,
                group_by="ticker", auto_adjust=False, threads=True, progress=False
            )
            if data is None or data.empty:
                continue

            def _store_one(tkr: str, dft: pd.DataFrame):
                if dft is None or dft.empty:
                    return
                idx = pd.to_datetime(dft.index)
                if getattr(idx, "tz", None) is None:
                    idx = idx.tz_localize(UTC)
                idx = idx.tz_convert(ET)
                dft = dft.copy()
                dft.index = idx
                dft.columns = [c.lower() for c in dft.columns]
                out = pd.DataFrame({
                    "open": dft.get("open"),
                    "high": dft.get("high"),
                    "low": dft.get("low"),
                    "close": dft.get("close"),
                    "volume": dft.get("volume"),
                }).dropna(subset=["open", "high", "low", "close"])
                cache.set(client._intraday_cache_key(tkr, interval, period), out)

            if isinstance(data.columns, pd.MultiIndex):
                for tkr in chunk:
                    if tkr in data.columns.get_level_values(0):
                        _store_one(tkr, data[tkr].dropna(how="all"))
            else:
                _store_one(chunk[0], data.dropna(how="all"))

        except Exception:
            continue

def compute_orh(df: pd.DataFrame, d: date, or_end_clock: dtime) -> float:
    session_open, _ = get_session_open_close(d)
    or_end_dt = dt_combine(d, or_end_clock, ET)
    w = df[(df.index >= session_open) & (df.index < or_end_dt)]
    return float(w["high"].max()) if not w.empty else np.nan

def simulate_tp_sl_for_rrgrid(
    day_df: pd.DataFrame,
    d: date,
    entry_clock: dtime,
    stop_buffer: float,
    rr_grid: List[float],
    tie_policy: str,
    atr14: float
) -> Optional[Dict[str, Any]]:
    if day_df is None or day_df.empty:
        return None

    session_open, session_close = get_session_open_close(d)
    entry_dt = dt_combine(d, entry_clock, ET)
    df = day_df[(day_df.index >= session_open) & (day_df.index <= session_close)].copy()
    if df.empty:
        return None

    pre = df[df.index <= entry_dt]
    if pre.empty:
        return None

    entry_ts = pre.index[-1]
    entry = float(pre["close"].iloc[-1])
    if not np.isfinite(entry) or entry <= 0:
        return None

    # Compute structural anchors from PRE data only (no lookahead)
    bars15 = resample_ohlcv_intraday(df[df.index <= entry_ts], session_open, "15min")
    last15_low = float(bars15["low"].iloc[-1]) if not bars15.empty else np.nan
    session_low = float(pre["low"].min()) if not pre.empty else np.nan
    range_pre = float(pre["high"].max() - pre["low"].min()) if len(pre) else np.nan
    atr_intra = intraday_atr_from_5m(pre)
    vol_mult = vol_cluster_multiplier(entry_ts.time(), atr_intra, atr14)

    stop = compute_stop_auto(
        entry=entry,
        last15_low=last15_low,
        session_low=session_low,
        buffer=stop_buffer,
        atr14=atr14,
        range_pre=range_pre,
        atr_intra=atr_intra,
        vol_mult=vol_mult,
    )



    if not (np.isfinite(stop) and stop > 0 and stop < entry):
        return None

    rps = entry - stop
    if rps <= 0:
        return None

    fut = df[df.index > entry_ts]
    targets = {float(rr): entry + float(rr) * rps for rr in rr_grid}

    results = {}
    if fut.empty:
        for rr in rr_grid:
            rr = float(rr)
            eod_close = float(df["close"].iloc[-1])
            R = float((eod_close - entry) / rps) if np.isfinite(eod_close) else 0.0
            results[rr] = {"label": "NONE", "R": float(clamp(R, -2.0, max(3.0, rr))), "target": float(targets[rr])}
        return {"entry_ts": entry_ts, "entry": float(entry), "stop": float(stop), "rps": float(rps), "results": results}

    for rr in rr_grid:
        rr = float(rr)
        target = float(targets[rr])
        hit = "NONE"
        for _, row in fut.iterrows():
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            if np.isfinite(o):
                if o <= stop:
                    hit = "SL"; break
                if o >= target:
                    hit = "TP"; break
            sl_in = np.isfinite(l) and (l <= stop)
            tp_in = np.isfinite(h) and (h >= target)
            if sl_in and tp_in:
                hit = "TIE" if tie_policy.upper() == "HALF" else "SL"
                break
            elif sl_in:
                hit = "SL"; break
            elif tp_in:
                hit = "TP"; break

        eod_close = float(df["close"].iloc[-1])
        R = (eod_close - entry) / rps if rps > 0 and np.isfinite(eod_close) else 0.0
        if hit == "TP":
            R = rr
        elif hit == "SL":
            R = -1.0
        elif hit == "TIE":
            R = (rr - 1.0) / 2.0
        R = float(clamp(R, -2.0, max(3.0, rr)))
        results[rr] = {"label": hit, "R": R, "target": float(target)}
    return {"entry_ts": entry_ts, "entry": float(entry), "stop": float(stop), "rps": float(rps), "results": results}


# -------------------------------
# Historical MFE/MAE caps (targets reachability)
# -------------------------------
def _mfe_mae_caps_from_raw_5m(raw_5m: pd.DataFrame, trading_days: List[date], entry_clock: dtime) -> Dict[str, Any]:
    """Compute typical reachable upside (MFE) and drawdown (MAE) after entry until close.

    Uses ONLY historical sessions in `trading_days` (caller should pass prior sessions).
    """
    mfe_list: List[float] = []
    mae_list: List[float] = []

    if raw_5m is None or raw_5m.empty:
        return {"n": 0, "mfe_q": np.nan, "mae_q": np.nan, "mfe_med": np.nan, "mae_med": np.nan}

    # Ensure tz-aware and sorted
    df = raw_5m.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    df = df.sort_index()

    for d in trading_days:
        try:
            session_open_dt, session_close_dt = get_session_open_close(d)
            entry_dt = dt_combine(d, entry_clock)
            if entry_dt < session_open_dt:
                entry_dt = session_open_dt
            if entry_dt >= session_close_dt:
                continue

            day_df = df.loc[(df.index >= session_open_dt) & (df.index <= session_close_dt)]
            if day_df.empty:
                continue

            pre = day_df.loc[day_df.index <= entry_dt]
            if pre.empty:
                continue

            entry = float(pre["close"].iloc[-1])
            if not np.isfinite(entry) or entry <= 0:
                continue

            fut = day_df.loc[day_df.index > pre.index[-1]]
            if fut.empty:
                continue

            mfe = (float(fut["high"].max()) - entry) / entry
            mae = (entry - float(fut["low"].min())) / entry
            mfe_list.append(float(max(0.0, mfe)))
            mae_list.append(float(max(0.0, mae)))
        except Exception:
            continue

    n = len(mfe_list)
    if n == 0:
        return {"n": 0, "mfe_q": np.nan, "mae_q": np.nan, "mfe_med": np.nan, "mae_med": np.nan}

    mfe_arr = np.array(mfe_list, dtype=float)
    mae_arr = np.array(mae_list, dtype=float)

    return {
        "n": int(n),
        "mfe_q": float(np.quantile(mfe_arr, MFE_QUANTILE)),
        "mae_q": float(np.quantile(mae_arr, 0.80)),  # informative; not used directly in risk sizing
        "mfe_med": float(np.median(mfe_arr)),
        "mae_med": float(np.median(mae_arr)),
    }


def get_mfe_mae_caps(ticker: str, trading_days: List[date], entry_clock: dtime,
                    interval: str = PROB_INTERVAL, period: str = PROB_PERIOD,
                    ttl_seconds: int = 6 * 3600) -> Dict[str, Any]:
    """Cached wrapper for MFE/MAE caps."""
    # Limit to configured lookback
    days = list(trading_days)[-int(MFE_LOOKBACK_SESSIONS):] if trading_days else []
    key = f"mfe_mae_caps_v2::{ticker}::{entry_clock.strftime('%H%M')}::{interval}::{period}::{len(days)}::{MFE_QUANTILE}"
    cached = cache.get(key, ttl_seconds=ttl_seconds)
    if isinstance(cached, dict) and cached.get("n", 0) >= 0:
        return cached

    raw_5m = client._fetch_intraday_raw(ticker, interval=interval, period=period, ttl=INTRADAY_CACHE_TTL_5M)
    caps = _mfe_mae_caps_from_raw_5m(raw_5m, days, entry_clock)
    cache.set(key, caps)
    return caps

def features_for_day_from_5m(
    ticker: str,
    day_df_5m: pd.DataFrame,
    bench_day_5m: pd.DataFrame,
    d: date,
    entry_clock: dtime,
    or_end_clock: dtime,
    vol_mean: float,
    vol_std: float,
    atr14: float
) -> Optional[Dict[str, Any]]:
    if day_df_5m is None or day_df_5m.empty:
        return None

    session_open, session_close = get_session_open_close(d)
    entry_dt = dt_combine(d, entry_clock, ET)

    df = day_df_5m[(day_df_5m.index >= session_open) & (day_df_5m.index <= session_close)].copy()
    if df.empty:
        return None

    pre = df[df.index <= entry_dt]
    if pre.empty:
        return None

    entry = float(pre["close"].iloc[-1])
    open_px = float(df["open"].iloc[0])
    if not (np.isfinite(entry) and np.isfinite(open_px) and open_px > 0):
        return None

    orh = compute_orh(df, d, or_end_clock)
    vwap_val, _ = compute_session_vwap(pre)
    if not np.isfinite(vwap_val) or vwap_val <= 0:
        vwap_val = entry

    vwap_dist = safe_div(entry - vwap_val, vwap_val, 0.0)
    or_break = safe_div(entry - orh, orh, 0.0) if np.isfinite(orh) and orh > 0 else 0.0

    cum_vol = float(pre["volume"].sum())
    vola_pre = safe_div(float(pre["high"].max()) - float(pre["low"].min()), entry, 0.0)
    gap_from_open = safe_div(entry - open_px, open_px, 0.0)
    ret_pre = safe_div(entry - open_px, open_px, 0.0)

    volume_z = float((cum_vol - vol_mean) / vol_std) if np.isfinite(vol_mean) and np.isfinite(vol_std) and vol_std > 1e-9 else np.nan

    range_pre = float(pre["high"].max() - pre["low"].min()) if len(pre) else np.nan
    atr_intra = intraday_atr_from_5m(pre)
    vol_mult = vol_cluster_multiplier(entry_clock, atr_intra, atr14)
    atr_intra_rel = float(atr_intra / atr14) if (np.isfinite(atr_intra) and np.isfinite(atr14) and atr14 > 1e-9) else np.nan
    atr_rel = float(range_pre / atr14) if np.isfinite(atr14) and atr14 > 1e-9 else np.nan

    # --- stop features at entry (robustness / alignment) ---
    # Compute the same stop logic used in allocation/simulation, using ONLY data known up to entry.
    last15_low = np.nan
    try:
        bars15_pre = resample_ohlcv_intraday(pre, session_open, "15min")
        if bars15_pre is not None and not bars15_pre.empty:
            last15_low = float(bars15_pre["low"].iloc[-1])
    except Exception:
        last15_low = np.nan

    session_low_pre = float(pre["low"].min()) if "low" in pre.columns and not pre.empty else np.nan
    stop_px = compute_stop_auto(
        entry=entry,
        last15_low=last15_low,
        session_low=session_low_pre,
        buffer=float(cfg.stop_buffer_below_15m_low),
        atr14=atr14,
        range_pre=range_pre,
        atr_intra=atr_intra,
        vol_mult=vol_mult
    )

    stop_pct = np.nan
    stop_rel_atr = np.nan
    if stop_px is not None and np.isfinite(stop_px) and stop_px > 0 and stop_px < entry and entry > 0:
        rps = float(entry - stop_px)
        stop_pct = float(rps / entry)
        if np.isfinite(atr14) and atr14 > 0:
            stop_rel_atr = float(rps / atr14)

    close_series = pre["close"].dropna()
    mac = compute_macd_state(close_series, min_bars=20, require_hist_rising=False)
    macd_bull = int(mac["bullish"])
    try:
        _, _, hist = macd(close_series)
        hist_rising = int(bool(len(hist) > 2 and hist.iloc[-1] > hist.iloc[-2]))
    except Exception:
        hist_rising = 0

    rs_last = 0.0
    rs_slope = 0.0
    regime_above = 0
    if bench_day_5m is not None and not bench_day_5m.empty:
        bpre = bench_day_5m[(bench_day_5m.index >= session_open) & (bench_day_5m.index <= entry_dt)]
        if not bpre.empty:
            rs = compute_relative_strength_series(close_series, bpre["close"].dropna())
            rs_last, rs_slope = compute_rs_metrics(rs)
            bvwap, _ = compute_session_vwap(bpre)
            blast = float(bpre["close"].iloc[-1])
            if np.isfinite(bvwap) and np.isfinite(blast):
                regime_above = int(blast > bvwap)

    mins_from_open = minutes_between(session_open, entry_dt)
    mins_to_close = max(0.0, minutes_between(entry_dt, session_close))

    return {
        "ticker": ticker,
        "date": d,
        "sector": get_sector_yf(ticker) or "Unknown",

        "open_px": float(open_px),
        "entry_px": float(entry),
        "ret_pre": float(ret_pre),

        "or_break": float(or_break),
        "vwap_dist": float(vwap_dist),

        "cum_vol": float(cum_vol),
        "volume_z": float(volume_z) if np.isfinite(volume_z) else np.nan,

        "vola_pre": float(vola_pre),
        "atr14": float(atr14) if np.isfinite(atr14) else np.nan,
        "atr_intra": float(atr_intra) if np.isfinite(atr_intra) else np.nan,
        "atr_intra_rel": float(atr_intra_rel) if np.isfinite(atr_intra_rel) else np.nan,
        "vol_mult": float(vol_mult) if np.isfinite(vol_mult) else 1.0,
        "stop_pct": float(stop_pct) if np.isfinite(stop_pct) else np.nan,
        "stop_rel_atr": float(stop_rel_atr) if np.isfinite(stop_rel_atr) else np.nan,
        "atr_rel": float(atr_rel) if np.isfinite(atr_rel) else np.nan,

        "gap_from_open": float(gap_from_open),
        "macd_bull": int(macd_bull),
        "hist_rising": int(hist_rising),
        "rs_last": float(rs_last),
        "rs_slope": float(rs_slope),
        "regime_above": int(regime_above),
        "mins_from_open": float(mins_from_open),
        "mins_to_close": float(mins_to_close),

        "rr": np.nan,
        "sector_delta": np.nan,
        "rvol_like": np.nan,
    }

def build_training_dataset_rraware(
    tickers: List[str],
    train_days: List[date],
    entry_clock: dtime,
    stop_buffer: float,
    interval: str,
    period: str,
    tie_policy: str,
    rr_grid: List[float]
) -> Tuple[pd.DataFrame, Dict[Tuple[str, float], Dict[str, Any]], Dict[float, float]]:
    bench_raw_5m = client._fetch_intraday_raw(REGIME_TICKER, interval, period, ttl=INTRADAY_CACHE_TTL_5M)

    dd_start = min(train_days) - timedelta(days=60)
    dd_end = max(train_days)
    daily_train = client.get_daily_data(tickers, dd_start, dd_end)
    atr_tbl = compute_atr_table(daily_train, atr_len=ATR_LEN)
    atr_lookup = build_atr_lookup(atr_tbl)

    rows = []
    emp: Dict[Tuple[str, float], Dict[str, Any]] = {}

    # We'll also compute global MeanR_NONE by rr for backoff
    none_R_by_rr: Dict[float, List[float]] = {float(rr): [] for rr in rr_grid}

    for tkr in tickers:
        raw_5m = client._fetch_intraday_raw(tkr, interval, period, ttl=INTRADAY_CACHE_TTL_5M)
        if raw_5m.empty:
            continue

        cum_by_day = {}
        for d in train_days:
            session_open, session_close = get_session_open_close(d)
            entry_dt = dt_combine(d, entry_clock, ET)
            day_df = raw_5m[(raw_5m.index >= session_open) & (raw_5m.index <= session_close)]
            pre = day_df[day_df.index <= entry_dt]
            if not pre.empty:
                cum_by_day[d] = float(pre["volume"].sum())

        if len(cum_by_day) < max(10, int(0.4 * len(train_days))):
            continue

        vol_mean = float(np.mean(list(cum_by_day.values())))
        vol_std  = float(np.std(list(cum_by_day.values()), ddof=0))

        counters = {float(rr): {"TP": 0, "SL": 0, "NONE": 0, "TIE": 0, "R_all": [], "R_none": []} for rr in rr_grid}

        for d in train_days:
            rr_sig = hash(tuple([float(x) for x in rr_grid]))
            cache_key = f"train_rr_nse_v5_{tkr}_{d}_{entry_clock.strftime('%H%M')}_{stop_buffer}_{interval}_{tie_policy}_{rr_sig}"
            cached = cache.get(cache_key, ttl_seconds=PROB_CACHE_TTL)

            if cached is not None:
                pack = cached
            else:
                day_df = raw_5m[raw_5m.index.date == d].copy()
                bday = bench_raw_5m[bench_raw_5m.index.date == d].copy() if not bench_raw_5m.empty else pd.DataFrame()

                atr14 = atr_asof(atr_lookup, tkr, d - timedelta(days=1))
                feat = features_for_day_from_5m(
                    ticker=tkr,
                    day_df_5m=day_df,
                    bench_day_5m=bday,
                    d=d,
                    entry_clock=entry_clock,
                    or_end_clock=cfg.opening_range_end,
                    vol_mean=vol_mean,
                    vol_std=vol_std,
                    atr14=atr14
                )
                if feat is None:
                    cache.set(cache_key, None)
                    continue

                mean_cum = max(vol_mean, 1.0)
                rvol_like = float(cum_by_day.get(d, 0.0) / mean_cum)

                sim_pack = simulate_tp_sl_for_rrgrid(
                    day_df=day_df,
                    d=d,
                    entry_clock=entry_clock,
                    stop_buffer=stop_buffer,
                    rr_grid=rr_grid,
                    tie_policy=tie_policy,
                    atr14=atr14
                )

                # stop/target geometry features (barriers influence probabilities)
                entry_px_sim = float(sim_pack.get("entry", np.nan))
                rps_sim = float(sim_pack.get("rps", np.nan))
                if np.isfinite(entry_px_sim) and entry_px_sim > 0 and np.isfinite(rps_sim) and rps_sim > 0:
                    feat["stop_pct"] = float(rps_sim / entry_px_sim)
                    if np.isfinite(atr14) and atr14 > 0:
                        feat["stop_rel_atr"] = float(rps_sim / atr14)
                    else:
                        feat["stop_rel_atr"] = np.nan
                else:
                    feat["stop_pct"] = np.nan
                    feat["stop_rel_atr"] = np.nan

                if sim_pack is None:
                    cache.set(cache_key, None)
                    continue

                feat["rvol_like"] = float(rvol_like)

                pack = {"feat": feat, "sim": sim_pack}
                cache.set(cache_key, pack)

            if not pack:
                continue

            feat = pack["feat"]
            sim_pack = pack["sim"]

            for rr in rr_grid:
                rr = float(rr)
                rr_res = sim_pack["results"].get(rr, None)
                if rr_res is None:
                    continue

                lab = str(rr_res["label"])
                R = float(rr_res["R"])

                row = dict(feat)
                row["rr"] = float(rr)
                row["label"] = lab
                row["R"] = float(R)
                rows.append(row)

                if lab in counters[rr]:
                    counters[rr][lab] += 1
                else:
                    counters[rr]["NONE"] += 1
                if np.isfinite(R):
                    counters[rr]["R_all"].append(R)
                    if lab == "NONE":
                        counters[rr]["R_none"].append(R)

        for rr in rr_grid:
            rr = float(rr)
            c = counters[rr]
            N = c["TP"] + c["SL"] + c["NONE"] + c["TIE"]
            if N <= 0:
                continue
            mean_all = float(np.mean(c["R_all"])) if c["R_all"] else np.nan
            mean_none = float(np.mean(c["R_none"])) if c["R_none"] else np.nan
            emp[(tkr, rr)] = {
                "N": int(N),
                "TP": int(c["TP"]),
                "SL": int(c["SL"]),
                "NONE": int(c["NONE"]),
                "TIE": int(c["TIE"]),
                "MeanR": mean_all,
                "MeanR_NONE": mean_none,
            }
            if np.isfinite(mean_none):
                none_R_by_rr[rr].append(mean_none)

    df = pd.DataFrame(rows)

    # Sector delta
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        rr0 = float(rr_grid[0]) if rr_grid else 1.0
        base = df[df["rr"] == rr0][["date","sector","ticker","ret_pre"]].drop_duplicates()
        sec_mean = base.groupby(["date","sector"])["ret_pre"].mean().reset_index().rename(columns={"ret_pre":"sector_ret_mean"})
        df = df.merge(sec_mean, on=["date","sector"], how="left")
        df["sector_delta"] = (df["ret_pre"] - df["sector_ret_mean"]).astype(float)

    # Global mean R when NONE, by rr (fallback for tickers with no empirical NONE mean)
    global_mean_none: Dict[float, float] = {}
    for rr, arr in none_R_by_rr.items():
        if arr:
            global_mean_none[rr] = float(np.nanmean(arr))
        else:
            global_mean_none[rr] = 0.0

    return df, emp, global_mean_none

def train_global_model_with_cal_split(df_train: pd.DataFrame):
    if (not SKLEARN_OK) or df_train is None or df_train.empty:
        return None, None, None, None

    y = df_train["label"].astype(str).copy()
    if PROB_TIE_POLICY.upper() == "STOP":
        y = y.replace({"TIE": "SL"})
    else:
        y = y.replace({"TIE": "NONE"})

    df_train = df_train.copy()
    df_train["y"] = y
    df_train["date"] = pd.to_datetime(df_train["date"])
    df_train = df_train.sort_values(["date", "ticker"])

    uniq_days = sorted(df_train["date"].dt.date.unique())
    if len(uniq_days) < 18:
        logger.warning("Not enough unique training days for robust ML calibration; skipping ML.")
        return None, None, None, None

    n_cal = max(8, int(0.2 * len(uniq_days)))
    cal_days = set(uniq_days[-n_cal:])
    train_mask = ~df_train["date"].dt.date.isin(cal_days)
    cal_mask = df_train["date"].dt.date.isin(cal_days)

    feature_cols_num = [
        "entry_px","open_px","ret_pre",
        "or_break","vwap_dist",
        "cum_vol","rvol_like",
        "volume_z",
        "vola_pre",
        "atr14","atr_intra","atr_intra_rel","vol_mult","atr_rel","stop_pct","stop_rel_atr",
        "gap_from_open",
        "macd_bull","hist_rising",
        "rs_last","rs_slope","regime_above",
        "mins_from_open","mins_to_close",
        "sector_delta",
        "rr"
    ]
    feature_cols_cat = ["sector"]

    X_train = df_train.loc[train_mask, feature_cols_num + feature_cols_cat]
    y_train = df_train.loc[train_mask, "y"]
    X_cal = df_train.loc[cal_mask, feature_cols_num + feature_cols_cat]
    y_cal = df_train.loc[cal_mask, "y"]

    if y_train.nunique() < 2:
        logger.warning("Training labels have <2 classes; skipping ML.")
        return None, None, None, None

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]), feature_cols_num),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), feature_cols_cat),
        ],
        remainder="drop"
    )

    base = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.06,
        max_iter=260,
        l2_regularization=0.02,
        min_samples_leaf=25,
        random_state=42
    )

    model = Pipeline(steps=[("pre", pre), ("clf", base)])

    try:
        model.fit(X_train, y_train)
        cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        cal.fit(X_cal, y_cal)

        proba = cal.predict_proba(X_cal)
        ll = log_loss(y_cal, proba, labels=cal.classes_)
        logger.info(f"ML prob model trained: samples={len(df_train)}, cal_days={len(cal_days)}, cal_logloss={ll:.4f}")
        return cal, X_cal, y_cal, cal_days
    except Exception as e:
        logger.warning(f"ML training failed: {e}")
        return None, None, None, None

def calibration_bucket_report_multiclass(model, X_cal: pd.DataFrame, y_cal: pd.Series, bins: int = 10):
    if model is None or X_cal is None or y_cal is None or len(X_cal) == 0:
        return
    try:
        proba = model.predict_proba(X_cal)
        classes = list(model.classes_)
        p = {classes[i]: proba[:, i] for i in range(len(classes))}
        p_tp = p.get("TP", None)
        p_sl = p.get("SL", None)
        if p_tp is None or p_sl is None:
            logger.info("Calibration report skipped: model missing TP or SL class.")
            return

        def _bucket_table(target_label: str, p_vec: np.ndarray):
            y_true = (y_cal.values.astype(str) == target_label).astype(int)
            edges = np.linspace(0.0, 1.0, bins + 1)
            rows = []
            for i in range(bins):
                lo, hi = edges[i], edges[i+1]
                mask = (p_vec >= lo) & (p_vec < hi) if i < bins - 1 else (p_vec >= lo) & (p_vec <= hi)
                n = int(mask.sum())
                if n == 0:
                    continue
                pred_mean = float(np.mean(p_vec[mask]))
                obs_rate = float(np.mean(y_true[mask]))
                rows.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": n, "pred": pred_mean, "obs": obs_rate, "diff": obs_rate - pred_mean})
            return pd.DataFrame(rows)

        tp_tbl = _bucket_table("TP", p_tp)
        sl_tbl = _bucket_table("SL", p_sl)

        print("\n" + "=" * 110)
        print("CALIBRATION REPORT (holdout days): TP buckets")
        print("=" * 110)
        print(tp_tbl.to_string(index=False) if not tp_tbl.empty else "(no bins)")
        print("\n" + "=" * 110)
        print("CALIBRATION REPORT (holdout days): SL buckets")
        print("=" * 110)
        print(sl_tbl.to_string(index=False) if not sl_tbl.empty else "(no bins)")
    except Exception as e:
        logger.warning(f"Calibration report failed: {e}")

def empirical_probs_from_counts(tp: int, sl: int, none: int, tie: int, policy: str) -> Tuple[float,float,float,float,int]:
    if policy.upper() == "STOP":
        sl_eff = sl + tie
        tp_eff = tp
        none_eff = none
        hitN = tp_eff + sl_eff
    else:
        sl_eff = sl
        tp_eff = tp
        none_eff = none + tie
        hitN = tp + sl + tie

    N = tp + sl + none + tie
    if N <= 0:
        return (np.nan, np.nan, np.nan, np.nan, 0)

    P_TP = float((tp_eff + 1) / (N + 3))
    P_SL = float((sl_eff + 1) / (N + 3))
    P_NONE = float((none_eff + 1) / (N + 3))

    if hitN > 0:
        tp_cond = tp + (0.5 * tie if policy.upper() == "HALF" else 0.0)
        P_TP_hit = float((tp_cond + 1) / (hitN + 2))
    else:
        P_TP_hit = np.nan

    return P_TP, P_SL, P_NONE, P_TP_hit, N

def blend(p_model: float, p_emp: float, n_emp: int, k: int) -> float:
    if not np.isfinite(p_model):
        return p_emp
    if not np.isfinite(p_emp) or n_emp <= 0:
        return p_model
    w = float(n_emp / (n_emp + k))
    return float(w * p_emp + (1.0 - w) * p_model)

# Precompute sector means for inference (sector_delta)
sector_mean_ret_today: Dict[str, float] = {}
if not df_alloc.empty:
    tmp = df_alloc.copy()
    tmp["ret_from_open"] = safe_div(tmp["Entry"] - tmp["Open"], tmp["Open"], np.nan)
    for sec, g in tmp.groupby("Sector", dropna=False):
        sec = str(sec) if sec else "Unknown"
        sector_mean_ret_today[sec] = float(np.nanmean(g["ret_from_open"].values)) if len(g) else 0.0

def compute_today_features_for_inference_rr(ticker: str, rr: float, stop_px: Optional[float] = None) -> Optional[Dict[str, Any]]:
    df_1m = client.get_intraday_data(ticker, analysis_date, interval="1m", period="8d")
    if df_1m.empty:
        return None
    session_open_et, session_close_et = get_session_open_close(analysis_date)
    cut_1m = df_1m[(df_1m.index >= session_open_et) & (df_1m.index < as_of_et)]
    if cut_1m.empty:
        return None

    entry = float(cut_1m["close"].iloc[-1])
    open_px = float(cut_1m["open"].iloc[0])
    if not (np.isfinite(entry) and np.isfinite(open_px) and open_px > 0):
        return None

    orh = compute_opening_range_high(cut_1m)
    vwap_val, _ = compute_session_vwap(cut_1m)
    if not np.isfinite(vwap_val) or vwap_val <= 0:
        vwap_val = entry

    vwap_dist = safe_div(entry - vwap_val, vwap_val, 0.0)
    or_break = safe_div(entry - orh, orh, 0.0) if np.isfinite(orh) and orh > 0 else 0.0

    cum_vol = float(cut_1m["volume"].sum())
    vola_pre = safe_div(float(cut_1m["high"].max()) - float(cut_1m["low"].min()), entry, 0.0)
    gap_from_open = safe_div(entry - open_px, open_px, 0.0)
    ret_pre = safe_div(entry - open_px, open_px, 0.0)

    vol_z = np.nan
    if ticker in stage2_map and np.isfinite(stage2_map[ticker].get("vol_z", np.nan)):
        vol_z = float(stage2_map[ticker]["vol_z"])

    atr14 = float(atr_for_stop(ticker))

    stop_pct = float('nan')
    stop_rel_atr = float('nan')
    if stop_px is not None and np.isfinite(stop_px) and stop_px > 0 and stop_px < entry and entry > 0:
        rps = entry - float(stop_px)
        stop_pct = float(rps / entry)
        if np.isfinite(atr14) and atr14 > 0:
            stop_rel_atr = float(rps / atr14)
    range_pre = float(cut_1m["high"].max() - cut_1m["low"].min())
    atr_rel = float(range_pre / atr14) if np.isfinite(atr14) and atr14 > 1e-9 else np.nan

    bars5 = resample_ohlcv_intraday(cut_1m, session_open_et, "5min")
    atr_intra = intraday_atr_from_5m(bars5)
    vol_mult = vol_cluster_multiplier(as_of_et.time(), atr_intra, atr14)
    atr_intra_rel = float(atr_intra / atr14) if (np.isfinite(atr_intra) and np.isfinite(atr14) and atr14 > 1e-9) else np.nan

    close_series = bars5["close"].dropna() if not bars5.empty else cut_1m["close"].dropna()
    mac = compute_macd_state(close_series, min_bars=20, require_hist_rising=False)
    macd_bull = int(mac["bullish"])
    try:
        _, _, hist = macd(close_series)
        hist_rising = int(bool(len(hist) > 2 and hist.iloc[-1] > hist.iloc[-2]))
    except Exception:
        hist_rising = 0

    rs_last = 0.0
    rs_slope = 0.0
    if not bench_today_5m.empty and not close_series.empty:
        rs = compute_relative_strength_series(close_series, bench_today_5m["close"].dropna())
        rs_last, rs_slope = compute_rs_metrics(rs)

    regime_above = 0
    if not bench_today_1m.empty:
        bcut = bench_today_1m[(bench_today_1m.index >= session_open_et) & (bench_today_1m.index < as_of_et)]
        if not bcut.empty:
            bvwap, _ = compute_session_vwap(bcut)
            blast = float(bcut["close"].iloc[-1])
            if np.isfinite(bvwap) and np.isfinite(blast):
                regime_above = int(blast > bvwap)

    rvol_like = 1.0
    if ticker in stage2_map:
        rv = stage2_map[ticker].get("rvol", np.nan)
        if np.isfinite(rv):
            rvol_like = float(rv)

    mins_from_open = minutes_between(session_open_et, as_of_et)
    mins_to_close = max(0.0, minutes_between(as_of_et, session_close_et))

    sec = get_sector_yf(ticker) or "Unknown"
    sec_mean = sector_mean_ret_today.get(sec, 0.0)
    sector_delta = float(ret_pre - sec_mean) if np.isfinite(ret_pre) and np.isfinite(sec_mean) else np.nan

    return {
        "entry_px": float(entry),
        "open_px": float(open_px),
        "ret_pre": float(ret_pre),

        "or_break": float(or_break),
        "vwap_dist": float(vwap_dist),

        "cum_vol": float(cum_vol),
        "rvol_like": float(rvol_like),

        "volume_z": float(vol_z) if np.isfinite(vol_z) else np.nan,

        "vola_pre": float(vola_pre),
        "atr14": float(atr14) if np.isfinite(atr14) else np.nan,
        "atr_intra": float(atr_intra) if np.isfinite(atr_intra) else np.nan,
        "atr_intra_rel": float(atr_intra_rel) if np.isfinite(atr_intra_rel) else np.nan,
        "vol_mult": float(vol_mult) if np.isfinite(vol_mult) else 1.0,
        "atr_rel": float(atr_rel) if np.isfinite(atr_rel) else np.nan,

        "gap_from_open": float(gap_from_open),
        "macd_bull": int(macd_bull),
        "hist_rising": int(hist_rising),
        "rs_last": float(rs_last),
        "rs_slope": float(rs_slope),
        "regime_above": int(regime_above),
        "mins_from_open": float(mins_from_open),
        "mins_to_close": float(mins_to_close),

        "sector": sec,
        "sector_delta": float(sector_delta) if np.isfinite(sector_delta) else np.nan,

        "rr": float(rr),
    }

if ENABLE_PROB_MODEL and not df_alloc.empty:
    entry_clock = as_of_et.time() if PROB_USE_ASOF_TIME else cfg.eval_time_default
    train_days = get_trading_days(analysis_date, PROB_TRAIN_LOOKBACK_SESSIONS)

    base_train = universe_scan[:min(PROB_TRAIN_TICKERS_MAX, len(universe_scan))]
    need = [t for t in df_alloc["Ticker"].tolist() if t not in base_train]
    train_tickers = base_train + need

    logger.info(f"Prob model: training tickers={len(train_tickers)}, days={len(train_days)}, entry_time={entry_clock}, interval={PROB_INTERVAL}, RR_GRID={RR_GRID}")
    prefetch_intraday_batch(train_tickers + [REGIME_TICKER], PROB_INTERVAL, PROB_PERIOD, batch_size=PROB_PREFETCH_BATCH)

    df_train, emp_stats, global_mean_none = build_training_dataset_rraware(
        tickers=train_tickers,
        train_days=train_days,
        entry_clock=entry_clock,
        stop_buffer=float(cfg.stop_buffer_below_15m_low),
        interval=PROB_INTERVAL,
        period=PROB_PERIOD,
        tie_policy=PROB_TIE_POLICY,
        rr_grid=RR_GRID
    )

    if df_train.empty:
        logger.warning("Prob model: training dataset empty. Skipping probabilities.")
    else:
        model, X_cal, y_cal, cal_days = train_global_model_with_cal_split(df_train)
        if model is not None and X_cal is not None and y_cal is not None:
            calibration_bucket_report_multiclass(model, X_cal, y_cal, bins=10)

        # Map for reachability checks (uses allocation stop/ATR)
        alloc_lookup = {str(r["Ticker"]): r for _, r in df_alloc.iterrows()}


        # Precompute historical MFE/MAE caps for the final candidates (used to cap RR choices).
        mfe_caps_map: Dict[str, Dict[str, Any]] = {}
        if ENABLE_DYNAMIC_RR_CAP_FROM_MFE:
            try:
                logger.info(
                    f"Computing MFE caps for {len(df_alloc)} tickers | lookback={MFE_LOOKBACK_SESSIONS} sessions | q={MFE_QUANTILE}"
                )
                for _t in df_alloc["Ticker"].tolist():
                    mfe_caps_map[_t] = get_mfe_mae_caps(_t, train_days, entry_clock=entry_clock)
            except Exception as e:
                logger.warning(f"MFE cap computation failed (continuing without caps): {e}")

        # Time scaling for reachability: later in the session we demand closer targets.
        session_minutes = max(1, int((session_close_et - session_open_et).total_seconds() / 60))
        mins_to_close_global = max(0, int((session_close_et - dt_combine(analysis_date, entry_clock)).total_seconds() / 60))
        remaining_scale = float(math.sqrt(max(min(mins_to_close_global / session_minutes, 1.0), 0.05)))
        out_rows = []
        for tkr in df_alloc["Ticker"].tolist():
            best = None
            ladder = []
            ar = alloc_lookup.get(str(tkr), {})
            entry_px_alloc = float(ar.get("Entry", np.nan))
            stop_px_alloc = float(ar.get("Stop", np.nan))
            atr14_alloc = float(ar.get("ATR14", np.nan))
            rps_alloc = float(entry_px_alloc - stop_px_alloc) if np.isfinite(entry_px_alloc) and np.isfinite(stop_px_alloc) else np.nan


# ============================================================
# BEST RR + PROBABILITY LADDER + MERGE (REPLACE OLD BLOCK FULLY)
# ============================================================

# If you don't already have this defined somewhere, keep it here:
EMP_MIN_SAMPLES_FOR_RR = 30   # use empirical probs only when >= this samples
PROB_DEBUG = False            # set True to print why some tickers fail

import numpy as np
import pandas as pd

out_rows = []
default_probs = np.array([0.33, 0.33, 0.34], dtype=float)

# iterate rows of df_alloc (must contain: Ticker, Entry, Stop, optionally ATR14)
for _, row in df_alloc.iterrows():
    tkr = str(row.get("Ticker", ""))

    entry_px_alloc = float(row.get("Entry", np.nan))
    stop_px_alloc  = float(row.get("Stop", np.nan))

    # risk per share
    rps_alloc = entry_px_alloc - stop_px_alloc
    if not (np.isfinite(entry_px_alloc) and np.isfinite(stop_px_alloc) and np.isfinite(rps_alloc) and rps_alloc > 0):
        if PROB_DEBUG:
            print("Skip invalid entry/stop:", tkr, entry_px_alloc, stop_px_alloc)
        continue

    atr14_alloc = float(row.get("ATR14", np.nan))

    # Candidate RR grid (optionally capped by dynamic reachability)
    rr_candidates = [float(x) for x in RR_GRID]
    max_rr_cap = float("inf")

    if ENABLE_TARGET_REACHABILITY_FILTER:
        dist_cap = float("inf")

        # Time-scaled cap: late in the day we demand closer targets.
        if np.isfinite(atr14_alloc) and atr14_alloc > 0:
            dist_cap = TARGET_MAX_ATR_MULT * atr14_alloc * remaining_scale
        else:
            dist_cap = TARGET_MAX_PCT * entry_px_alloc * remaining_scale

        # Historical cap: typical reachable upside for this ticker at this entry time.
        if ENABLE_DYNAMIC_RR_CAP_FROM_MFE:
            caps = mfe_caps_map.get(tkr, None)
            if isinstance(caps, dict) and caps.get("n", 0) >= MFE_MIN_SAMPLES and np.isfinite(caps.get("mfe_q", np.nan)):
                dist_cap = min(dist_cap, float(caps["mfe_q"]) * entry_px_alloc * MFE_RR_SLACK)

        if np.isfinite(dist_cap) and dist_cap > 0:
            max_rr_cap = float(dist_cap / rps_alloc)

        rr_candidates = [rr for rr in rr_candidates if rr <= max_rr_cap + 1e-9]
        if not rr_candidates:
            rr_candidates = [float(min(RR_GRID))]

    best = None
    ladder = []

    for rr in rr_candidates:
        rr = float(rr)

        feat = compute_today_features_for_inference_rr(tkr, rr, stop_px=stop_px_alloc)
        if feat is None:
            if PROB_DEBUG:
                print("feat None:", tkr, "rr", rr)
            continue

        # ---------- model probs ----------
        p_model_tp = p_model_sl = p_model_none = np.nan
        if model is not None:
            X = pd.DataFrame([feat])

            # Align columns to training schema (prevents missing/extra column errors)
            if hasattr(model, "feature_names_in_"):
                X = X.reindex(columns=list(model.feature_names_in_), fill_value=0.0)

            # Clean numeric issues that commonly break predict_proba
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            try:
                proba = model.predict_proba(X)[0]
                cls = list(model.classes_)
                p = {cls[i]: float(proba[i]) for i in range(len(cls))}
                p_model_tp = float(p.get("TP", np.nan))
                p_model_sl = float(p.get("SL", np.nan))
                p_model_none = float(p.get("NONE", np.nan))
            except Exception as e:
                if PROB_DEBUG:
                    print("predict_proba failed:", tkr, "rr", rr, "err", repr(e))

        # ---------- empirical probs ----------
        emp = emp_stats.get((tkr, rr), None)
        ER_none = np.nan
        N_e = 0

        if emp is not None:
            P_TP_e, P_SL_e, P_NONE_e, _, N_e = empirical_probs_from_counts(
                emp.get("TP", 0), emp.get("SL", 0), emp.get("NONE", 0), emp.get("TIE", 0), PROB_TIE_POLICY
            )
            ER_none = float(emp.get("MeanR_NONE", np.nan))
        else:
            P_TP_e = P_SL_e = P_NONE_e = np.nan

        # fallback for expected R of NONE
        if not np.isfinite(ER_none):
            ER_none = float(global_mean_none.get(rr, 0.0))

        # ---------- choose probs: empirical (if enough) else model ----------
        if emp is not None and int(N_e) >= EMP_MIN_SAMPLES_FOR_RR:
            P_TP, P_SL, P_NONE = float(P_TP_e), float(P_SL_e), float(P_NONE_e)
        else:
            P_TP, P_SL, P_NONE = float(p_model_tp), float(p_model_sl), float(p_model_none)

        # if NONE missing, backfill
        if not np.isfinite(P_NONE):
            tp = P_TP if np.isfinite(P_TP) else 0.0
            sl = P_SL if np.isfinite(P_SL) else 0.0
            P_NONE = max(0.0, 1.0 - tp - sl)

        # ---------- sanitize probs (prevents NaN EV / NaN best) ----------
        probs = np.array([P_TP, P_SL, P_NONE], dtype=float)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, 1.0)
        s = probs.sum()
        if s <= 0:
            probs = default_probs.copy()
        else:
            probs /= s

        P_TP, P_SL, P_NONE = probs
        P_HIT = P_TP + P_SL
        P_TP_GivenHit = (P_TP / P_HIT) if P_HIT > 0 else np.nan

        # ---------- Robust EV: include expected R if no barrier hit ----------
        EV_R = float(rr * P_TP - 1.0 * P_SL + ER_none * P_NONE)

        # uncertainty = entropy
        eps = 1e-12
        ent = -float(P_TP*np.log(P_TP+eps) + P_SL*np.log(P_SL+eps) + P_NONE*np.log(P_NONE+eps))

        ladder.append((rr, float(P_TP), float(P_SL), float(P_NONE), float(P_HIT), float(ER_none), float(EV_R), int(N_e)))

        if (best is None) or (EV_R > best["EV_R"]):
            best = {
                "Ticker": tkr,
                "BestRR": rr,
                "MaxRR_Cap": float(max_rr_cap) if np.isfinite(max_rr_cap) else np.nan,
                "MFE_q": float(mfe_caps_map.get(tkr, {}).get("mfe_q", np.nan)) if ENABLE_DYNAMIC_RR_CAP_FROM_MFE else np.nan,
                "MAE_q": float(mfe_caps_map.get(tkr, {}).get("mae_q", np.nan)) if ENABLE_DYNAMIC_RR_CAP_FROM_MFE else np.nan,
                "MFE_N": int(mfe_caps_map.get(tkr, {}).get("n", 0)) if ENABLE_DYNAMIC_RR_CAP_FROM_MFE else 0,
                "P_TP": float(P_TP),
                "P_SL": float(P_SL),
                "P_NONE": float(P_NONE),
                "P_HIT": float(P_HIT),
                "E_R_NONE": float(ER_none),
                "P_TP_GivenHit": float(P_TP_GivenHit) if np.isfinite(P_TP_GivenHit) else np.nan,
                "EV_R": float(EV_R),
                "Uncertainty": float(ent),
                "EmpN": int(N_e),
            }

    # If everything failed for this ticker, still write a row (avoids NaN columns)
    if best is None:
        rr0 = float(min(RR_GRID))
        best = {
            "Ticker": tkr,
            "BestRR": rr0,
            "MaxRR_Cap": float(max_rr_cap) if np.isfinite(max_rr_cap) else np.nan,
            "MFE_q": np.nan,
            "MAE_q": np.nan,
            "MFE_N": 0,
            "P_TP": float(default_probs[0]),
            "P_SL": float(default_probs[1]),
            "P_NONE": float(default_probs[2]),
            "P_HIT": float(default_probs[0] + default_probs[1]),
            "E_R_NONE": float(global_mean_none.get(rr0, 0.0)),
            "P_TP_GivenHit": np.nan,
            "EV_R": float(rr0 * default_probs[0] - 1.0 * default_probs[1] + global_mean_none.get(rr0, 0.0) * default_probs[2]),
            "Uncertainty": float(-np.sum(default_probs * np.log(default_probs + 1e-12))),
            "EmpN": 0,
        }
        ladder = [(rr0, best["P_TP"], best["P_SL"], best["P_NONE"], best["P_HIT"], best["E_R_NONE"], best["EV_R"], 0)]

    best["RR_Ladder"] = str([
        (r, round(pt, 3), round(ps, 3), round(pn, 3), round(ph, 3), round(ern, 3), round(ev, 3), n)
        for (r, pt, ps, pn, ph, ern, ev, n) in ladder
    ])

    out_rows.append(best)

# ---------------------------
# Merge ONCE (prevents MergeError + duplicate columns)
# ---------------------------
if out_rows:
    df_prob = pd.DataFrame(out_rows)

    # remove any older versions of these columns to keep merge clean
    prob_cols = [c for c in df_prob.columns if c != "Ticker"]
    df_alloc = df_alloc.drop(columns=prob_cols, errors="ignore")

    df_alloc = df_alloc.merge(df_prob, on="Ticker", how="left")

# ---------------------------
# Update Target based on BestRR
# ---------------------------
if "BestRR" in df_alloc.columns:
    rr_used = df_alloc["BestRR"].fillna(cfg.rr_min).astype(float)
    df_alloc["Target"] = df_alloc["Entry"] + rr_used * (df_alloc["Entry"] - df_alloc["Stop"])




# ============================================================
# TAKE / PASS Column (Robustness: include P_HIT and P_NONE caps)
# ============================================================
if ENABLE_TAKE_COLUMN and not df_alloc.empty and all(c in df_alloc.columns for c in ["EV_R","P_SL","P_TP_GivenHit","Uncertainty","EmpN","VWAP_Zone","P_NONE","P_HIT"]):
    df_alloc = df_alloc.copy()

    zone = df_alloc["VWAP_Zone"].astype(str).fillna("UNKNOWN")
    ev_mult = np.where(zone == "ACCUMULATION", 0.90,
              np.where(zone == "OVEREXTENDED", 1.35,
              np.where(zone == "BELOW_VWAP", 9.99, 1.00)))
    psl_mult = np.where(zone == "ACCUMULATION", 1.05,
               np.where(zone == "OVEREXTENDED", 0.85,
               np.where(zone == "BELOW_VWAP", 0.00, 1.00)))
    ptp_mult = np.where(zone == "ACCUMULATION", 0.95,
               np.where(zone == "OVEREXTENDED", 1.15,
               np.where(zone == "BELOW_VWAP", 9.99, 1.00)))
    unc_mult = np.where(zone == "ACCUMULATION", 1.05,
               np.where(zone == "OVEREXTENDED", 0.85,
               np.where(zone == "BELOW_VWAP", 0.00, 1.00)))
    emp_mult = np.where(zone == "OVEREXTENDED", 1.15, 1.00)

    min_ev = TAKE_EFF["min_ev"] * ev_mult
    max_psl = TAKE_EFF["max_psl"] * psl_mult
    min_ptp = TAKE_EFF["min_ptp_hit"] * ptp_mult
    max_unc = TAKE_EFF["max_unc"] * unc_mult
    min_emp = int(TAKE_EFF["min_emp"])

    df_alloc["TAKE"] = (
        (df_alloc["EV_R"] >= min_ev) &
        (df_alloc["P_SL"] <= max_psl) &
        (df_alloc["P_TP_GivenHit"] >= min_ptp) &
        (df_alloc["Uncertainty"] <= max_unc) &
        (df_alloc["EmpN"] >= (min_emp * emp_mult)) &
        (df_alloc["P_HIT"] >= TAKE_MIN_P_HIT) &
        (df_alloc["P_NONE"] <= TAKE_MAX_P_NONE)
    )

    df_alloc.loc[zone == "BELOW_VWAP", "TAKE"] = False
    df_alloc["TAKE"] = df_alloc["TAKE"].map({True: "TAKE", False: "PASS"})

# ============================================================
# REALIZED TP/SL CHECK
# Robustness: if neither hits by eval_end, label as EOD (close-out)
# ============================================================
ENABLE_REALIZED_HIT_CHECK = True
REALIZED_INTERVAL_PRIMARY = "1m"
REALIZED_PERIOD_PRIMARY = "8d"
REALIZED_INTERVAL_FALLBACK = "5m"
REALIZED_PERIOD_FALLBACK = "60d"

def _real_now_et_floor() -> dt:
    return dt.now(ET).replace(second=0, microsecond=0)

def _realized_eval_end_et() -> dt:
    real_now = _real_now_et_floor()
    if SIMULATION_MODE:
        return session_close_et
    if analysis_date < real_now.date():
        return session_close_et
    if real_now >= session_close_et:
        return session_close_et
    return min(real_now, session_close_et)

def _get_intraday_for_realized(ticker: str) -> pd.DataFrame:
    df = client.get_intraday_data(ticker, analysis_date, interval=REALIZED_INTERVAL_PRIMARY, period=REALIZED_PERIOD_PRIMARY)
    if df is None or df.empty:
        df = client.get_intraday_data(ticker, analysis_date, interval=REALIZED_INTERVAL_FALLBACK, period=REALIZED_PERIOD_FALLBACK)
    if df is None:
        return pd.DataFrame()
    df = df[(df.index >= session_open_et) & (df.index <= session_close_et)].copy()
    return df

def realized_tp_sl_outcome(
    ticker: str,
    entry_px: float,
    stop_px: float,
    target_px: float,
    as_of_et: dt,
    eval_end_et: dt,
    tie_policy: str = "STOP",
) -> Dict[str, Any]:
    out = {
        "Ticker": ticker,
        "Hit": "EOD",
        "HitPx": np.nan,
        "HitTime": pd.NaT,
        "RealizedR": np.nan,
        "CheckedThrough": eval_end_et,
    }

    if not (np.isfinite(entry_px) and np.isfinite(stop_px) and np.isfinite(target_px)):
        return out
    if stop_px >= entry_px or target_px <= entry_px:
        return out

    df = _get_intraday_for_realized(ticker)
    if df.empty:
        return out

    pre = df[df.index < as_of_et]
    if pre.empty:
        return out

    entry_ts = pre.index[-1]
    rps = entry_px - stop_px
    if not (np.isfinite(rps) and rps > 0):
        return out

    fut = df[(df.index > entry_ts) & (df.index <= eval_end_et)]
    if fut.empty:
        last_px = float(pre["close"].iloc[-1])
        out["Hit"] = "EOD"
        out["HitPx"] = float(last_px) if np.isfinite(last_px) else np.nan
        out["HitTime"] = pre.index[-1]
        out["RealizedR"] = float((last_px - entry_px) / rps) if np.isfinite(last_px) else np.nan
        return out

    hit = "NONE"
    hit_px = np.nan
    hit_time = pd.NaT

    for ts, row in fut.iterrows():
        o = float(row["open"]) if np.isfinite(row.get("open", np.nan)) else np.nan
        h = float(row["high"]) if np.isfinite(row.get("high", np.nan)) else np.nan
        l = float(row["low"])  if np.isfinite(row.get("low", np.nan))  else np.nan

        if np.isfinite(o):
            if o <= stop_px:
                hit, hit_px, hit_time = "SL", float(stop_px), ts
                break
            if o >= target_px:
                hit, hit_px, hit_time = "TP", float(target_px), ts
                break

        sl_in = np.isfinite(l) and (l <= stop_px)
        tp_in = np.isfinite(h) and (h >= target_px)

        if sl_in and tp_in:
            if tie_policy.upper() == "STOP":
                hit, hit_px, hit_time = "SL", float(stop_px), ts
            else:
                hit, hit_px, hit_time = "TIE", float((stop_px + target_px) / 2.0), ts
            break
        elif sl_in:
            hit, hit_px, hit_time = "SL", float(stop_px), ts
            break
        elif tp_in:
            hit, hit_px, hit_time = "TP", float(target_px), ts
            break

    if hit == "NONE":
        last_px = float(fut["close"].iloc[-1])
        hit, hit_px, hit_time = "EOD", float(last_px), fut.index[-1]

    out["Hit"] = hit
    out["HitPx"] = hit_px
    out["HitTime"] = hit_time

    if hit == "TP":
        out["RealizedR"] = float((target_px - entry_px) / rps)
    elif hit == "SL":
        out["RealizedR"] = -1.0
    elif hit == "TIE":
        out["RealizedR"] = float((hit_px - entry_px) / rps) if np.isfinite(hit_px) else np.nan
    else:
        out["RealizedR"] = float((hit_px - entry_px) / rps) if np.isfinite(hit_px) else np.nan

    return out

if ENABLE_REALIZED_HIT_CHECK and not df_alloc.empty:
    eval_end_et = _realized_eval_end_et()
    logger.info(f"Realized TP/SL check: scanning forward from entry until {eval_end_et} IST (tie_policy={PROB_TIE_POLICY})")

    realized_rows = []
    for _, r in df_alloc.iterrows():
        realized_rows.append(
            realized_tp_sl_outcome(
                ticker=str(r["Ticker"]),
                entry_px=float(r["Entry"]),
                stop_px=float(r["Stop"]),
                target_px=float(r["Target"]),
                as_of_et=as_of_et,
                eval_end_et=eval_end_et,
                tie_policy=PROB_TIE_POLICY,
            )
        )

    df_realized = pd.DataFrame(realized_rows)
    df_alloc = df_alloc.merge(df_realized, on="Ticker", how="left")

    hit_counts = df_alloc["Hit"].fillna("EOD").value_counts().to_dict()
    logger.info(f"Realized outcomes through {eval_end_et.time()} IST: {hit_counts}")

# ============================================================
# OUTPUT
# ============================================================
if df_alloc.empty:
    master_table = pd.DataFrame()
else:
    base_cols = [
        "Ticker","Sector","Catalyst",
        "VWAP","VWAP_+1s","VWAP_+2s","VWAP_Zone",
        "Open","Entry","Stop","ATR14","Target",
        "Shares","Position (₹)","Score","RVOL","VolZ",
        "NewsFlag","NewsLatest"
    ]
    prob_cols = ["BestRR","MaxRR_Cap","MFE_q","MAE_q","MFE_N","P_TP","P_SL","P_NONE","P_HIT","E_R_NONE","P_TP_GivenHit","EV_R","Uncertainty","EmpN"]
    take_cols = ["TAKE"] if "TAKE" in df_alloc.columns else []
    real_cols = ["Hit","HitPx","HitTime","RealizedR","CheckedThrough"]

    cols = base_cols + [c for c in prob_cols if c in df_alloc.columns] + take_cols + [c for c in real_cols if c in df_alloc.columns]
    master_table = df_alloc[cols].copy()

print("\n" + "=" * 160)
print("NSE INTRADAY SCANNER - ROBUST ML (CONSISTENT STOPS + P(NONE) AWARE EV + TARGET REACHABILITY + VWAP σ-BANDS + BREADTH REGIME)")
print("=" * 160)
print(master_table.to_string(index=False) if not master_table.empty else "(empty)")

total_exposure = float(df_alloc["Position (₹)"].sum()) if not df_alloc.empty else 0.0
portfolio_heat = float(df_alloc["Risk (₹)"].sum()) if not df_alloc.empty else 0.0

print("\n" + "=" * 160)
print("PORTFOLIO SUMMARY")
print("=" * 160)
print(f"Preset: {RELAX_PRESET} | Universe scan: {len(universe_scan)} | Universe source size: {len(universe_df)} | ML={SKLEARN_OK}")
print(f"Market Breadth A/D ratio: {breadth_ratio if np.isfinite(breadth_ratio) else 'NaN'} | Strict(20%): {breadth_strict}")
print(f"TAKE thresholds (effective): minEV={TAKE_EFF['min_ev']:.4f}, maxP_SL={TAKE_EFF['max_psl']:.4f}, minP_TP|hit={TAKE_EFF['min_ptp_hit']:.4f}, maxUnc={TAKE_EFF['max_unc']:.4f}, minEmpN={TAKE_EFF['min_emp']}")
print(f"Extra robustness: minP_HIT={TAKE_MIN_P_HIT:.2f}, maxP_NONE={TAKE_MAX_P_NONE:.2f}, reachability(ATR)={ENABLE_TARGET_REACHABILITY_FILTER} (<= {TARGET_MAX_ATR_MULT}x ATR14)")
if early_block_reasons:
    print(f"Early blocked tickers: {len(early_block_reasons)} (earnings/news)")

print(f"Total Exposure: ₹{total_exposure:,.2f}")
print(f"Portfolio Heat: ₹{portfolio_heat:,.2f}")
print(regime_note)

if ENABLE_PROB_MODEL and not df_alloc.empty and "BestRR" in df_alloc.columns:
    entry_clock = as_of_et.time() if PROB_USE_ASOF_TIME else cfg.eval_time_default
    print(f"Prob model: interval={PROB_INTERVAL}, train_days≈{PROB_TRAIN_LOOKBACK_SESSIONS}, train_tickers≤{PROB_TRAIN_TICKERS_MAX}, entry_time={entry_clock}")
    print(f"RR_GRID={RR_GRID} | BestRR chosen by max EV_R (includes E[R|NONE]) | Blend: n/(n+{PROB_PRIOR_BLEND_K}) | Tie policy: {PROB_TIE_POLICY}")

if "TAKE" in df_alloc.columns:
    print("TAKE is breadth-regime + VWAP-zone aware and now also enforces min P(HIT) + max P(NONE).")

print("=" * 160)
print("SCAN COMPLETE (IST)")
print("=" * 160)
