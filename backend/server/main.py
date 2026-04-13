# backend/server/main.py

import re
import sqlite3
import json
import time
import collections

import csv
import io
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import yfinance as yf

app = FastAPI(title="Stock Dashboard Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://localhost:5177", "http://localhost:5178"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rate limiting — 60 requests per minute per IP (fixed window)
# ---------------------------------------------------------------------------
_rate_limit_store: dict = collections.defaultdict(list)
RATE_LIMIT_MAX    = 60   # max requests
RATE_LIMIT_WINDOW = 60   # seconds

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip  = request.client.host
    now = time.time()

    # Purge timestamps outside the current window
    _rate_limit_store[ip] = [
        ts for ts in _rate_limit_store[ip] if now - ts < RATE_LIMIT_WINDOW
    ]

    if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please wait a moment and try again."},
        )

    _rate_limit_store[ip].append(now)
    return await call_next(request)

# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------
DB_PATH = "cache.db"

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache "
        "(key TEXT PRIMARY KEY, data TEXT, ts REAL)"
    )
    conn.commit()
    conn.close()

_init_db()


def _cache_get(key: str, ttl: float):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT data, ts FROM cache WHERE key=?", (key,)
    ).fetchone()
    conn.close()
    if row and (time.time() - row[1]) < ttl:
        return json.loads(row[0])
    return None


def _cache_set(key: str, data):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO cache VALUES (?,?,?)",
        (key, json.dumps(data), time.time()),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
VALID_PERIODS = {
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
}
VALID_INTERVALS = {
    "1m", "5m", "15m", "30m", "60m", "1h", "1d", "1wk", "1mo"
}
TICKER_RE = re.compile(r"^[A-Z0-9.\-\^=]{1,10}$")


def _validate_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    if not TICKER_RE.match(t):
        raise HTTPException(status_code=400, detail=f"Invalid ticker symbol: '{ticker}'")
    return t


def _validate_period(period: str) -> str:
    if period not in VALID_PERIODS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period '{period}'. Valid: {sorted(VALID_PERIODS)}",
        )
    return period


def _validate_interval(interval: str) -> str:
    if interval not in VALID_INTERVALS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interval '{interval}'. Valid: {sorted(VALID_INTERVALS)}",
        )
    return interval


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------
def _flatten_df(df):
    """Flatten MultiIndex columns, reset index, ensure Date column."""
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns.to_list()]

    df = df.reset_index()

    first_col = df.columns[0]
    if first_col != "Date":
        df = df.rename(columns={first_col: "Date"})
    df["Date"] = df["Date"].astype(str)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = None

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.where(df.notnull(), None)
    return df


def _df_to_records(df) -> list:
    data = []
    for r in df.to_dict(orient="records"):
        data.append({
            "Date":   r["Date"],
            "Open":   float(r["Open"])   if r["Open"]   is not None else None,
            "High":   float(r["High"])   if r["High"]   is not None else None,
            "Low":    float(r["Low"])    if r["Low"]    is not None else None,
            "Close":  float(r["Close"])  if r["Close"]  is not None else None,
            "Volume": float(r["Volume"]) if r["Volume"] is not None else None,
        })
    return data


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
def _compute_summary_metrics(df) -> dict:
    """Compute last_close, period_return, and volatility_20d from a clean OHLCV DataFrame."""
    import pandas as pd

    close = df["Close"].dropna().astype(float)
    if close.empty:
        return {"last_close": None, "period_return": None, "volatility_20d": None}

    last_close   = round(float(close.iloc[-1]), 4)
    first_close  = float(close.iloc[0])
    period_return = round(((last_close - first_close) / first_close) * 100, 2) if first_close else None

    daily_returns = close.pct_change()
    vol_series    = daily_returns.rolling(20).std() * 100
    last_vol      = vol_series.dropna()
    volatility_20d = round(float(last_vol.iloc[-1]), 2) if not last_vol.empty else None

    return {
        "last_close":    last_close,
        "period_return": period_return,
        "volatility_20d": volatility_20d,
    }


def _compute_data_summary(df) -> dict:
    """Compute date_coverage, row_count, missing_value_count, and warnings[]."""
    warnings = []

    row_count = len(df)

    # Date coverage — first and last Date strings
    dates = df["Date"].dropna()
    date_start = str(dates.iloc[0])  if not dates.empty else None
    date_end   = str(dates.iloc[-1]) if not dates.empty else None
    date_coverage = f"{date_start} to {date_end}" if date_start and date_end else None

    # Missing value count across all OHLCV columns
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_value_count = int(df[ohlcv_cols].isnull().sum().sum())

    if missing_value_count > 0:
        warnings.append(f"{missing_value_count} missing value(s) detected in OHLCV data.")

    if row_count == 0:
        warnings.append("No data returned for the requested ticker and period.")

    return {
        "date_coverage":       date_coverage,
        "row_count":           row_count,
        "missing_value_count": missing_value_count,
        "warnings":            warnings,
    }


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
def _safe_list(series) -> list:
    """Convert pandas Series to a list of float/None, replacing NaN with None."""
    return [None if v != v else float(v) for v in series]


def _compute_indicators(df, indicator_list: list) -> dict:
    result = {}
    close = df["Close"].astype(float)

    if "MA20" in indicator_list:
        result["MA20"] = _safe_list(close.rolling(20).mean())

    if "MA50" in indicator_list:
        result["MA50"] = _safe_list(close.rolling(50).mean())

    if "MA200" in indicator_list:
        result["MA200"] = _safe_list(close.rolling(200).mean())

    if "BB" in indicator_list:
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        result["BB_upper"]  = _safe_list(sma20 + 2 * std20)
        result["BB_middle"] = _safe_list(sma20)
        result["BB_lower"]  = _safe_list(sma20 - 2 * std20)

    if "RSI" in indicator_list:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, float("nan"))
        result["RSI"] = _safe_list(100 - (100 / (1 + rs)))

    if "MACD" in indicator_list:
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        result["MACD"]        = _safe_list(macd)
        result["MACD_signal"] = _safe_list(signal)
        result["MACD_hist"]   = _safe_list(macd - signal)

    return result


# ---------------------------------------------------------------------------
# ML feature engineering
# ---------------------------------------------------------------------------
def _build_features(df):
    """
    Build a feature matrix (pandas DataFrame) with 7 features per row.

    Features:
      rsi            — RSI(14)
      macd_hist      — MACD histogram (MACD line - signal line)
      ma_ratio       — Close / MA20  (measures distance above/below trend)
      bb_position    — (Close - BB_lower) / (BB_upper - BB_lower), clamped [0,1]
      daily_return   — percentage daily return
      volatility_20  — 20-day rolling std of daily returns (as %)
      volume_change  — percentage change in volume vs previous day

    Rows where any feature is NaN are dropped — callers receive a clean matrix.
    """
    import pandas as pd
    import numpy as np

    close  = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))

    # MACD histogram
    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal    = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal

    # MA ratio
    ma20     = close.rolling(20).mean()
    ma_ratio = close / ma20

    # BB position
    std20    = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_range = bb_upper - bb_lower
    bb_pos   = (close - bb_lower) / bb_range.replace(0, np.nan)
    bb_pos   = bb_pos.clip(0, 1)

    # Daily return
    daily_return = close.pct_change() * 100

    # 20-day rolling volatility of daily returns
    volatility_20 = daily_return.rolling(20).std()

    # Volume change
    volume_change = volume.pct_change() * 100

    # ma_ratio, bb_position, and rsi are excluded — they directly encode
    # Close vs MA20, which is the label definition, causing perfect leakage.
    # The 4 retained features are independent momentum/volatility signals.
    features = pd.DataFrame({
        "macd_hist":     macd_hist,
        "daily_return":  daily_return,
        "volatility_20": volatility_20,
        "volume_change": volume_change,
    }, index=df.index)

    return features


# ---------------------------------------------------------------------------
# ML model training
# ---------------------------------------------------------------------------

# In-memory model cache: key = "ticker:period:interval" → trained model bundle
_model_cache: dict = {}

def _train_model(df, labels: list) -> dict:
    """
    Time-split train/test and fit a RandomForestClassifier.

    Split rule: last 30 labeled rows = test, everything before = train.
    Returns a bundle dict with the fitted model, split indices, and
    the aligned feature/label arrays ready for evaluation.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    features = _build_features(df)

    # Align features with labels, drop rows where any feature is NaN
    import pandas as pd
    label_series = pd.Series(labels, index=df.index, name="label")
    combined     = features.join(label_series).dropna()

    # Only keep rows with valid labels (no None)
    combined = combined[combined["label"].isin(["invest", "no-invest"])]

    if len(combined) < 40:
        raise ValueError("Not enough clean data to train (need at least 40 labeled rows after NaN drop).")

    X = combined[features.columns].values
    y = (combined["label"] == "invest").astype(int).values  # 1=invest, 0=no-invest

    # Time-based split — last 30 rows as test
    split = len(X) - 30
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    return {
        "model":          clf,
        "feature_names":  list(features.columns),
        "X_test":         X_test,
        "y_test":         y_test,
        "train_size":     len(X_train),
        "test_size":      len(X_test),
        "last_row":       X[-1],          # most recent row for prediction
    }


# ---------------------------------------------------------------------------
# Label generation (Stage 4)
# ---------------------------------------------------------------------------
def _compute_labels(df) -> list:
    """Return a label per row: 'invest' if Close > MA20, else 'no-invest'.

    Rows where MA20 is not yet available (first 19 rows) receive None.
    """
    close = df["Close"].astype(float)
    ma20  = close.rolling(20).mean()
    labels = []
    for c, m in zip(close, ma20):
        if m != m:          # NaN check — rolling hasn't filled yet
            labels.append(None)
        elif c > m:
            labels.append("invest")
        else:
            labels.append("no-invest")
    return labels


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ohlcv")
def ohlcv(
    ticker: str = "NVDA",
    period: str = "1y",
    interval: str = "1d",
    indicators: str = "",
):
    ticker   = _validate_ticker(ticker)
    period   = _validate_period(period)
    interval = _validate_interval(interval)

    indicator_list = [i.strip() for i in indicators.split(",") if i.strip()]

    # TTL: shorter for intraday data
    intraday = interval in {"1m", "5m", "15m", "30m", "60m", "1h"}
    ttl = 60 if intraday else 300

    cache_key = f"{ticker}:{period}:{interval}"
    cached_records = _cache_get(cache_key, ttl)

    if cached_records is not None:
        import pandas as pd
        df_cached = pd.DataFrame(cached_records)
        inds    = _compute_indicators(df_cached, indicator_list) if indicator_list else {}
        metrics = _compute_summary_metrics(df_cached)
        summary = _compute_data_summary(df_cached)
        labels  = _compute_labels(df_cached)
        return {"ticker": ticker, "data": cached_records, "indicators": inds, "metrics": metrics, "summary": summary, "labels": labels}

    # Fetch from Yahoo Finance
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Yahoo Finance error: {exc}")

    if df is None or df.empty:
        empty_summary = {"date_coverage": None, "row_count": 0, "missing_value_count": 0, "warnings": ["No data returned for the requested ticker and period."]}
        return {"ticker": ticker, "data": [], "indicators": {}, "metrics": {"last_close": None, "period_return": None, "volatility_20d": None}, "summary": empty_summary, "labels": []}

    df = _flatten_df(df)
    records = _df_to_records(df)
    _cache_set(cache_key, records)

    inds    = _compute_indicators(df, indicator_list) if indicator_list else {}
    metrics = _compute_summary_metrics(df)
    summary = _compute_data_summary(df)
    labels  = _compute_labels(df)

    return {"ticker": ticker, "data": records, "indicators": inds, "metrics": metrics, "summary": summary, "labels": labels}


@app.get("/compare")
def compare(
    tickers: str = "NVDA,AAPL",
    period: str = "1y",
    interval: str = "1d",
):
    period   = _validate_period(period)
    interval = _validate_interval(interval)

    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")
    if len(ticker_list) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 tickers for comparison.")

    ticker_list = [_validate_ticker(t) for t in ticker_list]

    intraday = interval in {"1m", "5m", "15m", "30m", "60m", "1h"}
    ttl = 60 if intraday else 300

    series: dict = {}
    dates: list = []

    for t in ticker_list:
        cache_key = f"{t}:{period}:{interval}"
        records = _cache_get(cache_key, ttl)

        if records is None:
            try:
                df = yf.download(t, period=period, interval=interval, progress=False)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Yahoo Finance error for {t}: {exc}")

            if df is None or df.empty:
                continue

            df = _flatten_df(df)
            records = _df_to_records(df)
            _cache_set(cache_key, records)

        if not records:
            continue

        closes = [r["Close"] for r in records if r["Close"] is not None]
        record_dates = [r["Date"] for r in records if r["Close"] is not None]

        if not closes:
            continue

        base = closes[0]
        normalized = [round((c / base) * 100, 4) for c in closes]
        series[t] = normalized

        # Use the longest date list as the reference
        if len(record_dates) > len(dates):
            dates = record_dates

    return {"tickers": list(series.keys()), "dates": dates, "series": series}


# ---------------------------------------------------------------------------
# Backtest strategy computation
# ---------------------------------------------------------------------------
def _run_backtest(df, labels: list, initial_capital: float) -> dict:
    """
    Simulate the label-based strategy vs buy-and-hold benchmark.

    Strategy rule:
      - "invest" day  → hold the stock (capture that day's return)
      - "no-invest" / None day → stay in cash (return = 0)

    Day 0 has no previous close so its return is treated as 0 for both curves.

    Returns a dict with:
      portfolio_value[]  — strategy equity curve
      benchmark_value[]  — buy-and-hold equity curve
      dates[]            — corresponding date strings
    """
    closes = df["Close"].astype(float).tolist()
    dates  = df["Date"].astype(str).tolist()
    n = len(closes)

    port_val  = initial_capital
    bench_val = initial_capital
    portfolio_value = []
    benchmark_value = []

    for i in range(n):
        if i == 0:
            daily_return = 0.0
        else:
            prev = closes[i - 1]
            daily_return = ((closes[i] - prev) / prev) if prev else 0.0

        # Benchmark always invested
        bench_val *= (1.0 + daily_return)

        # Strategy only invested on "invest" days
        if labels[i] == "invest":
            port_val *= (1.0 + daily_return)

        portfolio_value.append(round(port_val, 4))
        benchmark_value.append(round(bench_val, 4))

    return {
        "dates":           dates,
        "portfolio_value": portfolio_value,
        "benchmark_value": benchmark_value,
    }


# ---------------------------------------------------------------------------
# Backtest summary metrics
# ---------------------------------------------------------------------------
def _compute_backtest_summary(portfolio_value: list, benchmark_value: list, labels: list) -> dict:
    """
    Compute summary metrics from the equity curves and labels.

    strategy_return  — total % return of the label-based strategy
    buyhold_return   — total % return of buy-and-hold benchmark
    max_drawdown     — largest peak-to-trough % decline in the strategy equity curve
    invest_days      — number of rows where label == "invest"
    """
    if not portfolio_value or not benchmark_value:
        return {
            "strategy_return": None,
            "buyhold_return":  None,
            "max_drawdown":    None,
            "invest_days":     0,
        }

    initial = portfolio_value[0]

    strategy_return = round(((portfolio_value[-1] - initial) / initial) * 100, 2) if initial else None
    buyhold_return  = round(((benchmark_value[-1] - initial) / initial) * 100, 2) if initial else None

    # Max drawdown: largest % drop from a running peak
    peak = portfolio_value[0]
    max_dd = 0.0
    for v in portfolio_value:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
    max_drawdown = round(max_dd, 2)

    invest_days = sum(1 for l in labels if l == "invest")

    return {
        "strategy_return": strategy_return,
        "buyhold_return":  buyhold_return,
        "max_drawdown":    max_drawdown,
        "invest_days":     invest_days,
    }


# ---------------------------------------------------------------------------
# Backtest endpoint
# ---------------------------------------------------------------------------
@app.get("/backtest")
def backtest(
    ticker: str = "NVDA",
    period: str = "1y",
    interval: str = "1d",
    initial_capital: float = 10000.0,
):
    # --- Input validation ---
    ticker   = _validate_ticker(ticker)
    period   = _validate_period(period)
    interval = _validate_interval(interval)

    if initial_capital <= 0:
        raise HTTPException(status_code=400, detail="initial_capital must be greater than 0.")
    if initial_capital > 1_000_000_000:
        raise HTTPException(status_code=400, detail="initial_capital must be 1,000,000,000 or less.")

    # Intraday intervals not meaningful for daily label-based strategy
    if interval in {"1m", "5m", "15m", "30m", "60m", "1h"}:
        raise HTTPException(
            status_code=400,
            detail="Backtesting requires a daily or wider interval (1d, 1wk, 1mo).",
        )

    # --- Fetch / cache OHLCV ---
    ttl = 300
    cache_key = f"{ticker}:{period}:{interval}"
    records = _cache_get(cache_key, ttl)

    if records is None:
        try:
            import pandas as pd
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Yahoo Finance error: {exc}")

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data returned for '{ticker}'.")

        df = _flatten_df(df)
        records = _df_to_records(df)
        _cache_set(cache_key, records)

    if len(records) < 20:
        raise HTTPException(
            status_code=400,
            detail="Not enough data to run backtest (need at least 20 rows for MA20).",
        )

    import pandas as pd
    df = pd.DataFrame(records)
    labels = _compute_labels(df)

    result  = _run_backtest(df, labels, initial_capital)
    summary = _compute_backtest_summary(result["portfolio_value"], result["benchmark_value"], labels)

    return {
        "ticker":           ticker,
        "period":           period,
        "interval":         interval,
        "initial_capital":  initial_capital,
        "dates":            result["dates"],
        "portfolio_value":  result["portfolio_value"],
        "benchmark_value":  result["benchmark_value"],
        "summary":          summary,
    }


# ---------------------------------------------------------------------------
# Export endpoint
# ---------------------------------------------------------------------------
@app.get("/export")
def export_csv(
    ticker: str = "NVDA",
    period: str = "1y",
    interval: str = "1d",
):
    ticker   = _validate_ticker(ticker)
    period   = _validate_period(period)
    interval = _validate_interval(interval)

    intraday = interval in {"1m", "5m", "15m", "30m", "60m", "1h"}
    ttl = 60 if intraday else 300
    cache_key = f"{ticker}:{period}:{interval}"
    records = _cache_get(cache_key, ttl)

    if records is None:
        try:
            import pandas as pd
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Yahoo Finance error: {exc}")

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data returned for '{ticker}'.")

        df = _flatten_df(df)
        records = _df_to_records(df)
        _cache_set(cache_key, records)

    import pandas as pd
    df = pd.DataFrame(records)

    # Compute all indicators
    all_indicators = ["MA20", "MA50", "BB", "RSI", "MACD"]
    inds   = _compute_indicators(df, all_indicators)
    labels = _compute_labels(df)

    # Build CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Metadata header rows
    writer.writerow(["# Ticker",   ticker])
    writer.writerow(["# Period",   period])
    writer.writerow(["# Interval", interval])
    writer.writerow(["# Exported", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")])
    writer.writerow([])  # blank separator

    # Column headers
    writer.writerow([
        "Date", "Open", "High", "Low", "Close", "Volume",
        "MA20", "MA50", "BB_Upper", "BB_Lower",
        "RSI", "MACD", "Label",
    ])

    # Data rows
    n = len(records)
    for i in range(n):
        r = records[i]
        def fmt(v):
            return round(v, 4) if v is not None else ""

        writer.writerow([
            r["Date"],
            fmt(r["Open"]),
            fmt(r["High"]),
            fmt(r["Low"]),
            fmt(r["Close"]),
            fmt(r["Volume"]),
            fmt(inds.get("MA20",  [None]*n)[i]),
            fmt(inds.get("MA50",  [None]*n)[i]),
            fmt(inds.get("BB_upper", [None]*n)[i]),
            fmt(inds.get("BB_lower", [None]*n)[i]),
            fmt(inds.get("RSI",   [None]*n)[i]),
            fmt(inds.get("MACD",  [None]*n)[i]),
            labels[i] if labels[i] is not None else "",
        ])

    output.seek(0)
    filename = f"{ticker}_{period}_{interval}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/export/backtest")
def export_backtest_csv(
    ticker: str = "NVDA",
    period: str = "1y",
    interval: str = "1d",
    initial_capital: float = 10000.0,
):
    ticker   = _validate_ticker(ticker)
    period   = _validate_period(period)
    interval = _validate_interval(interval)

    if initial_capital <= 0:
        raise HTTPException(status_code=400, detail="initial_capital must be greater than 0.")

    if interval in {"1m", "5m", "15m", "30m", "60m", "1h"}:
        raise HTTPException(
            status_code=400,
            detail="Backtesting requires a daily or wider interval (1d, 1wk, 1mo).",
        )

    ttl = 300
    cache_key = f"{ticker}:{period}:{interval}"
    records = _cache_get(cache_key, ttl)

    if records is None:
        try:
            import pandas as pd
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Yahoo Finance error: {exc}")

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data returned for '{ticker}'.")

        df = _flatten_df(df)
        records = _df_to_records(df)
        _cache_set(cache_key, records)

    if len(records) < 20:
        raise HTTPException(
            status_code=400,
            detail="Not enough data to run backtest (need at least 20 rows for MA20).",
        )

    import pandas as pd
    df      = pd.DataFrame(records)
    labels  = _compute_labels(df)
    result  = _run_backtest(df, labels, initial_capital)
    summary = _compute_backtest_summary(result["portfolio_value"], result["benchmark_value"], labels)

    # Build CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Metadata header rows
    writer.writerow(["# Ticker",           ticker])
    writer.writerow(["# Period",           period])
    writer.writerow(["# Interval",         interval])
    writer.writerow(["# Initial Capital",  initial_capital])
    writer.writerow(["# Strategy Return",  f"{summary['strategy_return']}%"])
    writer.writerow(["# Buy & Hold Return",f"{summary['buyhold_return']}%"])
    writer.writerow(["# Max Drawdown",     f"{summary['max_drawdown']}%"])
    writer.writerow(["# Invest Days",      summary["invest_days"]])
    writer.writerow(["# Exported",         datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")])
    writer.writerow([])  # blank separator

    # Column headers
    writer.writerow(["Date", "Strategy_Value", "Benchmark_Value", "Label"])

    # Data rows
    for i, date in enumerate(result["dates"]):
        writer.writerow([
            date,
            result["portfolio_value"][i],
            result["benchmark_value"][i],
            labels[i] if labels[i] is not None else "",
        ])

    output.seek(0)
    filename = f"{ticker}_{period}_{interval}_backtest.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------
@app.get("/predict")
def predict(
    ticker: str = "NVDA",
    period: str = "1y",
    interval: str = "1d",
):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pandas as pd

    ticker   = _validate_ticker(ticker)
    period   = _validate_period(period)
    interval = _validate_interval(interval)

    if interval in {"1m", "5m", "15m", "30m", "60m", "1h"}:
        raise HTTPException(
            status_code=400,
            detail="ML prediction requires a daily or wider interval (1d, 1wk, 1mo).",
        )

    # Fetch / cache OHLCV
    ttl = 300
    cache_key = f"{ticker}:{period}:{interval}"
    records = _cache_get(cache_key, ttl)

    if records is None:
        try:
            df_raw = yf.download(ticker, period=period, interval=interval, progress=False)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Yahoo Finance error: {exc}")

        if df_raw is None or df_raw.empty:
            raise HTTPException(status_code=404, detail=f"No data returned for '{ticker}'.")

        df_raw = _flatten_df(df_raw)
        records = _df_to_records(df_raw)
        _cache_set(cache_key, records)

    if len(records) < 60:
        raise HTTPException(
            status_code=400,
            detail="Not enough data for ML (need at least 60 rows).",
        )

    df = pd.DataFrame(records)
    labels = _compute_labels(df)

    # Use cached model if available, otherwise train
    model_key = f"{ticker}:{period}:{interval}"
    if model_key not in _model_cache:
        try:
            bundle = _train_model(df, labels)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _model_cache[model_key] = bundle
    else:
        bundle = _model_cache[model_key]

    clf           = bundle["model"]
    X_test        = bundle["X_test"]
    y_test        = bundle["y_test"]
    feature_names = bundle["feature_names"]
    last_row      = bundle["last_row"]

    # Predict on latest row
    pred     = clf.predict([last_row])[0]
    proba    = clf.predict_proba([last_row])[0]
    signal   = "BUY" if pred == 1 else "SELL"
    confidence = round(float(max(proba)) * 100, 1)

    # Evaluation metrics on held-out test set
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    # Feature importances
    feature_importances = {
        name: round(float(imp), 4)
        for name, imp in zip(feature_names, clf.feature_importances_)
    }

    # ML artifact metadata
    ml_metadata = {
        "ticker":      ticker,
        "date_range":  f"{records[0]['Date']} to {records[-1]['Date']}",
        "random_seed": 42,
        "train_size":  bundle["train_size"],
        "test_size":   bundle["test_size"],
    }

    return {
        "ticker":               ticker,
        "signal":               signal,
        "confidence":           confidence,
        "metrics":              metrics,
        "feature_importances":  feature_importances,
        "ml_metadata":          ml_metadata,
    }
