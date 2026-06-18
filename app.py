# stock_app_fixed.py
import os
import joblib
import pandas as pd
import numpy as np
import importlib
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt

# dynamic imports so script can run in non-streamlit contexts
try:
    st = importlib.import_module("streamlit")
except Exception:
    st = None
try:
    yf = importlib.import_module("yfinance")
except Exception:
    yf = None
try:
    pdr = importlib.import_module("pandas_datareader.data")
except Exception:
    pdr = None

try:
    from keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None

MODEL_DIR = Path("models")

# -------------------------
# Indicator helpers
# -------------------------
def sma(x, n):
    return x.rolling(n, min_periods=1).mean()

def ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=1).mean()

def rsi(x, n=14):
    delta = x.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.rolling(n, min_periods=1).mean()
    ma_down = down.rolling(n, min_periods=1).mean().replace(0, np.nan)
    rs = ma_up.div(ma_down).replace([np.inf, -np.inf], np.nan).fillna(0)
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)

def macd(x, span_short=12, span_long=26, span_signal=9):
    short = ema(x, span_short)
    long = ema(x, span_long)
    macd_line = short - long
    signal = macd_line.ewm(span=span_signal, adjust=False, min_periods=1).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def atr(df, n=14):
    # return Series of ATR or an empty Series if input invalid
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    if not {"High", "Low", "Close"}.issubset(set(df.columns)):
        return pd.Series(np.nan, index=df.index)
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    tr1 = (h - l).abs()
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def obv(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    if "Close" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    close = df["Close"]
    vol = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)
    out = [0.0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            out.append(out[-1] + float(vol.iloc[i]))
        elif close.iloc[i] < close.iloc[i - 1]:
            out.append(out[-1] - float(vol.iloc[i]))
        else:
            out.append(out[-1])
    return pd.Series(out, index=df.index)

def momentum(x, n=10):
    return x - x.shift(n)

def add_pro_indicators(df):
    """
    Adds indicators to a DataFrame that at least contains 'Close'.
    Defensive: returns df copy and avoids any boolean checks on Series.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("input must be a pandas DataFrame")
    if df.empty:
        return df.copy()
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    df = df.copy()
    df["SMA_14"] = sma(df["Close"], 14)
    df["SMA_50"] = sma(df["Close"], 50)
    df["EMA_20"] = ema(df["Close"], 20)
    df["RSI_14"] = rsi(df["Close"], 14)
    macd_line, macd_signal, macd_hist = macd(df["Close"])
    df["MACD"] = macd_line
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist
    df["ATR_14"] = atr(df, 14)
    df["OBV"] = obv(df) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)
    df["MOM_10"] = momentum(df["Close"], 10)

    # replace infinite, back/forward fill and then fill leftover with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(method="ffill").fillna(0)
    return df

# -------------------------
# Feature building / selection
# -------------------------
def build_rf_features_from_df(df):
    keys = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_14", "SMA_50", "EMA_20", "RSI_14",
        "MACD", "MACD_signal", "MACD_hist", "ATR_14", "OBV", "MOM_10"
    ]
    row = {}
    for k in keys:
        if k in df.columns and not df[k].isna().all():
            try:
                # take last valid numeric value
                row[k] = float(df[k].dropna().iloc[-1])
            except Exception:
                row[k] = 0.0
        else:
            row[k] = 0.0
    return pd.DataFrame([row])

def select_features_for_model(feature_df, rf_model):
    """
    Selects features that the rf_model expects (if available).
    Defensive: avoids boolean evaluation of arrays/Series.
    Returns: (selected_feature_df, feature_list_used)
    """
    expected = None
    if rf_model is not None and hasattr(rf_model, "feature_names_in_"):
        try:
            # convert to list safely
            expected = list(rf_model.feature_names_in_)
        except Exception:
            expected = None

    cols = []
    if expected is not None and isinstance(expected, (list, tuple)) and len(expected) > 0:
        cols = [c for c in expected if c in feature_df.columns]
        # fallback if none of expected features present
        if len(cols) == 0:
            cols = feature_df.columns.tolist()[: min(7, feature_df.shape[1])]
    else:
        # default heuristic
        default = ["Close", "SMA_14", "EMA_20", "RSI_14", "MACD", "MACD_signal", "ATR_14"]
        cols = [c for c in default if c in feature_df.columns]
        # pad with any other available columns until we have up to 7
        if len(cols) < 7:
            for c in feature_df.columns:
                if c not in cols:
                    cols.append(c)
                if len(cols) >= 7:
                    break

    # Ensure we do not slice with zero columns
    if len(cols) == 0 and feature_df.shape[1] > 0:
        cols = feature_df.columns.tolist()[:1]

    cols = cols[: max(1, min(len(cols), feature_df.shape[1]))]
    return feature_df[cols], cols

# -------------------------
# Model loading wrappers
# -------------------------
def load_rf_model_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return joblib.load(str(p))

def load_lstm_model_file(path):
    p = Path(path)
    if keras_load_model is None:
        raise RuntimeError("keras not available")
    if not p.exists():
        raise FileNotFoundError(str(p))
    return keras_load_model(str(p))

def load_scaler_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return joblib.load(str(p))

# -------------------------
# LSTM input + predict helpers
# -------------------------
def prepare_lstm_input(df, scaler, lookback=60):
    if "Close" not in df.columns:
        raise ValueError("no 'Close' in data")
    closes = df["Close"].values.reshape(-1, 1)
    if closes.size == 0:
        raise ValueError("no close prices")
    if len(closes) < lookback:
        pad_len = lookback - len(closes)
        # pad by repeating the first close value
        pad = np.repeat([[closes[0, 0]]], pad_len, axis=0)
        closes = np.vstack([pad, closes])
    scaled = scaler.transform(closes)
    return scaled[-lookback:].reshape(1, lookback, 1)

def predict_lstm(lstm_model, scaler, df, lookback=60):
    seq = prepare_lstm_input(df, scaler, lookback)
    pred_scaled = lstm_model.predict(seq)
    try:
        inv = scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1))
        return float(inv[-1, 0])
    except Exception:
        return float(np.asarray(pred_scaled).flatten()[-1])

# -------------------------
# RF prediction
# -------------------------
def predict_rf_from_df(rf_model, feature_df):
    X_sel, cols = select_features_for_model(feature_df, rf_model)
    arr = X_sel.values
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    pred = rf_model.predict(arr)
    return float(np.asarray(pred).flatten()[0]), cols

# -------------------------
# Data download
# -------------------------
def download_data(ticker, period_days=365):
    end = datetime.now()
    start = end - timedelta(days=period_days)
    if yf is not None:
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
    elif pdr is not None:
        df = pdr.get_data_yahoo(ticker, start=start, end=end)
    else:
        raise ModuleNotFoundError("neither yfinance nor pandas_datareader is available; pip install yfinance or pandas_datareader")
    if df is None or getattr(df, "empty", True):
        raise ValueError("no data")
    df = df.reset_index()
    expect = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for c in expect:
        if c not in df.columns:
            df[c] = 0
    df = df[expect]
    df.set_index("Date", inplace=True)
    return df

# -------------------------
# Streamlit app
# -------------------------
def run_streamlit_app():
    st.set_page_config(layout="wide", page_title="Stock App")
    st.title("TCS Stock - Pro Indicators + RF + LSTM (Fixed)")

    st.sidebar.header("Config")
    ticker_input = st.sidebar.text_input("Ticker", "TCS.NS")
    days = st.sidebar.slider("Days", 90, 1825, 365)
    show = st.sidebar.checkbox("Show Indicators", True)

    lstm_p = MODEL_DIR / "lstm_tcs_close.h5"
    rf_p = MODEL_DIR / "rf_tcs_close.joblib"
    scalers = list(MODEL_DIR.glob("*scaler*"))
    scaler_p = scalers[0] if len(scalers) > 0 else MODEL_DIR / "lstm_scaler.save"

    st.sidebar.write("Model files in models/:")
    for f in MODEL_DIR.glob("*"):
        st.sidebar.write(f.name)

    if st.sidebar.button("Load Models"):
        # load models into session state with clear errors
        try:
            st.session_state["rf"] = load_rf_model_file(rf_p)
            st.sidebar.success("RF loaded")
        except Exception as e:
            st.sidebar.error(f"RF load: {e}")
            st.session_state["rf"] = None
        try:
            st.session_state["lstm"] = load_lstm_model_file(lstm_p)
            st.sidebar.success("LSTM loaded")
        except Exception as e:
            st.sidebar.error(f"LSTM load: {e}")
            st.session_state["lstm"] = None
        try:
            st.session_state["scaler"] = load_scaler_file(scaler_p)
            st.sidebar.success("Scaler loaded")
        except Exception as e:
            st.sidebar.error(f"Scaler load: {e}")
            st.session_state["scaler"] = None

    rf_model = st.session_state.get("rf")
    lstm_model = st.session_state.get("lstm")
    scaler = st.session_state.get("scaler")

    # Data fetch + indicators
    try:
        df = download_data(ticker_input, days)
        df = add_pro_indicators(df)
        st.success("Data & indicators loaded")
    except Exception as e:
        # show full traceback-style message
        st.error(f"Data download / indicator error: {e}")
        return

    # show chart / data
    if "Close" in df.columns and not df["Close"].empty:
        st.line_chart(df["Close"])
    else:
        st.line_chart(pd.Series(dtype=float))

    if show:
        st.dataframe(df.tail(50))

    # Prediction controls
    with st.sidebar.expander("Predictions"):
        if rf_model is None or lstm_model is None or scaler is None:
            st.info("Load models to enable predictions")
        else:
            feat_df = build_rf_features_from_df(df)
            st.write("Feature snapshot:", feat_df.to_dict(orient="records")[0])

            # RF prediction
            try:
                rf_pred, used = predict_rf_from_df(rf_model, feat_df)
                st.metric("RF next prediction", f"{rf_pred:.2f}")
                st.caption("RF used features: " + ", ".join(used))
            except Exception as e:
                st.error(f"RF predict: {e}")

            # LSTM prediction
            try:
                lstm_pred = predict_lstm(lstm_model, scaler, df)
                st.metric("LSTM next prediction", f"{lstm_pred:.2f}")
            except Exception as e:
                st.error(f"LSTM predict: {e}")

    # Plot indicators
    fig, ax = plt.subplots(figsize=(10, 4))
    if "Close" in df.columns:
        ax.plot(df.index, df["Close"], label="Close")
    if "SMA_14" in df.columns:
        ax.plot(df.index, df["SMA_14"], label="SMA14")
    if "EMA_20" in df.columns:
        ax.plot(df.index, df["EMA_20"], label="EMA20")
    ax.legend()
    st.pyplot(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--streamlit", action="store_true")
    parser.add_argument("--ticker", type=str, default="TCS.NS")
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    if args.streamlit:
        if st is None:
            print("streamlit is not installed. Run 'pip install streamlit' or run without --streamlit")
        else:
            run_streamlit_app()
    else:
        df = download_data(args.ticker, args.days)
        df = add_pro_indicators(df)
        print(df[["Close", "SMA_14", "RSI_14"]].tail())
