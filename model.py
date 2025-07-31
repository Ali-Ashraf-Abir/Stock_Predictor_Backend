import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta

def fetch_stock_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
    
    # If columns are MultiIndex (for multiple tickers), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(filter(None, col)).strip() for col in df.columns.values]
    
    # Try to find 'Close' column
    if "Close" in df.columns:
        close_col = "Close"
    else:
        # Sometimes close might be like "Close <symbol>" if multiIndex flattened
        close_cols = [col for col in df.columns if col.lower().startswith("close")]
        if not close_cols:
            raise ValueError("Close price column not found in data")
        close_col = close_cols[0]
    
    df = df.reset_index()
    df = df[['Date', close_col]]
    df.rename(columns={close_col: "Close"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    return df


def prepare_data(df: pd.DataFrame):
    # Convert dates to integer days from start
    df['DateInt'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['DateInt']].values
    y = df['Close'].values
    return X, y, df['Date']

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_future(model, last_date: pd.Timestamp, days: int, X):
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    last_day_int = X[-1][0]
    future_ints = np.array([last_day_int + i for i in range(1, days + 1)]).reshape(-1, 1)
    preds = model.predict(future_ints)
    return [
        {"date": d.strftime("%Y-%m-%d"), "predicted_close": round(p.item(), 2)}
        for d, p in zip(future_dates, preds)
    ]
