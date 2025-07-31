import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta

def fetch_stock_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.download(symbol, period=period)
    df = df.reset_index()[["Date", "Close"]]
    return df

def prepare_data(df: pd.DataFrame):
    df["DateInt"] = (df["Date"] - df["Date"].min()).dt.days
    X = df[["DateInt"]].values
    y = df["Close"].values
    return X, y, df["Date"]

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_future(model, last_date: pd.Timestamp, days: int, X):
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    date_ints = np.array([X[-1][0] + i for i in range(1, days + 1)]).reshape(-1, 1)
    predictions = model.predict(date_ints)
    return [
        {
            "date": d.strftime("%Y-%m-%d"),
            "predicted_close": round(p.item(), 2)  # Ensure it's a float
        }
        for d, p in zip(future_dates, predictions)
    ]
