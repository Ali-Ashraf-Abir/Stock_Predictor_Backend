# 📈 Stock Market Trend Predictor API

This is a simple FastAPI backend that fetches historical stock prices using Yahoo Finance (`yfinance`), trains a linear regression model, and predicts the next N days of stock closing prices.

---

## 🚀 Features

- Fetches real-time historical stock data
- Trains a regression model dynamically per request
- Predicts future stock closing prices
- Simple REST API interface
- Easily connectable to a frontend (React, etc.)

---

## 📦 Tech Stack

- **FastAPI** for building the REST API
- **Scikit-learn** for the regression model
- **yfinance** to fetch stock data
- **Pandas / NumPy** for data processing
- **Uvicorn** as the ASGI server

---

## 🔧 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/stock-predictor-api.git
cd stock-predictor-api

# Install dependencies
pip install -r requirements.txt

#Running Server
uvicorn main:app --reload
