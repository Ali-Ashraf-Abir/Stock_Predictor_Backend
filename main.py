from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from model import fetch_stock_data, prepare_data, train_model, predict_future

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict_stock(symbol: str = Query(...), days: int = Query(7)):
    try:
        df = fetch_stock_data(symbol)
        X, y, dates = prepare_data(df)
        model = train_model(X, y)
        prediction = predict_future(model, dates.iloc[-1], days, X)
        return {
            "symbol": symbol.upper(),
            "last_date": dates.iloc[-1].strftime("%Y-%m-%d"),
            "predictions": prediction
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
def get_history(symbol: str = Query(...), period: str = "6mo"):
    try:
        df = fetch_stock_data(symbol, period)
        return [
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": round(row["Close"], 2)
            }
            for _, row in df.iterrows()
        ]
    except Exception as e:
        return {"error": str(e)}
