import pandas as pd

def get_all_stock_symbols():
    # Get Nasdaq listed stocks
    nasdaq_url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    nyse_url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    df1 = pd.read_csv(nasdaq_url, sep='|')
    df2 = pd.read_csv(nyse_url, sep='|')

    df = pd.concat([df1[['Symbol', 'Security Name']], df2[['Symbol', 'Security Name']]])
    df = df[df['Symbol'].str.isalpha()]  # Optional: filter clean symbols
    return df.to_dict(orient="records")
