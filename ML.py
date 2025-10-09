pip install yfinance

import yfinance as yf

nifty = yf.Ticker("^NSEI")

df = nifty.history(period="3y")

print(df)

df.to_csv("nift50_data.csv")