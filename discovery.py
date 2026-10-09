import yfinance as yf

df = yf.download(
    "DSY.JO",
    start="2005-01-01",
    end="2025-06-30"
)

df.to_csv("Discovery.csv")