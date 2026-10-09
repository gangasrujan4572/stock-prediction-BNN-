import yfinance as yf

# FirstRand
df = yf.download(
    "FSR.JO",
    start="2005-01-01",
    end="2025-06-30"
)

df.to_csv("FirstRand.csv")

print(df.head())