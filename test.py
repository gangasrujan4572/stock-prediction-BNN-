import pandas as pd

# Load datasets
first = pd.read_csv("FirstRand.csv")
discovery = pd.read_csv("Discovery.csv")

# Remove the first two rows
first = first.iloc[2:].copy()
discovery = discovery.iloc[2:].copy()

# Rename Price column to Date
first.rename(columns={"Price": "Date"}, inplace=True)
discovery.rename(columns={"Price": "Date"}, inplace=True)

# Convert Date column to datetime
first["Date"] = pd.to_datetime(first["Date"])
discovery["Date"] = pd.to_datetime(discovery["Date"])

# Convert numerical columns to numbers
numeric_columns = ["Close", "High", "Low", "Open", "Volume"]

for col in numeric_columns:
    first[col] = pd.to_numeric(first[col], errors="coerce")
    discovery[col] = pd.to_numeric(discovery[col], errors="coerce")

# Reset index
first.reset_index(drop=True, inplace=True)
discovery.reset_index(drop=True, inplace=True)

print("FirstRand:")
print(first.head())
print(first.info())

print("\nDiscovery:")
print(discovery.head())
print(discovery.info())