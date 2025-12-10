import pandas as pd
import os

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("abcnews-date-text.csv")
df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format="%Y%m%d", errors="coerce")
df.to_csv("outputs/abcnews-date-text-preprocessed.csv", index=False)
print(f"✓ Cleaned {len(df):,} headlines")
