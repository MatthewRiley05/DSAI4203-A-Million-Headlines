import pandas as pd

df = pd.read_csv("abcnews-date-text.csv")
df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format="%Y%m%d", errors="coerce")
df.to_csv("outputs/abcnews-date-text-preprocessed.csv", index=False)
print("Data cleaning complete!")
