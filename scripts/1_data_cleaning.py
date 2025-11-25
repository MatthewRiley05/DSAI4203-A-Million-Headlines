import pandas as pd

# Load the dataset
df = pd.read_csv("abcnews-date-text.csv")

# Convert the date column to datetime
date_column = df.columns[0]
df[date_column] = pd.to_datetime(df[date_column], format="%Y%m%d", errors="coerce")

# Save preprocessed data
df.to_csv("outputs/abcnews-date-text-preprocessed.csv", index=False)
