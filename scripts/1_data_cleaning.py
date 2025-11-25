import pandas as pd

# Load the dataset
df = pd.read_csv("abcnews-date-text.csv")

# Convert the date column to datetime
date_column = "publish_date"
df[date_column] = pd.to_datetime(df[date_column], format="%Y%m%d", errors="coerce")

print("Saving preprocessed data...")
df.to_csv("outputs/abcnews-date-text-preprocessed.csv", index=False)
print("Data cleaning complete!")
