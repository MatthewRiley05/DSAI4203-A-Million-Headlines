import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
text_col = df.columns[1]

# Clean text
df[text_col] = df[text_col].apply(
    lambda x: re.sub(r"\s+", " ", re.sub(r"\d+", "", x.lower())).strip()
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=10000,
    min_df=10,
    max_df=0.5,
    ngram_range=(1, 3),
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
)
tfidf_matrix = vectorizer.fit_transform(df[text_col])

# Save
with open("outputs/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("outputs/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)
print("Feature engineering complete!")
