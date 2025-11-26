import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
text_col = df.columns[1]
df[text_col] = df[text_col].apply(
    lambda x: re.sub(r"\s+", " ", re.sub(r"\d+", "", x.lower())).strip()
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    min_df=10,
    max_df=0.5,
    ngram_range=(1, 3),
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
)
tfidf_matrix = vectorizer.fit_transform(df[text_col])

for name, obj in [("tfidf_vectorizer", vectorizer), ("tfidf_matrix", tfidf_matrix)]:
    with open(f"outputs/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
print("Feature engineering complete!")
