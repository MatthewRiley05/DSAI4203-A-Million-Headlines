import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import pickle
import re

df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
text_col = df.columns[1]

# Enhanced preprocessing
df[text_col] = df[text_col].astype(str)
df = df.drop_duplicates(subset=[text_col])  # Remove duplicate headlines
df = df[df[text_col].str.len() > 20]  # Remove very short headlines

# Better text cleaning
df[text_col] = df[text_col].apply(
    lambda x: re.sub(r"\s+", " ", re.sub(r"\d+", "", x.lower())).strip()
)

# Custom stopwords for news headlines
custom_stopwords = list(ENGLISH_STOP_WORDS) + [
    "said",
    "says",
    "new",
    "report",
    "australia",
    "australian",
    "nsw",
    "qld",
    "vic",
    "sa",
    "wa",
    "nt",
    "tas",
    "act",
]

vectorizer = TfidfVectorizer(
    max_features=15000,  # Increased from 10000
    min_df=5,  # More lenient (was 10)
    max_df=0.7,  # More strict (was 0.5)
    ngram_range=(1, 3),
    stop_words=custom_stopwords,
    token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # Min 3 chars
    sublinear_tf=True,  # Use log-scaling for TF
    norm="l2",  # L2 normalization
)
tfidf_matrix = vectorizer.fit_transform(df[text_col])

# Save preprocessed dataframe
df.to_csv("outputs/abcnews-cleaned.csv", index=False)

for name, obj in [("tfidf_vectorizer", vectorizer), ("tfidf_matrix", tfidf_matrix)]:
    with open(f"outputs/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

print(f"✓ Created {len(df):,} samples, {tfidf_matrix.shape[1]:,} features")
