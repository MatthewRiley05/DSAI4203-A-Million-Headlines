import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessing import stem_analyzer
import pickle
import os

MIN_HEADLINE_LENGTH = 20

os.makedirs("outputs/models", exist_ok=True)

df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
text_col = df.columns[1]

vectorizer = TfidfVectorizer(
    max_features=10000,
    min_df=10,
    max_df=0.5,
    analyzer=stem_analyzer,
    token_pattern=None,
)

tfidf_matrix = vectorizer.fit_transform(df[text_col])

# Save preprocessed dataframe
df.to_csv("outputs/abcnews-cleaned.csv", index=False)

for name, obj in [("tfidf_vectorizer", vectorizer), ("tfidf_matrix", tfidf_matrix)]:
    with open(f"outputs/models/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

print(f"✓ Created {len(df):,} samples, {tfidf_matrix.shape[1]:,} features")
