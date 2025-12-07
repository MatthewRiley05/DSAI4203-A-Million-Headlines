import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessing import stem_analyzer
import pickle

df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
text_col = df.columns[1]

vectorizer = TfidfVectorizer(
    max_features=10000,
    min_df=10,
    max_df=0.5,
    analyzer=stem_analyzer,
    token_pattern= None
)

tfidf_matrix = vectorizer.fit_transform(df[text_col])

for name, obj in [("tfidf_vectorizer", vectorizer), ("tfidf_matrix", tfidf_matrix)]:
    with open(f"outputs/models/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
print("Feature engineering complete!")
