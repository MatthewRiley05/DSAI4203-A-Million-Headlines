import pandas as pd
import numpy as np
import pickle, os
from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessing import stem_analyzer
import logging
import builtins

os.makedirs("outputs/models/semisupervised", exist_ok=True)

# Logging allows to print everything to a log file
logging.basicConfig(
    filename="outputs/logs/8_2_feature_engineering.log",
    level=logging.INFO,
    format="%(message)s",
    filemode="w"
)

logger= logging.getLogger()
builtins.print = logger.info

# Load merged dataset
df_lab = pd.read_csv("dataset/merged_news_labels.csv")
df_orig = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
print(f"Labeled data shape: {df_lab.shape}")
print(f"Original data shape: {df_orig.shape}")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=10000,
    min_df=10,
    max_df=0.5,
    analyzer=stem_analyzer,
    token_pattern=None,
)

tfidf_matrix_lab = vectorizer.fit_transform(df_lab["title"])

# Apply same vectorizer to 1_Mil headlines
text_col = df_orig.columns[1]
tfidf_matrix_orig = vectorizer.transform(df_orig[text_col])

print(f"TF-IDF shape: {tfidf_matrix_lab.shape}")
print("\n First 100 features:")
first_hundread_features= vectorizer.get_feature_names_out()[:100]
print(first_hundread_features.tolist())

# ============================
# Save TF-IDF features + vectorizer
# ============================
with open("outputs/models/semisupervised/tfidf_matrix_lab.pkl", "wb") as f:
    pickle.dump(tfidf_matrix_lab, f)

with open("outputs/models/semisupervised/tfidf_matrix_orig.pkl", "wb") as f:
    pickle.dump(tfidf_matrix_orig, f)

with open("outputs/models/semisupervised/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Saved TF-IDF features.")