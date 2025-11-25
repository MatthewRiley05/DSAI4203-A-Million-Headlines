import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load preprocessed data
df = pd.read_csv('outputs/abcnews-date-text-preprocessed.csv')

# Get the text column (assuming second column is the headline text)
text_column = df.columns[1]

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words="english",
)

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])

# Save the vectorizer for later use
with open('outputs/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Save the TF-IDF matrix
with open('outputs/tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)
