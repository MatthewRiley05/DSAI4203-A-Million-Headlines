import pandas as pd
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import pickle
import re

MIN_HEADLINE_LENGTH = 20

df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
text_col = df.columns[1]

stemmer = SnowballStemmer("english")
stop_words = ENGLISH_STOP_WORDS

def stem_analyzer(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)            # remove digits
    text = re.sub(r"[^a-z\s]", " ", text)       # remove punctuation and symbols
    text = re.sub(r"\s+", " ", text)          # normalize whitespace
    tokens = text.split()
    tokens = [t for t in text.split() if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]

    # remove short tokens consisting of less than 3 characters
    tokens = [t for t in tokens if len(t) >= 3]

    # generate 2-grams and 3-grams
    bigrams = [tokens[i] + " " + tokens[i+1] for i in range(len(tokens)-1)]
    trigrams = [tokens[i] + " " + tokens[i+1] + " " + tokens[i+2] for i in range(len(tokens) - 2)]
    
    return tokens + bigrams + trigrams

vectorizer = TfidfVectorizer(
    max_features=10000,
    min_df=10,
    max_df=0.5,
    analyzer=stem_analyzer,
    token_pattern= None
)

tfidf_matrix = vectorizer.fit_transform(df[text_col])

# Save preprocessed dataframe
df.to_csv("outputs/abcnews-cleaned.csv", index=False)

for name, obj in [("tfidf_vectorizer", vectorizer), ("tfidf_matrix", tfidf_matrix)]:
    with open(f"outputs/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

print(f"✓ Created {len(df):,} samples, {tfidf_matrix.shape[1]:,} features")
