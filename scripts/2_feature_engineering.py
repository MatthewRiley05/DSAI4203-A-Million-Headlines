import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
print("Downloading NLTK resources...")
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("omw-1.4", quiet=True)

print("Loading preprocessed data...")
df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")

# Get the text column (assuming second column is the headline text)
text_column = df.columns[1]

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


# Clean and lemmatize text
def clean_and_lemmatize(text):
    # Remove numbers (including those with m, k suffixes like 100m, 5k)
    text = re.sub(r"\b\d+[mk]?\b", "", text)
    # Remove standalone numbers
    text = re.sub(r"\d+", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize and lemmatize
    tokens = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]

    return " ".join(lemmatized)


print("Cleaning and lemmatizing text...")
df[text_column] = df[text_column].apply(clean_and_lemmatize)

print("Applying TF-IDF vectorization...")
# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit to top 5000 features
    min_df=5,  # Ignore terms that appear in fewer than 5 documents
    max_df=0.8,  # Ignore terms that appear in more than 80% of documents
    ngram_range=(1, 2),  # Use unigrams and bigrams
    stop_words="english",
)

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])

print("Saving TF-IDF vectorizer and matrix...")
# Save the vectorizer for later use
with open("outputs/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

# Save the TF-IDF matrix
with open("outputs/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

print("Feature engineering complete!")
