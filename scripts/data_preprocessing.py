from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

stemmer = SnowballStemmer("english")
stop_words = ENGLISH_STOP_WORDS

def stem_analyzer(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)            # remove digits
    text = re.sub(r"[^a-z\s]", " ", text)       # remove punctuation and symbols
    text = re.sub(r"\s+", " ", text)          # normalize whitespace
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]

    # remove short tokens consisting of less than 3 characters
    tokens = [t for t in tokens if len(t) >= 3]

    # generate 2-grams and 3-grams
    bigrams = [tokens[i] + " " + tokens[i+1] for i in range(len(tokens)-1)]
    trigrams = [tokens[i] + " " + tokens[i+1] + " " + tokens[i+2] for i in range(len(tokens) - 2)]
    
    return tokens + bigrams + trigrams