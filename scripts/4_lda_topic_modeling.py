import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ============================================================================
# STEP 1: PREPARE DATA FOR LDA
# ============================================================================
print("=" * 80)
print("PREPARING DATA FOR LDA")
print("=" * 80)

df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
documents = df[df.columns[1]].tolist()

# LDA works better with term frequency (CountVectorizer) than TF-IDF
print("Creating term-frequency matrix...")
# Add custom stop words for Australian news
custom_stop_words = [
    "said",
    "says",
    "day",
    "year",
    "years",
    "people",
    "time",
    "wa",
    "nsw",
    "qld",
    "vic",
    "sa",
    "nt",
    "tas",
    "act",  # State abbreviations less meaningful
]
count_vectorizer = CountVectorizer(
    max_features=3000,
    max_df=0.90,  # More aggressive - remove very common words
    min_df=15,  # Higher threshold for rare words
    stop_words=list(
        set(
            list(
                __import__(
                    "sklearn.feature_extraction.text", fromlist=["ENGLISH_STOP_WORDS"]
                ).ENGLISH_STOP_WORDS
            )
            + custom_stop_words
        )
    ),
    ngram_range=(1, 2),  # Include bigrams for better context
    token_pattern=r"\b[a-z]{3,}\b",  # Only words 3+ chars
)
tf_matrix = count_vectorizer.fit_transform(documents)

with open("outputs/count_vectorizer.pkl", "wb") as f:
    pickle.dump(count_vectorizer, f)
with open("outputs/tf_matrix.pkl", "wb") as f:
    pickle.dump(tf_matrix, f)

print(f"Term-frequency matrix shape: {tf_matrix.shape}")
print("Data preparation complete!")

# ============================================================================
# STEP 2: LDA TOPIC MODELING
# ============================================================================
print("\n" + "=" * 80)
print("LDA TOPIC MODELING")
print("=" * 80)

n_topics = 30
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=15,  # Slightly more iterations for better convergence
    learning_method="online",
    learning_decay=0.7,
    learning_offset=10.0,
    batch_size=2048,
    n_jobs=-1,
    verbose=1,
    evaluate_every=5,
    perp_tol=0.1,
    doc_topic_prior=0.1,  # Alpha: lower = sparser topic distribution per doc
    topic_word_prior=0.01,  # Beta: lower = sparser word distribution per topic
)

print(f"\nFitting LDA model with {n_topics} topics...")
doc_topic_dist = lda.fit_transform(tf_matrix)

# Save model and results
with open("outputs/lda_model.pkl", "wb") as f:
    pickle.dump(lda, f)
with open("outputs/doc_topic_distribution.pkl", "wb") as f:
    pickle.dump(doc_topic_dist, f)

# Assign each document to its dominant topic
dominant_topics = doc_topic_dist.argmax(axis=1)
with open("outputs/lda_topic_labels.pkl", "wb") as f:
    pickle.dump(dominant_topics, f)

print("LDA modeling complete!")

# ============================================================================
# STEP 3: TOPIC INTERPRETATION
# ============================================================================
print("\n" + "=" * 80)
print("TOPIC INTERPRETATION")
print("=" * 80)

df["topic"] = dominant_topics
df["topic_weight"] = doc_topic_dist.max(axis=1)
feature_names = count_vectorizer.get_feature_names_out()

topics = {}
n_top_words = 20  # Show more words for better interpretation

for topic_idx, topic in enumerate(lda.components_):
    top_word_indices = topic.argsort()[-n_top_words:][::-1]
    top_words = [feature_names[i] for i in top_word_indices]
    top_word_scores = [topic[i] for i in top_word_indices]

    # Store both words and scores
    topics[topic_idx] = {"words": top_words, "scores": top_word_scores}

    print(f"\nTopic {topic_idx}:")
    # Show top 10 words with scores for interpretation
    top_10_with_scores = [
        f"{word}({score:.3f})"
        for word, score in zip(top_words[:10], top_word_scores[:10])
    ]
    print(f"Top words: {', '.join(top_10_with_scores)}")

    # Show sample headlines with high topic weight
    topic_df = df[df["topic"] == topic_idx].sort_values("topic_weight", ascending=False)
    print("\nSample headlines (highest topic weight):")
    for j, (_, row) in enumerate(topic_df.head(5).iterrows(), 1):
        headline = row[df.columns[1]]
        weight = row["topic_weight"]
        print(f"  {j}. [{weight:.3f}] {headline}")

    print(
        f"\nTopic size: {len(topic_df)} headlines ({len(topic_df) / len(df) * 100:.2f}%)"
    )

with open("outputs/lda_topics.pkl", "wb") as f:
    pickle.dump(topics, f)

print("\n" + "=" * 80)
print("Topic interpretation complete!")
print("=" * 80)

# ============================================================================
# STEP 4: TOPIC EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("TOPIC EVALUATION")
print("=" * 80)

# Perplexity (lower is better)
perplexity = lda.perplexity(tf_matrix)
print(f"\nPerplexity: {perplexity:.2f}")
print("  Lower is better | Measures how well model predicts held-out data")

# Log-likelihood (higher is better)
log_likelihood = lda.score(tf_matrix)
print(f"\nLog-likelihood: {log_likelihood:.2f}")
print("  Higher is better | Measures model fit")

# Topic distribution statistics
print("\n" + "-" * 80)
print("Topic Assignment Statistics:")
print("-" * 80)
print(f"Average topic weight: {df['topic_weight'].mean():.4f}")
print(f"Median topic weight: {df['topic_weight'].median():.4f}")
print(f"Min topic weight: {df['topic_weight'].min():.4f}")
print(f"Max topic weight: {df['topic_weight'].max():.4f}")

# Topic size distribution
print("\n" + "-" * 80)
print("Topic Size Distribution:")
print("-" * 80)
unique, counts = np.unique(dominant_topics, return_counts=True)
for topic_id, cnt in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
    print(
        f"Topic {topic_id}: {cnt:,} headlines ({cnt / len(dominant_topics) * 100:.2f}%)"
    )

# Topic diversity - measure overlap between topics
print("\n" + "-" * 80)
print("Topic Diversity:")
print("-" * 80)
top_words_per_topic = [
    set([w for w in topics[i]["words"][:10]]) for i in range(n_topics)
]
unique_words = set()
for words in top_words_per_topic:
    unique_words.update(words)
diversity_score = len(unique_words) / (n_topics * 10)
print(f"Topic diversity score: {diversity_score:.3f}")
print(f"  1.0 = no overlap, lower = more duplicate words across topics")

# Topic coherence - simple approach using top word co-occurrence
print("\n" + "-" * 80)
print("Topic Coherence (Qualitative):")
print("-" * 80)
print("Review the top words for each topic above.")
print("Good topics should have:")
print("  - Semantically related words")
print("  - Clear, interpretable themes")
print("  - Minimal overlap with other topics")
print("  - High word scores for top terms")

# Save evaluation metrics
metrics = {
    "perplexity": perplexity,
    "log_likelihood": log_likelihood,
    "avg_topic_weight": df["topic_weight"].mean(),
    "median_topic_weight": df["topic_weight"].median(),
    "topic_sizes": dict(zip(unique.tolist(), counts.tolist())),
    "diversity_score": diversity_score,
    "n_unique_top_words": len(unique_words),
}
with open("outputs/lda_evaluation_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

# Save topic-labeled dataset
df.to_csv("outputs/abcnews-with-lda-topics.csv", index=False)
print(f"\nSaved topic-labeled dataset to outputs/abcnews-with-lda-topics.csv")

print("\n" + "=" * 80)
print("LDA topic evaluation complete!")
print("=" * 80)
