import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")
# STEP 1: PREPARE DATA FOR LDA
print_section("PREPARING DATA FOR LDA")
df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")

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
    "act",
]
count_vectorizer = CountVectorizer(
    max_features=3000,
    max_df=0.90,
    min_df=15,
    stop_words=list(ENGLISH_STOP_WORDS.union(custom_stop_words)),
    ngram_range=(1, 2),
    token_pattern=r"\b[a-z]{3,}\b",
)
tf_matrix = count_vectorizer.fit_transform(df[df.columns[1]])

for name, obj in [("count_vectorizer", count_vectorizer), ("tf_matrix", tf_matrix)]:
    with open(f"outputs/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
print(f"Term-frequency matrix shape: {tf_matrix.shape}\nData preparation complete!")

# STEP 2: LDA TOPIC MODELING
print_section("LDA TOPIC MODELING")
n_topics = 30
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=15,
    learning_method="online",
    learning_decay=0.7,
    learning_offset=10.0,
    batch_size=2048,
    n_jobs=-1,
    verbose=1,
    evaluate_every=5,
    perp_tol=0.1,
    doc_topic_prior=0.1,
    topic_word_prior=0.01,
)

print(f"\nFitting LDA model with {n_topics} topics...")
doc_topic_dist = lda.fit_transform(tf_matrix)
dominant_topics = doc_topic_dist.argmax(axis=1)

for name, obj in [
    ("lda_model", lda),
    ("doc_topic_distribution", doc_topic_dist),
    ("lda_topic_labels", dominant_topics),
]:
    with open(f"outputs/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
print("LDA modeling complete!")

# STEP 3: TOPIC INTERPRETATION
print_section("TOPIC INTERPRETATION")
df["topic"] = dominant_topics
df["topic_weight"] = doc_topic_dist.max(axis=1)
feature_names = count_vectorizer.get_feature_names_out()
topics, n_top_words = {}, 20

for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-n_top_words:][::-1]
    top_words, top_scores = feature_names[top_indices], topic[top_indices]
    topics[topic_idx] = {"words": top_words.tolist(), "scores": top_scores.tolist()}

    topic_df = df[df["topic"] == topic_idx].sort_values("topic_weight", ascending=False)
    print(f"\nTopic {topic_idx}:")
    print(
        f"Top words: {', '.join(f'{w}({s:.3f})' for w, s in zip(top_words[:10], top_scores[:10]))}"
    )
    print("\nSample headlines (highest topic weight):")
    for j, (_, row) in enumerate(topic_df.head(5).iterrows(), 1):
        print(f"  {j}. [{row['topic_weight']:.3f}] {row[df.columns[1]]}")
    print(
        f"\nTopic size: {len(topic_df)} headlines ({len(topic_df) / len(df) * 100:.2f}%)"
    )

with open("outputs/lda_topics.pkl", "wb") as f:
    pickle.dump(topics, f)
print_section("Topic interpretation complete!")

# STEP 4: TOPIC EVALUATION
print_section("TOPIC EVALUATION")
perplexity = lda.perplexity(tf_matrix)
log_likelihood = lda.score(tf_matrix)

print(f"\nPerplexity: {perplexity:.2f}")
print("  Lower is better | Measures model fit on training data (not held-out data)")
print(f"\nLog-likelihood: {log_likelihood:.2f}")
print("  Higher is better | Measures model fit")

print(f"\n{'-' * 80}\nTopic Assignment Statistics:\n{'-' * 80}")
for stat, val in [
    ("Average", df["topic_weight"].mean()),
    ("Median", df["topic_weight"].median()),
    ("Min", df["topic_weight"].min()),
    ("Max", df["topic_weight"].max()),
]:
    print(f"{stat} topic weight: {val:.4f}")

print(f"\n{'-' * 80}\nTopic Size Distribution:\n{'-' * 80}")
unique, counts = np.unique(dominant_topics, return_counts=True)
for topic_id, cnt in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
    print(
        f"Topic {topic_id}: {cnt:,} headlines ({cnt / len(dominant_topics) * 100:.2f}%)"
    )

print(f"\n{'-' * 80}\nTopic Diversity:\n{'-' * 80}")
unique_words = set().union(*[set(topics[i]["words"][:10]) for i in range(n_topics)])
diversity_score = len(unique_words) / (n_topics * 10)
print(f"Topic diversity score: {diversity_score:.3f}")
print("  1.0 = no overlap, lower = more duplicate words across topics")

print(f"\n{'-' * 80}\nTopic Coherence (Qualitative):\n{'-' * 80}")
print("Review the top words for each topic above.\nGood topics should have:")
print("  - Semantically related words\n  - Clear, interpretable themes")
print("  - Minimal overlap with other topics\n  - High word scores for top terms")

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

df.to_csv("outputs/abcnews-with-lda-topics.csv", index=False)
print("\nSaved topic-labeled dataset to outputs/abcnews-with-lda-topics.csv")
print_section("LDA topic evaluation complete!")
