import pickle
import pandas as pd
import numpy as np

# Load required data
with open("outputs/kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("outputs/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("outputs/cluster_labels.pkl", "rb") as f:
    cluster_labels = pickle.load(f)

# Load preprocessed data for sample headlines
df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
df["cluster"] = cluster_labels

# Get feature names from TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get cluster centroids
centroids = kmeans.cluster_centers_

# Find top terms for each cluster
n_top_terms = 15
cluster_topics = {}

print("=" * 80)
print("TOPIC INTERPRETATION")
print("=" * 80)

for cluster_id in range(kmeans.n_clusters):
    # Get top term indices for this cluster
    centroid = centroids[cluster_id]
    top_indices = centroid.argsort()[-n_top_terms:][::-1]
    top_terms = [feature_names[i] for i in top_indices]

    cluster_topics[cluster_id] = top_terms

    print(f"\nCluster {cluster_id}:")
    print(f"Top terms: {', '.join(top_terms)}")

    # Show sample headlines from this cluster
    cluster_headlines = df[df["cluster"] == cluster_id][df.columns[1]].head(5)
    print(f"\nSample headlines:")
    for idx, headline in enumerate(cluster_headlines, 1):
        print(f"  {idx}. {headline}")

    print(f"\nCluster size: {(cluster_labels == cluster_id).sum()} headlines")

# Save cluster topics
with open("outputs/cluster_topics.pkl", "wb") as f:
    pickle.dump(cluster_topics, f)

print("\n" + "=" * 80)
print("Topic interpretation complete!")
print("=" * 80)
