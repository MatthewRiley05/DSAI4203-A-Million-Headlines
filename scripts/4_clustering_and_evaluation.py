import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# ============================================================================
# STEP 1: CLUSTERING
# ============================================================================
print("=" * 80)
print("CLUSTERING")
print("=" * 80)

with open("outputs/svd_features.pkl", "rb") as f:
    svd_features = pickle.load(f)

kmeans = MiniBatchKMeans(n_clusters=30, random_state=42, batch_size=1000)
clusters = kmeans.fit_predict(svd_features)

with open("outputs/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)
with open("outputs/cluster_labels.pkl", "wb") as f:
    pickle.dump(clusters, f)
print("Clustering complete!")

# ============================================================================
# STEP 2: TOPIC INTERPRETATION
# ============================================================================
print("\n" + "=" * 80)
print("TOPIC INTERPRETATION")
print("=" * 80)

with open("outputs/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
df["cluster"] = clusters
feature_names = vectorizer.get_feature_names_out()

topics = {}
for i in range(kmeans.n_clusters):
    top_terms = [
        feature_names[idx] for idx in kmeans.cluster_centers_[i].argsort()[-15:][::-1]
    ]
    topics[i] = top_terms
    print(f"\nCluster {i}:")
    print(f"Top terms: {', '.join(top_terms)}")
    print("\nSample headlines:")
    cluster_df = df[df["cluster"] == i]
    for j, h in enumerate(cluster_df[df.columns[1]].head(5), 1):
        print(f"  {j}. {h}")
    print(f"\nCluster size: {len(cluster_df)} headlines")

with open("outputs/cluster_topics.pkl", "wb") as f:
    pickle.dump(topics, f)
print("\n" + "=" * 80)
print("Topic interpretation complete!")
print("=" * 80)

# ============================================================================
# STEP 3: CLUSTERING EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("CLUSTERING EVALUATION")
print("=" * 80)

# Silhouette score (sampled)
np.random.seed(42)
sample_idx = np.random.choice(
    len(svd_features), min(10000, len(svd_features)), replace=False
)
sil = silhouette_score(svd_features[sample_idx], clusters[sample_idx])
print(f"\nSilhouette Score (sample of {len(sample_idx)}): {sil:.4f}")
print("  Range: [-1, 1] | Higher is better | >0.5 is good")

# Calinski-Harabasz index
ch = calinski_harabasz_score(svd_features, clusters)
print(f"\nCalinski-Harabasz Index: {ch:.2f}")
print("  Higher is better | No fixed range")

# Cluster distribution
print("\n" + "-" * 80)
print("Cluster Size Distribution:")
print("-" * 80)
unique, counts = np.unique(clusters, return_counts=True)
for cid, cnt in zip(unique, counts):
    print(f"Cluster {cid}: {cnt:,} headlines ({cnt / len(clusters) * 100:.2f}%)")

# Save metrics
metrics = {
    "silhouette_score": sil,
    "calinski_harabasz_score": ch,
    "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
}
with open("outputs/evaluation_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("\n" + "=" * 80)
print("Clustering evaluation complete!")
print("=" * 80)
