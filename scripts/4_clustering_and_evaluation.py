import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


# STEP 1: CLUSTERING
with open("outputs/svd_features.pkl", "rb") as f:
    svd_features = pickle.load(f)

# Normalize features for better clustering
svd_features_norm = normalize(svd_features, norm="l2")

# Find optimal number of clusters
n_clusters_options = [20, 25, 30, 35, 40]
best_score = float("inf")
best_k = 30
best_kmeans = None
clusters = None
for k in n_clusters_options:
    kmeans_temp = MiniBatchKMeans(
        n_clusters=k, random_state=42, batch_size=2000, n_init=10, max_iter=300, init="k-means++"
    )
    labels_temp = kmeans_temp.fit_predict(svd_features_norm)
    db_score = davies_bouldin_score(svd_features_norm, labels_temp)
    if db_score < best_score:
        best_score = db_score
        best_k = k
        best_kmeans = kmeans_temp
        clusters = labels_temp

# Use the best model and labels found during optimization
kmeans = best_kmeans

# clusters already set to the best labels


# clusters = kmeans.fit_predict(svd_features_norm)  # Removed redundant re-fit
for name, obj in [("kmeans_model", kmeans), ("cluster_labels", clusters)]:
    with open(f"outputs/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

print(f"✓ Clustered into {best_k} topics (DB score: {best_score:.4f})")

# STEP 2: TOPIC INTERPRETATION
with open("outputs/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the cleaned dataframe (after duplicates removed)
df = pd.read_csv("outputs/abcnews-cleaned.csv")
df["cluster"] = clusters
feature_names = vectorizer.get_feature_names_out()

topics = {}
for i in range(kmeans.n_clusters):
    top_terms = [
        feature_names[idx] for idx in kmeans.cluster_centers_[i].argsort()[-10:][::-1]
    ]
    topics[i] = top_terms

# Show all clusters
for i in range(kmeans.n_clusters):
    cluster_df = df[df["cluster"] == i]
    print(
        f"\nCluster {i:2d} ({len(cluster_df):>6,} headlines): {', '.join(topics[i][:5])}"
    )
    print(f"  Example: {cluster_df[df.columns[1]].iloc[0]}")

with open("outputs/cluster_topics.pkl", "wb") as f:
    pickle.dump(topics, f)

print(f"\n✓ Identified {kmeans.n_clusters} topics")

# STEP 3: CLUSTERING EVALUATION
np.random.seed(42)
sample_size = min(20000, len(svd_features_norm))
sample_idx = np.random.choice(len(svd_features_norm), sample_size, replace=False)

sil = silhouette_score(svd_features_norm[sample_idx], clusters[sample_idx])
ch = calinski_harabasz_score(svd_features_norm, clusters)
db = davies_bouldin_score(svd_features_norm, clusters)

print(f"\n{'=' * 60}")
print(f"CLUSTERING RESULTS")
print(f"{'=' * 60}")
print(f"Silhouette Score:    {sil:>8.4f}  (higher is better)")
print(f"Calinski-Harabasz:   {ch:>8.2f}  (higher is better)")
print(f"Davies-Bouldin:      {db:>8.4f}  (lower is better)")
print(f"Number of Clusters:  {best_k:>8}")
print(f"{'=' * 60}")

unique, counts = np.unique(clusters, return_counts=True)
metrics = {
    "silhouette_score": sil,
    "calinski_harabasz_score": ch,
    "davies_bouldin_score": db,
    "n_clusters": best_k,
    "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
}
with open("outputs/evaluation_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)
