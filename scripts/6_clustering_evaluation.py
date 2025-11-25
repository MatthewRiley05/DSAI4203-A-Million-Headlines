import pickle
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Load required data
with open("outputs/svd_features.pkl", "rb") as f:
    svd_features = pickle.load(f)

with open("outputs/cluster_labels.pkl", "rb") as f:
    cluster_labels = pickle.load(f)

print("=" * 80)
print("CLUSTERING EVALUATION")
print("=" * 80)

# Calculate Silhouette Score (sample for speed with large datasets)
print("\nCalculating Silhouette Score...")
sample_size = min(10000, len(svd_features))
sample_indices = np.random.choice(len(svd_features), sample_size, replace=False)
silhouette = silhouette_score(
    svd_features[sample_indices], cluster_labels[sample_indices]
)
print(f"Silhouette Score (sample of {sample_size}): {silhouette:.4f}")
print("  Range: [-1, 1] | Higher is better | >0.5 is good")

# Calculate Calinski-Harabasz Index
print("\nCalculating Calinski-Harabasz Index...")
ch_score = calinski_harabasz_score(svd_features, cluster_labels)
print(f"Calinski-Harabasz Index: {ch_score:.2f}")
print("  Higher is better | No fixed range")

# Cluster size distribution
print("\n" + "-" * 80)
print("Cluster Size Distribution:")
print("-" * 80)
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(cluster_labels)) * 100
    print(f"Cluster {cluster_id}: {count:,} headlines ({percentage:.2f}%)")

# Save evaluation metrics
metrics = {
    "silhouette_score": silhouette,
    "calinski_harabasz_score": ch_score,
    "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
}

with open("outputs/evaluation_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("\n" + "=" * 80)
print("Clustering evaluation complete!")
print("=" * 80)
