import pickle
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score

with open("outputs/svd_features.pkl", "rb") as f:
    features = pickle.load(f)
with open("outputs/cluster_labels.pkl", "rb") as f:
    labels = pickle.load(f)

print("=" * 80)
print("CLUSTERING EVALUATION")
print("=" * 80)

# Silhouette score (sampled)
sample_idx = np.random.choice(len(features), min(10000, len(features)), replace=False)
sil = silhouette_score(features[sample_idx], labels[sample_idx])
print(f"\nSilhouette Score (sample of {len(sample_idx)}): {sil:.4f}")
print("  Range: [-1, 1] | Higher is better | >0.5 is good")

# Calinski-Harabasz index
ch = calinski_harabasz_score(features, labels)
print(f"\nCalinski-Harabasz Index: {ch:.2f}")
print("  Higher is better | No fixed range")

# Cluster distribution
print("\n" + "-" * 80)
print("Cluster Size Distribution:")
print("-" * 80)
unique, counts = np.unique(labels, return_counts=True)
for cid, cnt in zip(unique, counts):
    print(f"Cluster {cid}: {cnt:,} headlines ({cnt / len(labels) * 100:.2f}%)")

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
