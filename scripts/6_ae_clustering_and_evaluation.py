import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
import logging
import builtins

# Logging allows to print everything to a log file
logging.basicConfig(
    filename="outputs/logs/ae_clustering.log",
    level=logging.INFO,
    format="%(message)s",
    filemode="w"
)

logger= logging.getLogger()
builtins.print = logger.info


def print_section(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n")


# ------------------------------------------------------------------
# STEP 1: LOAD AE FEATURES AND NORMALIZE
# ------------------------------------------------------------------
print_section("LOADING AE FEATURES")

with open("outputs/models/ae_features.pkl", "rb") as f:
    Z = pickle.load(f)           # (N, latent_dim)

Z = Z.astype("float32")
Z_norm = normalize(Z)           # L2-normalize → cosine-like

n_samples, latent_dim = Z_norm.shape
print(f"AE latent features shape: {Z_norm.shape}")

# ------------------------------------------------------------------
# STEP 2: K-SEARCH (ON FULL DATA, SILHOUETTE ON SAMPLE)
# ------------------------------------------------------------------
print_section("K-SEARCH FOR CLUSTERING")

rng = np.random.RandomState(42)
sample_size = min(10_000, n_samples)
sample_idx = rng.choice(n_samples, sample_size, replace=False)
Z_sample = Z_norm[sample_idx]

k_values = [20, 30, 40, 50]
results = []

print("k  | silhouette |    CH       | max_cluster_share")
print("----+-----------+------------+-------------------")

for k in k_values:
    kmeans_tmp = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=2000,
        n_init=20,
        max_iter=200,
    )
    labels_tmp = kmeans_tmp.fit_predict(Z_norm)

    sil = silhouette_score(Z_sample, labels_tmp[sample_idx])
    ch = calinski_harabasz_score(Z_norm, labels_tmp)

    unique, counts = np.unique(labels_tmp, return_counts=True)
    max_share = counts.max() / len(labels_tmp)

    print(f"{k:2d} |   {sil: .4f}  | {ch:10.2f} | {max_share:7.3f}")
    results.append((k, sil, ch, max_share))

# choose k: highest silhouette but avoid clusters >40% of data
filtered = [r for r in results if r[3] < 0.3]
if filtered:
    best_k = max(filtered, key=lambda r: r[1])[0]
else:
    best_k = max(results, key=lambda r: r[1])[0]

print(f"\nChosen k = {best_k} based on silhouette and cluster balance.")

# ------------------------------------------------------------------
# STEP 3: FINAL CLUSTERING WITH BEST K
# ------------------------------------------------------------------
print_section("FINAL CLUSTERING")

kmeans = MiniBatchKMeans(
    n_clusters=best_k,
    random_state=42,
    batch_size=2000,
    n_init=20,
    max_iter=200,
)
clusters = kmeans.fit_predict(Z_norm)

print("Clustering complete!")

with open("outputs/models/ae_kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)
with open("outputs/models/ae_cluster_labels.pkl", "wb") as f:
    pickle.dump(clusters, f)

# ------------------------------------------------------------------
# STEP 4: INTERPRET CLUSTERS USING TF-IDF
# ------------------------------------------------------------------
print_section("LOADING TF-IDF AND HEADLINES")

with open("outputs/models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("outputs/models/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
text_col = df.columns[1]
df["ae_cluster"] = clusters

vocab = vectorizer.get_feature_names_out()

print_section("TOPIC-LIKE INTERPRETATION OF AE CLUSTERS")

cluster_topics = {}
n_top_terms = 15

for cid in range(best_k):
    idx = np.where(clusters == cid)[0]
    cluster_size = len(idx)
    share = cluster_size / n_samples * 100

    cluster_tfidf = tfidf_matrix[idx]                  # sparse
    mean_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).ravel()
    top_idx = mean_tfidf.argsort()[-n_top_terms:][::-1]
    top_terms = [vocab[i] for i in top_idx]

    cluster_topics[cid] = top_terms

    cluster_df = df.iloc[idx]

    print(f"\nCluster {cid}:")
    print(f"Size: {cluster_size} headlines ({share:.2f}%)")
    print(f"Top terms: {', '.join(top_terms)}\n")
    print("Sample headlines:")
    for j, (_, row) in enumerate(cluster_df[[text_col]].head(5).iterrows(), 1):
        print(f"  {j}. {row[text_col]}")

with open("outputs/models/ae_cluster_topics.pkl", "wb") as f:
    pickle.dump(cluster_topics, f)

print_section("AE CLUSTERING + INTERPRETATION COMPLETE")

# STEP 4: CLUSTERING EVALUATION
print_section("CLUSTERING EVALUATION")

# Silhouette on sample, CH on full data
sil = silhouette_score(Z_sample, clusters[sample_idx])
ch = calinski_harabasz_score(Z_norm, clusters)

print(f"\nSilhouette Score (sample of {len(Z_sample)}): {sil:.4f}")
print("  Range: [-1, 1] | Higher is better | >0.5 is good")
print(f"\nCalinski-Harabasz Index: {ch:.2f}")
print("  Higher is better | No fixed range")

print(f"\n{'-' * 80}\nCluster Size Distribution:\n{'-' * 80}")
unique, counts = np.unique(clusters, return_counts=True)
for cid, cnt in zip(unique, counts):
    print(f"Cluster {cid}: {cnt:,} headlines ({cnt / len(clusters) * 100:.2f}%)")

metrics = {
    "best_k": best_k,
    "silhouette_score": sil,
    "calinski_harabasz_score": ch,
    "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
}

with open("outputs/models/ae_evaluation_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print_section("Clustering evaluation complete!")
