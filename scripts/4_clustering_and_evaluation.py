import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
import logging
import builtins

# Logging allows to print everything to a log file
logging.basicConfig(
    filename="outputs/logs/clustering.log",
    level=logging.INFO,
    format="%(message)s",
    filemode="w"
)

logger= logging.getLogger()
builtins.print = logger.info

def print_section(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")

# STEP 0: LOAD AND NORMALIZE SVD FEATURES
# ----------------------------------------
# Normalize SVD features so KMeans behaves like cosine clustering.
# Without normalization, KMeans is dominated by vector magnitude,
# causing huge "garbage clusters" (one had 52% of all points). 
# Normalizing forces clustering based on direction (topic similarity),
# which leads to more balanced and meaningful clusters.
# ----------------------------------------
# After doing so, there is still 2 clusters with >20% of data, but it's much better.


with open("outputs/models/svd_features.pkl", "rb") as f:
    svd_features = pickle.load(f)
svd_features_norm = normalize(svd_features)

# STEP 1: SERCHING FOR K

print_section("SEARCHING FOR GOOD k")

rng = np.random.RandomState(42)
sample_idx = rng.choice(len(svd_features_norm), min(20000,len(svd_features_norm)), replace=False)
X_sample = svd_features_norm[sample_idx]

k_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
results = []

print("k  | silhouette |    CH       | max_cluster_share")
print("----+-----------+------------+-------------------")

for k in k_values:
    kmeans_tmp = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=2000,
        n_init=20,
        max_iter=200
    )
    labels_tmp = kmeans_tmp.fit_predict(svd_features_norm)

    sil = silhouette_score(X_sample, labels_tmp[sample_idx])
    ch = calinski_harabasz_score(svd_features_norm, labels_tmp)

    unique, counts = np.unique(labels_tmp, return_counts=True)
    max_share = counts.max() / len(labels_tmp)

    print(f"{k:2d} |   {sil: .4f}  | {ch:10.2f} | {max_share:7.3f}")
    results.append((k, sil, ch, max_share))

# Pick best k: maximize silhouette, but penalize if one cluster is > 40% of data
filtered = [r for r in results if r[3] < 0.4]
if filtered:
    best_k = max(filtered, key=lambda r: r[1])[0]
else:
    best_k = max(results, key=lambda r: r[1])[0]

print(f"\nChosen k = {best_k} based on silhouette and cluster balance.")

# STEP 2: FINAL CLUSTERING
print_section("CLUSTERING")

kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=2000, n_init=20, max_iter=300)
clusters = kmeans.fit_predict(svd_features_norm)

for name, obj in [("kmeans_model", kmeans), ("cluster_labels", clusters)]:
    with open(f"outputs/models/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

print("Clustering complete!")

# STEP 3: TOPIC INTERPRETATION
print_section("TOPIC INTERPRETATION")

with open("outputs/models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("outputs/models/svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)

df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
df["cluster"] = clusters
feature_names = vectorizer.get_feature_names_out()

# You did clustering on SVD features and interpretation with this smaller 200 features, 
# but not original 10000 on tfidf. So we need to get the cluster centers in TF-IDF space:
# we need to project the cluster centers back using the SVD components.

centers_tfidf = kmeans.cluster_centers_.dot(svd_model.components_)

topics = {}
for i in range(kmeans.n_clusters):
    top_ind = centers_tfidf[i].argsort()[-15:][::-1]
    top_terms = [feature_names[idx] for idx in top_ind]

    topics[i] = top_terms

# Show all clusters
for i in range(kmeans.n_clusters):
    cluster_df = df[df["cluster"] == i]
    
    print(f"""\nCluster {i}:
          \nTop terms: {', '.join(top_terms)}\n
          \nSample headlines:""")
    
    for j, h in enumerate(cluster_df[df.columns[1]].head(5), 1):
        print(f"  {j}. {h}")
    print(f"\nCluster size: {len(cluster_df)} headlines")

with open("outputs/models/cluster_topics.pkl", "wb") as f:
    pickle.dump(topics, f)

print(f"\n✓ Identified {kmeans.n_clusters} topics")

# STEP 4: CLUSTERING EVALUATION
print_section("CLUSTERING EVALUATION")
sil = silhouette_score(X_sample, clusters[sample_idx])
ch = calinski_harabasz_score(svd_features_norm, clusters)

print(f"\nSilhouette Score (sample of {len(X_sample)}): {sil:.4f}")
print("  Range: [-1, 1] | Higher is better | >0.5 is good")
print(f"\nCalinski-Harabasz Index: {ch:.2f}")
print("  Higher is better | No fixed range")

print(f"\n{'-' * 80}\nCluster Size Distribution:\n{'-' * 80}")
unique, counts = np.unique(clusters, return_counts=True)
metrics = {
    "best_k": best_k,
    "silhouette_score": sil,
    "calinski_harabasz_score": ch,
    "davies_bouldin_score": db,
    "n_clusters": best_k,
    "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
}
with open("outputs/models/evaluation_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)
