import pickle
from sklearn.cluster import MiniBatchKMeans

# K-Means clustering configuration
# Number of clusters: chosen based on expected number of distinct groups in the data.
N_CLUSTERS = 10
# Random state for reproducibility.
RANDOM_STATE = 42
# Batch size for MiniBatchKMeans: balances speed and memory usage.
BATCH_SIZE = 1000

# Load SVD features
with open("outputs/svd_features.pkl", "rb") as f:
    svd_features = pickle.load(f)

# Apply MiniBatch K-Means clustering
kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, batch_size=BATCH_SIZE
)
clusters = kmeans.fit_predict(svd_features)

# Save the model
with open("outputs/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# Save cluster labels
with open("outputs/cluster_labels.pkl", "wb") as f:
    pickle.dump(clusters, f)

print("Clustering complete!")
