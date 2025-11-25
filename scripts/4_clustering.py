import pickle
from sklearn.cluster import MiniBatchKMeans

# Load SVD features
with open('outputs/svd_features.pkl', 'rb') as f:
    svd_features = pickle.load(f)

# Apply MiniBatch K-Means clustering
kmeans = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=1000)
clusters = kmeans.fit_predict(svd_features)

# Save the model
with open('outputs/kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Save cluster labels
with open('outputs/cluster_labels.pkl', 'wb') as f:
    pickle.dump(clusters, f)
