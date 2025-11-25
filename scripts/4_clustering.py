import pickle
from sklearn.cluster import MiniBatchKMeans

with open("outputs/svd_features.pkl", "rb") as f:
    svd_features = pickle.load(f)

kmeans = MiniBatchKMeans(n_clusters=30, random_state=42, batch_size=1000)
clusters = kmeans.fit_predict(svd_features)

with open("outputs/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)
with open("outputs/cluster_labels.pkl", "wb") as f:
    pickle.dump(clusters, f)
print("Clustering complete!")
