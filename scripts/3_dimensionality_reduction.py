import pickle
from sklearn.decomposition import TruncatedSVD

with open("outputs/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

svd = TruncatedSVD(n_components=200, random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)

for name, obj in [("svd_model", svd), ("svd_features", svd_matrix)]:
    with open(f"outputs/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
print("Dimensionality reduction complete!")
