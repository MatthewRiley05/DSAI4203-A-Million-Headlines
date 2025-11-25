import pickle
from sklearn.decomposition import TruncatedSVD

with open("outputs/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

svd = TruncatedSVD(n_components=200, random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)

with open("outputs/svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)
with open("outputs/svd_features.pkl", "wb") as f:
    pickle.dump(svd_matrix, f)
print("Dimensionality reduction complete!")
