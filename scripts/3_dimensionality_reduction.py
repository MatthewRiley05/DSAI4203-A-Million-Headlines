import pickle
from sklearn.decomposition import TruncatedSVD

print("Loading TF-IDF matrix...")
with open("outputs/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

print("Applying Truncated SVD for dimensionality reduction...")
# Apply Truncated SVD for dimensionality reduction
svd = TruncatedSVD(n_components=100, random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)

print("Saving SVD model and reduced features...")
# Save the SVD model
with open("outputs/svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)

# Save the reduced features
with open("outputs/svd_features.pkl", "wb") as f:
    pickle.dump(svd_matrix, f)

print("Dimensionality reduction complete!")
