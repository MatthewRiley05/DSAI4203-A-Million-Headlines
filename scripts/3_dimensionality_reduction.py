import pickle
from sklearn.decomposition import TruncatedSVD

with open("outputs/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

# Optimize n_components based on variance explained
n_components = 300  # Increased from 200
svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=10)
svd_matrix = svd.fit_transform(tfidf_matrix)

for name, obj in [("svd_model", svd), ("svd_features", svd_matrix)]:
    with open(f"outputs/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

variance = svd.explained_variance_ratio_.sum()
print(f"✓ Reduced to {n_components} components ({variance:.1%} variance)")
