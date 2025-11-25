import pickle
import pandas as pd

with open("outputs/kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)
with open("outputs/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("outputs/cluster_labels.pkl", "rb") as f:
    labels = pickle.load(f)

df = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")
df["cluster"] = labels
feature_names = vectorizer.get_feature_names_out()

print("=" * 80)
print("TOPIC INTERPRETATION")
print("=" * 80)

topics = {}
for i in range(kmeans.n_clusters):
    top_terms = [
        feature_names[idx] for idx in kmeans.cluster_centers_[i].argsort()[-15:][::-1]
    ]
    topics[i] = top_terms
    print(f"\nCluster {i}:")
    print(f"Top terms: {', '.join(top_terms)}")
    print("\nSample headlines:")
    for j, h in enumerate(df[df["cluster"] == i][df.columns[1]].head(5), 1):
        print(f"  {j}. {h}")
    print(f"\nCluster size: {(labels == i).sum()} headlines")

with open("outputs/cluster_topics.pkl", "wb") as f:
    pickle.dump(topics, f)
print("\n" + "=" * 80)
print("Topic interpretation complete!")
print("=" * 80)
