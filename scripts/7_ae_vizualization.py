import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import logging
import builtins

# Logging allows to print everything to a log file
logging.basicConfig(
    filename="outputs/logs/ae_vizualization.log",
    level=logging.INFO,
    format="%(message)s",
    filemode="w"
)

logger= logging.getLogger()
builtins.print = logger.info


def print_section(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n")


print_section("LOADING AE FEATURES AND CLUSTERS")

with open("outputs/models/ae_features.pkl", "rb") as f:
    Z = pickle.load(f)
with open("outputs/models/ae_cluster_labels.pkl", "rb") as f:
    clusters = pickle.load(f)

Z = Z.astype("float32")
n_samples = Z.shape[0]

# sample for visualization
print_section("SAMPLING FOR VISUALIZATION")

rng = np.random.RandomState(42)
vis_size = min(20_000, n_samples)
vis_idx = rng.choice(n_samples, vis_size, replace=False)

Z_vis = Z[vis_idx]
clusters_vis = np.array(clusters)[vis_idx]

print(f"Visualizing {vis_size} points out of {n_samples}.")

# ------------------------------------------------------------------
# TSNE TO 2D
# ------------------------------------------------------------------
print_section("RUNNING t-SNE (this may take a bit)")

tsne = TSNE(
    n_components=2,
    perplexity=50,
    learning_rate="auto",
    init="random",
    random_state=42,
)

Z_2d = tsne.fit_transform(Z_vis)

# ------------------------------------------------------------------
# PLOT
# ------------------------------------------------------------------
print_section("PLOTTING")

n_clusters = int(clusters_vis.max()) + 1
base_cmap = matplotlib.colormaps.get_cmap("tab20").resampled(20)
colors = base_cmap(np.linspace(0, 1, 20))
repeats = int(np.ceil(n_clusters / 20))
colors = np.tile(colors, (repeats, 1))[:n_clusters]
cmap = matplotlib.colors.ListedColormap(colors)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    Z_2d[:, 0],
    Z_2d[:, 1],
    c=clusters_vis,
    cmap=cmap,
    s=3,
    alpha=0.6,
)
plt.title("t-SNE of AE Latent Space Colored by AE Clusters")
plt.xticks([])
plt.yticks([])
plt.tight_layout()

plt.savefig("outputs/pictures/ae_tsne_clusters.png", dpi=200)
plt.close()

print("Saved visualization to 'outputs/pictures/ae_tsne_clusters.png'.")
print_section("VISUALIZATION COMPLETE")
