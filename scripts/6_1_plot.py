import matplotlib.pyplot as plt
import os

os.makedirs("outputs/pictures", exist_ok=True)

# Format: (latent_dim, epochs, silhouette, max_share)
# Result from manual experiment document Experiment.txt
results = [
    (10, 10, 0.2352, 0.188),
    (10, 20, 0.2365, 0.218),
    (10, 30, 0.2326, 0.255),

    (32, 10, 0.1851, 0.300),
    (32, 20, 0.2804, 0.372),
    (32, 30, 0.2247, 0.345),

    (50, 10, 0.2197, 0.342),
    (50, 20, 0.2487, 0.366),
    (50, 30, 0.2071, 0.323),
]

dims = sorted(list(set(r[0] for r in results)))

plt.figure(figsize=(10,6))

for d in dims:
    subset = [r for r in results if r[0] == d]
    epochs = [r[1] for r in subset]
    sils = [r[2] for r in subset]
    plt.plot(epochs, sils, marker='o', label=f"latent_dim={d}")

plt.xlabel("Epochs")
plt.ylabel("Silhouette Score")
plt.title("AE Latent Dimension vs Silhouette Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/pictures/ae_clust1.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10,6))

for d in dims:
    subset = [r for r in results if r[0] == d]
    epochs = [r[1] for r in subset]
    max_shares = [r[3] for r in subset]
    plt.plot(epochs, max_shares, marker='o', label=f"latent_dim={d}")

plt.xlabel("Epochs")
plt.ylabel("Max Cluster Share")
plt.title("Cluster Imbalance Across AE Settings")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/pictures/ae_clust2.png", dpi=200, bbox_inches="tight")
plt.show()

"""AE generally has better clustering results than SVD, both in terms of silhouette and max_share scores
        (silhouette rose significantly, max_share reduced significantly with Nx10, in all other cases
        there were only 1 big cluster, while SVD had at least 2)
    Less dimensions, better results
    vertraining on more epochs reduces performance
    Autoencoder embeddings significantly outperform SVD embeddings.
    Optimal latent dimension is 32.
    Optimal number of epochs is around 20.
    Larger latent sizes (50+) introduce noise and reduce structure.
    Very small latent sizes (10) produce balanced clusters but slightly worse silhouette.
    Autoencoders reduce cluster imbalance dramatically (no oversized 50% clusters).
    AE representations are more suitable for KMeans and topic interpretation.
"""