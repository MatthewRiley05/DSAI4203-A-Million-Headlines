import subprocess
import sys

print("=" * 80)
print("NEWS HEADLINE TOPIC MODELING PIPELINE")
print("=" * 80)

# Run preprocessing and feature engineering
print("\n[1/4] Data Cleaning...")
subprocess.run([sys.executable, "scripts/1_data_cleaning.py"], check=True)

print("\n[2/4] Feature Engineering...")
subprocess.run([sys.executable, "scripts/2_feature_engineering.py"], check=True)

print("\n[3/4] Dimensionality Reduction...")
subprocess.run([sys.executable, "scripts/3_dimensionality_reduction.py"], check=True)

# Choose clustering approach:
print("\n[4/4] Topic Modeling...")

# Option A: K-Means Clustering (Optimized - Best Performance)
# print("\n>>> Running Optimized K-Means Clustering <<<")
# subprocess.run([sys.executable, "scripts/4_clustering_and_evaluation.py"], check=True)

# Option B: LDA Topic Modeling (uncomment to use)
# print("\n>>> Running LDA Topic Modeling <<<")
# subprocess.run([sys.executable, "scripts/4_lda_topic_modeling.py"], check=True)


# === AUTOENCODER & CLUSTERING PIPELINE ===
print("\n[5] Autoencoder on SVD Features...")
subprocess.run([sys.executable, "scripts/5_autoencoder_on_svd.py"], check=True)

print("\n[6] Plot Autoencoder Results...")
subprocess.run([sys.executable, "scripts/6_1_plot.py"], check=True)

print("\n[7] AE Clustering and Evaluation...")
subprocess.run(
    [sys.executable, "scripts/6_ae_clustering_and_evaluation.py"], check=True
)

print("\n[8] AE Visualization...")
subprocess.run([sys.executable, "scripts/7_ae_vizualization.py"], check=True)

# === FAKE/REAL NEWS SEMI-SUPERVISED PIPELINE ===
print("\n[9] Prepare Fake/Real News Datasets...")
subprocess.run(
    [sys.executable, "scripts/8_1_fake_real_dataset_preparing.py"], check=True
)

print("\n[10] Feature Engineering for Fake/Real News...")
subprocess.run([sys.executable, "scripts/8_2_feature_engineering.py"], check=True)

print("\n[11] Self-Training for Semi-Supervised Learning...")
subprocess.run([sys.executable, "scripts/8_3_self_trainig.py"], check=True)

print("\n[12] Evaluation of Semi-Supervised Results...")
subprocess.run([sys.executable, "scripts/8_3_evaluation.py"], check=True)

print("\n[13] Final Classification...")
subprocess.run([sys.executable, "scripts/8_4_classifications.py"], check=True)

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
