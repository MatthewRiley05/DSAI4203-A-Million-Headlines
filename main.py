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
print("\n>>> Running LDA Topic Modeling <<<")
subprocess.run([sys.executable, "scripts/4_lda_topic_modeling.py"], check=True)

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
