import subprocess
import sys

# Run preprocessing and feature engineering
subprocess.run([sys.executable, "scripts/1_data_cleaning.py"], check=True)
subprocess.run([sys.executable, "scripts/2_feature_engineering.py"], check=True)
subprocess.run([sys.executable, "scripts/3_dimensionality_reduction.py"], check=True)

# Choose one of the following topic modeling approaches:

# Option A: K-Means Clustering
# subprocess.run([sys.executable, "scripts/4_clustering_and_evaluation.py"], check=True)

# Option B: LDA Topic Modeling (uncomment to use)
subprocess.run([sys.executable, "scripts/4_lda_topic_modeling.py"], check=True)
