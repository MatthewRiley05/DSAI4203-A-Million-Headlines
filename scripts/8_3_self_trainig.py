import pickle
import pandas as pd
import numpy as np
import logging
import builtins
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack as sp_vstack

# Logging allows to print everything to a log file
logging.basicConfig(
    filename="outputs/logs/8_3_iterative_self_training.log",
    level=logging.INFO,
    format="%(message)s",
    filemode="w"
)
logger = logging.getLogger()
builtins.print = logger.info

# ----------------------------------------------------
# Load labeled + unlabeled TF-IDF matrices
# ----------------------------------------------------
with open("outputs/models/semisupervised/tfidf_matrix_lab.pkl", "rb") as f:
    X_lab = pickle.load(f)

with open("outputs/models/semisupervised/tfidf_matrix_orig.pkl", "rb") as f:
    X_unlab = pickle.load(f)

df_lab = pd.read_csv("dataset/merged_news_labels.csv")
df_unlab = pd.read_csv("outputs/abcnews-date-text-preprocessed.csv")

y_lab = df_lab["real"].astype(int).values

print(f"Initial labeled: {X_lab.shape}")
print(f"Initial unlabeled: {X_unlab.shape}")

# ----------------------------------------------------
# Teacher Model
# ----------------------------------------------------
model = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)

# ----------------------------------------------------
# Iteration thresholds
# ----------------------------------------------------
thr = 0.98
min_thr = 0.9
step= 0.02

i =1 
while thr>=min_thr - 1e-9:
    print(f"\n=== Iteration {i} | Threshold {thr} ===")

    # Train on current labeled set
    model.fit(X_lab, y_lab)

    # Predict probabilities on unlabeled
    probs = model.predict_proba(X_unlab)[:, 1]

    # High confidence mask
    mask = (probs >= thr) | (probs <= 1 - thr)
    num_added = mask.sum()
    print(f"High-confidence selected: {num_added}")

    if num_added == 0:
        thr = thr - step
        thr = round(thr,4)
        continue

    # Extract confident samples
    X_new = X_unlab[mask]
    y_new = np.where(probs[mask] >= 0.5, 1, 0)

    # Add to labeled set
    X_lab = sp_vstack([X_lab, X_new])
    y_lab = np.concatenate([y_lab, y_new])

    # Remove added samples from unlabeled pool
    df_unlab = df_unlab[~mask].reset_index(drop=True)
    X_unlab = X_unlab[~mask]
    
    i+=1
    print(f"New labeled size: {X_lab.shape}")
    print(f"Remaining unlabeled: {X_unlab.shape}")

# ----------------------------------------------------
# Save results
# ----------------------------------------------------
with open("outputs/models/semisupervised/self_train_final_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("outputs/models/semisupervised/self_train_X_lab.pkl", "wb") as f:
    pickle.dump(X_lab, f)

with open("outputs/models/semisupervised/self_train_y_lab.pkl", "wb") as f:
    pickle.dump(y_lab, f)

print("Self-training complete.")
print(f"Final training set size: {len(y_lab)}")