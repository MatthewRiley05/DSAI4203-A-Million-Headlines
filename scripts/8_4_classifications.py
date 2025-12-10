import pickle
import logging
import builtins
import numpy as np
from scipy.sparse import vstack as sp_vstack

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Classical models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Logging allows to print everything to a log file
logging.basicConfig(
    filename="outputs/logs/8_4_final_classification.log",
    level=logging.INFO,
    format="%(message)s",
    filemode="w"
)
logger = logging.getLogger()
builtins.print = logger.info

# ----------------------------------------------------
# Load self-trained data
# ----------------------------------------------------
with open("outputs/models/semisupervised/self_train_X_lab.pkl", "rb") as f:
    X_all = pickle.load(f)

with open("outputs/models/semisupervised/self_train_y_lab.pkl", "rb") as f:
    y_all = pickle.load(f)

print(f"Loaded enlarged dataset: {X_all.shape}, labels: {y_all.shape}")

# ----------------------------------------------------
# Separate original vs pseudo-labeled
# ----------------------------------------------------
n_orig = 82417  # FIRST 82,417 rows are true labeled data

idx_orig = np.arange(n_orig)
idx_pseudo = np.arange(n_orig, X_all.shape[0])

y_orig = y_all[idx_orig]

# ----------------------------------------------------
# Validation split (ON TRUE LABELS ONLY)
# ----------------------------------------------------
train_idx_orig, val_idx_orig = train_test_split(
    idx_orig,
    test_size=0.2,
    random_state=42,
    stratify=y_orig,
)

X_train_orig = X_all[train_idx_orig]
y_train_orig = y_all[train_idx_orig]

X_test = X_all[val_idx_orig]
y_val = y_all[val_idx_orig]

# Add ALL pseudo-labels to the training pool
if len(idx_pseudo) > 0:
    X_train = sp_vstack([X_train_orig, X_all[idx_pseudo]])
    y_train = np.concatenate([y_train_orig, y_all[idx_pseudo]])
else:
    X_train = X_train_orig
    y_train = y_train_orig

print(f"Training set: {X_train.shape}, Validation set: {X_test.shape}")

# ----------------------------------------------------
# Evaluation helper
# ----------------------------------------------------
def evaluate_model(name, model):
    print(f"\n=== Training: {name} ===")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="binary", pos_label=1
    )

    print(f"ACC: {acc:.4f} | PREC: {prec:.4f} | REC: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y_val, y_pred))

    return {
        "name": name,
        "model": model,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1
    }

# ----------------------------------------------------
# Define all models
# ----------------------------------------------------
models = [

    ("LogisticRegression (saga)",
     LogisticRegression(max_iter=300, solver="saga", n_jobs=-1)),

    ("LinearSVC",
     LinearSVC()),

    ("MultinomialNB",
     MultinomialNB()),

    ("BernoulliNB",
     BernoulliNB()),

    ("XGBoost",
     XGBClassifier(
         n_estimators=300,
         max_depth=8,
         learning_rate=0.15,
         subsample=0.8,
         colsample_bytree=0.8,
         tree_method="hist",  # FAST
         n_jobs=-1
     )),

    ("LightGBM",
     LGBMClassifier(
         n_estimators=500,
         boosting_type="gbdt",
         learning_rate=0.1,
         num_leaves=64,
         n_jobs=-1
     ))
]

# ----------------------------------------------------
# Train & evaluate all models
# ----------------------------------------------------
results = []
for name, model in models:
    try:
        result = evaluate_model(name, model)
        results.append(result)
    except Exception as e:
        print(f"Error training {name}: {e}")

# ----------------------------------------------------
# Save best model
# ----------------------------------------------------
best = max(results, key=lambda r: r["f1"])
best_model = best["model"]
best_name = best["name"]

model_path = f"outputs/models/semisupervised/final_classifier_{best_name}.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

print("\n=== Final Ranking (by F1 score) ===")
for r in sorted(results, key=lambda r: r["f1"], reverse=True):
    print(f"{r['name']}: F1={r['f1']:.4f}")

print(f"\nBest model saved to: {model_path}")
