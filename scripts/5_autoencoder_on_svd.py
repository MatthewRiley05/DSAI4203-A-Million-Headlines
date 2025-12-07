import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger
import logging
import builtins

tf.random.set_seed(42)
np.random.seed(42)

# Logging allows to print everything to a log file
logging.basicConfig(
    filename="outputs/logs/ae.log",
    level=logging.INFO,
    format="%(message)s",
    filemode="w"
)

logger= logging.getLogger()
builtins.print = logger.info
csv_logger = CSVLogger("outputs/logs/ae_history.csv", append=True)

def print_section(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n")

# ------------------------------------------------------------------

print_section("LOADING SVD FEATURES")

with open("outputs/models/svd_features.pkl", "rb") as f:
    svd_features = pickle.load(f)          # shape: (1213004, 200)

X = svd_features.astype("float32")
n_samples, input_dim = X.shape
print(f"SVD feature matrix shape: {X.shape}")

# ------------------------------------------------------------------
# STEP 1: TRAIN AUTOENCODER ON A SUBSET (for speed)
# ------------------------------------------------------------------
print_section("TRAINING AUTOENCODER ON SUBSET")

rng = np.random.RandomState(42)
subset_size = min(200_000, n_samples)     
subset_idx = rng.choice(n_samples, subset_size, replace=False)
X_train = X[subset_idx]

latent_dim = 10

inputs = layers.Input(shape=(input_dim,), name="svd_input")
x = layers.Dense(128, activation="relu")(inputs)
x = layers.Dense(64, activation="relu")(x)
z = layers.Dense(latent_dim, activation="linear", name="latent")(x)

x = layers.Dense(64, activation="relu")(z)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(input_dim, activation="linear")(x)

autoencoder = models.Model(inputs, outputs, name="svd_autoencoder")
encoder = models.Model(inputs, z, name="svd_encoder")

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

epochs = 10
batch_size = 2048

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.1,
    verbose=1,
    callbacks=[csv_logger], 
)

print("Autoencoder training complete!")

# ------------------------------------------------------------------
# STEP 2: ENCODE ALL DOCUMENTS
# ------------------------------------------------------------------
print_section("ENCODING FULL DATASET")

Z = encoder.predict(X, batch_size=4096)
print(f"Latent representation shape: {Z.shape}")

# ------------------------------------------------------------------
# STEP 3: SAVE ENCODER + LATENT FEATURES
# ------------------------------------------------------------------
print_section("SAVING MODELS AND FEATURES")

encoder.save("outputs/models/ae_encoder.keras")
with open("outputs/models/ae_features.pkl", "wb") as f:
    pickle.dump(Z, f)

with open("outputs/models/ae_training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Saved encoder, latent features, and training history to 'outputs/models'.")
print_section("AUTOENCODER PIPELINE COMPLETE")
