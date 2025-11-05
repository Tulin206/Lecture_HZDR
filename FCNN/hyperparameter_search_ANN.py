"""
hyperparameter_search_ANN.py

Description:
This script demonstrates simple hyperparameter tuning (grid search and random search)
for a small Artificial Neural Network (ANN) on the CIFAR-10 dataset using TensorFlow/Keras.

Why this is useful:
- Hyperparameters (like learning rate, number of hidden units, dropout) strongly affect
  model performance.
- Grid search systematically explores combinations; random search samples combinations
  randomly and can be more efficient when only a few hyperparameters matter.
- This script is intentionally simple and uses manual loops so beginners can see the
  tuning process without extra dependencies.

Notes for beginners:
- Keep epochs small for tuning to save time.
- Use a validation split to compare hyperparameter settings.
- After identifying good hyperparameters, retrain a final model with more epochs.
"""

# Simple and clear imports
import itertools
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility (optional)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# 1) Load and preprocess CIFAR-10
# -------------------------
# Load data
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixels to range [0,1]
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# For a simple ANN, flatten images (32x32x3 -> 3072)
x_train_full_flat = x_train_full.reshape((x_train_full.shape[0], -1))
x_test_flat = x_test.reshape((x_test.shape[0], -1))

# Convert labels to integers (already integers) and keep shape (n,)
y_train_full = y_train_full.ravel()
y_test = y_test.ravel()

# Split a validation set from training data for hyperparameter search
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full_flat, y_train_full, test_size=0.1, random_state=SEED, stratify=y_train_full
)

# -------------------------
# 2) Model builder function
# -------------------------
def create_model(input_shape, num_classes,
                 hidden_units=512, dropout_rate=0.5,
                 activation="relu", learning_rate=1e-3):
    """
    Build and compile a simple dense (fully-connected) network.
    Arguments are hyperparameters we will tune.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    # First dense block
    model.add(layers.Dense(hidden_units, activation=activation))
    model.add(layers.Dropout(dropout_rate))
    # Second dense block
    model.add(layers.Dense(hidden_units // 2, activation=activation))
    model.add(layers.Dropout(dropout_rate / 2))
    # Output layer
    model.add(layers.Dense(num_classes, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# -------------------------
# 3) Hyperparameter search configuration
# -------------------------
# Define grid for grid search (keep small for speed/demonstration)
param_grid = {
    "hidden_units": [256, 512],
    "dropout_rate": [0.3, 0.5],
    "learning_rate": [1e-3, 1e-4],
    "batch_size": [64, 128],
    "activation": ["relu"],
}

# Number of random samples for random search
n_random_search = 6  # small number for demo; increase as needed

# Number of epochs to train during the search (small for speed)
search_epochs = 5

# -------------------------
# 4) Helper: evaluate one hyperparameter setting
# -------------------------
def evaluate_setting(params):
    """
    Train model for a few epochs on the training set and return validation accuracy.
    params: dict with keys hidden_units, dropout_rate, learning_rate, batch_size, activation
    """
    model = create_model(
        input_shape=x_train.shape[1],
        num_classes=10,
        hidden_units=params["hidden_units"],
        dropout_rate=params["dropout_rate"],
        activation=params["activation"],
        learning_rate=params["learning_rate"],
    )

    # Verbose=0 to keep output tidy during many trainings
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=search_epochs,
        batch_size=params["batch_size"],
        verbose=0,
    )

    # Get the best validation accuracy achieved during this run
    val_acc = max(history.history["val_accuracy"])
    return val_acc

# -------------------------
# 5) Grid Search (manual)
# -------------------------
print("Starting manual grid search over {} combinations...".format(
    np.prod([len(v) for v in param_grid.values()])
))

# Create list of all combinations
keys = list(param_grid.keys())
all_combinations = list(itertools.product(*[param_grid[k] for k in keys]))

grid_results = []
for comb in all_combinations:
    params = dict(zip(keys, comb))
    print("Grid search trying:", params)
    val_acc = evaluate_setting(params)
    print(f" -> val_accuracy = {val_acc:.4f}")
    grid_results.append((val_acc, params))

# Find best grid result
best_grid_acc, best_grid_params = max(grid_results, key=lambda x: x[0])
print("\nBest grid search setting:")
print(best_grid_params)
print(f"Best validation accuracy (grid) = {best_grid_acc:.4f}")

# -------------------------
# 6) Random Search (manual)
# -------------------------
print("\nStarting manual random search ({} trials)...".format(n_random_search))
random_results = []
# For random search, sample from ranges (here we sample from the grid lists for simplicity)
for i in range(n_random_search):
    params = {
        "hidden_units": random.choice(param_grid["hidden_units"]),
        "dropout_rate": random.choice(param_grid["dropout_rate"]),
        "learning_rate": random.choice(param_grid["learning_rate"]),
        "batch_size": random.choice(param_grid["batch_size"]),
        "activation": random.choice(param_grid["activation"]),
    }
    print("Random search trial {}: {}".format(i + 1, params))
    val_acc = evaluate_setting(params)
    print(f" -> val_accuracy = {val_acc:.4f}")
    random_results.append((val_acc, params))

best_random_acc, best_random_params = max(random_results, key=lambda x: x[0])
print("\nBest random search setting:")
print(best_random_params)
print(f"Best validation accuracy (random) = {best_random_acc:.4f}")

# -------------------------
# 7) Summary and suggestion for final training
# -------------------------
print("\nSummary:")
print(f"Grid best val_accuracy = {best_grid_acc:.4f} with params = {best_grid_params}")
print(f"Random best val_accuracy = {best_random_acc:.4f} with params = {best_random_params}")

print("\nSuggestion: retrain a final model with the best hyperparameters for more epochs (e.g., 30-100) and evaluate on the test set.")

# Example: quick final training with best grid params (brief)
final_params = best_grid_params
print("\nQuick final training using best grid params (for demonstration)...")
final_model = create_model(
    input_shape=x_train.shape[1],
    num_classes=10,
    hidden_units=final_params["hidden_units"],
    dropout_rate=final_params["dropout_rate"],
    activation=final_params["activation"],
    learning_rate=final_params["learning_rate"],
)
final_model.fit(
    np.concatenate([x_train, x_val]),  # use both train+val for final training
    np.concatenate([y_train, y_val]),
    epochs=10,  # keep small here; increase in real use
    batch_size=final_params["batch_size"],
    verbose=1,
)

test_loss, test_acc = final_model.evaluate(x_test_flat, y_test, verbose=0)
print(f"Test accuracy of final model (demo) = {test_acc:.4f}")

