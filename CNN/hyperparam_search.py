"""
hyperparam_search.py

Simple demonstration of Grid Search, Randomized Search, and Optuna for hyperparameter tuning
on a small CNN using CIFAR-10 (already used in this repo). This script is self-contained and
keeps models small so it runs reasonably fast on CPU.

Usage: python hyperparam_search.py

Notes:
- This script uses scikit-learn wrappers and Optuna.
- To keep things simple and avoid heavy installs, the implementations are minimal. If Optuna is
  not installed in the environment, the Optuna section will be skipped with a helpful message.

"""

# ===============================================
# Imports and Setup
# ===============================================
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from itertools import product

# Optional: optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

# ===============================================
# Set Random Seeds
# ===============================================
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# ===============================================
# Data Preparation
# ===============================================
# Load CIFAR-10 and preprocess
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# For faster demo, use smaller subset (2000 train, 500 test)
train_subset = 2000
test_subset = 500
x = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test]).flatten()

x = x.astype("float32") / 255.0

# select subsets
x_small, _, y_small, _ = train_test_split(
    x, y,
    train_size=train_subset + test_subset,
    stratify=y,
    random_state=seed
)
x_train_small, x_test_small, y_train_small, y_test_small = train_test_split(
    x_small, y_small,
    train_size=train_subset,
    stratify=y_small,
    random_state=seed
)

# ===============================================
# Model Definition
# ===============================================
num_classes = 10
input_shape = x_train_small.shape[1:]

def build_model(conv_filters=16, kernel_size=3, dense_units=64, dropout_rate=0.3, learning_rate=1e-3):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(conv_filters, kernel_size, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(conv_filters*2, kernel_size, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def evaluate_model(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return acc

# ===============================================
# 1. Grid Search Implementation
# ===============================================
DEFAULT_EPOCHS = 3
DEFAULT_BATCH = 64

param_grid = {
    'conv_filters': [8, 16],
    'dense_units': [32, 64],
    'dropout_rate': [0.2, 0.4],
}

print('\nStarting simple manual grid search...')
start = time.time()
best_acc = 0.0
best_params = None

for conv_filters, dense_units, dropout_rate in product(
    param_grid['conv_filters'],
    param_grid['dense_units'],
    param_grid['dropout_rate']
):
    params = dict(conv_filters=conv_filters, dense_units=dense_units, dropout_rate=dropout_rate)
    print('Testing params:', params)
    model = build_model(**params)
    model.fit(x_train_small, y_train_small, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH, verbose=0)
    acc = evaluate_model(model, x_test_small, y_test_small)
    print(' Accuracy:', acc)
    if acc > best_acc:
        best_acc = acc
        best_params = params

print('Grid search done. Best accuracy: {:.4f} with {}'.format(best_acc, best_params))
print('Time:', time.time() - start)

# ===============================================
# 2. Random Search Implementation
# ===============================================
print('\nStarting random search...')
start = time.time()
best_acc_rs = 0.0
best_params_rs = None

param_distributions = {
    'conv_filters': [8, 16, 24],
    'dense_units': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.4],
    'learning_rate': [1e-3, 5e-4, 1e-4]
}

n_iter = 4
for i in range(n_iter):
    params = {k: random.choice(v) for k, v in param_distributions.items()}
    print('Iter', i+1, 'params:', params)
    model = build_model(**params)
    model.fit(x_train_small, y_train_small, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH, verbose=0)
    acc = evaluate_model(model, x_test_small, y_test_small)
    print(' Accuracy:', acc)
    if acc > best_acc_rs:
        best_acc_rs = acc
        best_params_rs = params

print('Random search done. Best accuracy: {:.4f} with {}'.format(best_acc_rs, best_params_rs))
print('Time:', time.time() - start)

# ===============================================
# 3. Optuna Implementation (if available)
# ===============================================
if OPTUNA_AVAILABLE:
    print('\nStarting Optuna optimization...')
    start = time.time()

    def objective(trial):
        conv = trial.suggest_categorical('conv_filters', [8, 16, 24])
        dense = trial.suggest_categorical('dense_units', [32, 64, 128])
        drop = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lr = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

        model = build_model(
            conv_filters=conv,
            dense_units=dense,
            dropout_rate=drop,
            learning_rate=lr
        )
        model.fit(x_train_small, y_train_small, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH, verbose=0)
        acc = evaluate_model(model, x_test_small, y_test_small)
        return acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=4)
    print('Optuna best value:', study.best_value)
    print('Optuna best params:', study.best_params)
    print('Time:', time.time() - start)
else:
    print('\nOptuna not installed. Skipping Optuna section. To run Optuna, install it with:\n    pip install optuna')

print('\nAll searches completed.')
