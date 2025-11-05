# Artificial Neural Network - Exercise Version
# Complete the TODOs to implement the neural network with hyperparameter tuning

# Part 1 - Data Preprocessing

# Basic imports are provided as examples
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# TODO: Import additional libraries you'll need
# Hint: You need seaborn for visualization and specific sklearn metrics


# Example of ML library imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Deep learning imports (provided)
from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier

# Importing and examining the dataset (provided as example)
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Let's examine our data
print("Feature shape:", X.shape)
print("\nFeature columns:")
print(X.columns.tolist())
print("\nUnique values in categorical columns:")
print("Geography:", X["Geography"].unique())
print("Gender:", X["Gender"].unique())

# TODO: Create dummy variables for categorical features
# Hint: Use pd.get_dummies() for Geography and Gender
# Remember to handle dummy variable trap (drop_first=True)
# Your code here


# TODO: Concatenate dummy variables with original features
# Hint: Use pd.concat with appropriate axis
# Your code here


# TODO: Drop the original categorical columns
# Your code here


# Data splitting (provided)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# TODO: Implement feature scaling
# Hint: Use StandardScaler
# Remember to fit only on training data
# Your code here


## Perform Hyperparameter Optimization

def create_model(layers_list, activation, n_features):
    """
    layers_list : e.g. [40, 20]  -> two hidden layers: 40 neurons then 20 neurons
    activation  : e.g. "relu" or "sigmoid" for hidden layers
    n_features  : number of input features (X_train.shape[1])
    """
    # Sequential model creation (provided)
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_features,)))

    # TODO: Implement the hidden layers
    # For each size in layers_list:
    #   1. Add Dense layer with that size
    #   2. Add activation function
    #   3. Add Dropout (0.3)
    # Your code here


    # Output layer (provided)
    model.add(layers.Dense(1, kernel_initializer='glorot_uniform', activation="sigmoid"))

    # TODO: Compile the model
    # Use 'adam' optimizer and 'binary_crossentropy' loss
    # Your code here

    return model

# KerasClassifier setup (provided)
clf = KerasClassifier(
    model=create_model,
    n_features=X_train.shape[1],
    verbose=0,
)

# TODO: Define your hyperparameter grid
# Requirements:
# - Try at least 2 different layer configurations
# - Try both 'relu' and 'sigmoid' activations
# - Try different batch sizes
# Your code here
param_grid = {
    "model__layers_list": # Your layer configurations here,
    "model__activation": # Your activation functions here,
    "batch_size": # Your batch sizes here,
    "epochs": [30]  # We'll keep this fixed for simplicity
}

# GridSearchCV setup (provided)
grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)

# Model training (provided)
grid_result = grid.fit(X_train, y_train)
print("Best parameters:", grid_result.best_params_)
print("Best cross-validation score:", grid_result.best_score_)

# TODO: Evaluate the model on test set
# 1. Get predictions using grid.predict()
# 2. Convert predictions to binary (0 or 1)
# 3. Create confusion matrix
# Your code here


# TODO: Create visualization
# Create a heatmap of the confusion matrix using seaborn
# Add appropriate labels and title
# Your code here
