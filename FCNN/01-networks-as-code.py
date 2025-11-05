"""
Penguin Species Classification with Neural Networks

This script walks you through building a neural network to classify penguin species using the Palmer Penguins dataset. Here’s what happens step by step:

1. Data Loading & Exploration: The dataset is loaded and visualized to understand feature relationships and species distribution.
2. Preprocessing: Irrelevant columns are dropped, missing values are removed, and the target variable ('species') is one-hot encoded for neural network training.
3. Data Splitting: The data is split into training and test sets, stratified by species to ensure balanced classes.
4. Model Building: A simple feedforward neural network is constructed using TensorFlow/Keras, with one hidden layer and a softmax output for multiclass classification.
5. Training: The model is trained for 100 epochs, and training loss is visualized to monitor learning progress.
6. Prediction & Evaluation: Predictions are made on the test set, and the model’s performance is evaluated using a confusion matrix and heatmap to show classification accuracy for each species.

This script helps learners understand the full workflow of preparing data, building and training a neural network for multiclass classification, and evaluating results using visual tools.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from numpy.random import seed
from tensorflow.random import set_seed
from sklearn.metrics import confusion_matrix

# 1. Data Loading & Exploration
penguins = sns.load_dataset('penguins')
sns.pairplot(penguins, hue="species")
plt.show()

# 2. Preprocessing
penguins['species'] = penguins['species'].astype('category')
penguins_filtered = penguins.drop(columns=['island', 'sex']).dropna()
penguins_features = penguins_filtered.drop(columns=['species'])
target = pd.get_dummies(penguins_filtered['species'])

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    penguins_features, target, test_size=0.2, random_state=0, shuffle=True, stratify=target
)

# Print class counts
print('Full target set counts:\n', target.value_counts())
print('Train set counts:\n', y_train.value_counts())
print('Test set counts:\n', y_test.value_counts())

# 4. Model Building
seed(1)
set_seed(2)
inputs = keras.Input(shape=X_train.shape[1])
hidden_layer = keras.layers.Dense(10, activation="relu")(inputs)
output_layer = keras.layers.Dense(3, activation="softmax")(hidden_layer)
model = keras.Model(inputs=inputs, outputs=output_layer)
model.summary()

# 5. Training
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
history = model.fit(X_train, y_train, epochs=100)

# Plot training loss
plt.figure()
sns.lineplot(x=history.epoch, y=history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 6. Prediction & Evaluation
y_pred = model.predict(X_test)
prediction = pd.DataFrame(y_pred, columns=target.columns)
predicted_species = prediction.idxmax(axis="columns")
true_species = y_test.idxmax(axis="columns")
matrix = confusion_matrix(true_species, predicted_species)
print('Confusion matrix:\n', matrix)

# Confusion matrix as DataFrame for heatmap
confusion_df = pd.DataFrame(matrix, index=y_test.columns.values, columns=y_test.columns.values)
confusion_df.index.name = 'True Label'
confusion_df.columns.name = 'Predicted Label'

plt.figure()
sns.heatmap(confusion_df, annot=True)
plt.title('Confusion Matrix')
plt.show()

