"""
Wine Classification with Neural Networks

This script demonstrates how to classify wine samples into three categories using a neural network and the classic sklearn wine dataset. The workflow includes:

1. Data Loading & Exploration: The wine dataset is loaded and its features are inspected.
2. Preprocessing: The target labels are one-hot encoded for multiclass classification. Features are normalized using standard scaling to improve model performance.
3. Data Splitting: The data is split into training and test sets, stratified by class to ensure balanced representation.
4. Model Building: A simple feedforward neural network is constructed using TensorFlow/Keras, with one hidden layer and a softmax output for multiclass classification.
5. Training: The model is trained for 18 epochs, and training loss is visualized to monitor learning progress.
6. Prediction & Evaluation: Predictions are made on the test set, and the model’s performance is evaluated using a confusion matrix and accuracy score.

This script helps learners understand the full workflow of preparing data, building and training a neural network for multiclass classification, and evaluating results using visual tools.
"""

from sklearn.datasets import load_wine
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from numpy.random import seed
from tensorflow.random import set_seed
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Data Loading & Exploration

dataset = load_wine()
X = dataset['data']
y = pd.get_dummies(dataset['target'])

print('Feature names:', dataset['feature_names'])
print('X shape:', X.shape)
print('y shape:', y.shape)

# 2. Preprocessing
scaler = StandardScaler().fit(X)
X_ = scaler.transform(X)
print('Mean values per column after scaling:', X_.mean(axis=0))
print('Std values per column after scaling:', X_.std(axis=0))

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X_, y, test_size=0.2, random_state=0, shuffle=True, stratify=y
)
print('Train/Test shapes:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 4. Model Building
seed(1)
set_seed(2)
def create_model(X_shape):
    inputs = keras.Input(shape=(X_shape[1],))
    hidden_layer = keras.layers.Dense(16, activation="relu")(inputs)
    output_layer = keras.layers.Dense(3, activation="softmax")(hidden_layer)
    model = keras.Model(inputs=inputs, outputs=output_layer)
    return model

model = create_model(X_train.shape)
model.summary()

# 5. Training
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
history = model.fit(X_train, y_train, epochs=18)

# Plot training loss
plt.figure()
sns.lineplot(x=history.epoch, y=history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 6. Prediction & Evaluation
y_pred = model.predict(X_test)
prediction = pd.DataFrame(y_pred, columns=y.columns)
predicted_species = prediction.idxmax(axis="columns")
true_species = y_test.idxmax(axis="columns")

matrix = confusion_matrix(true_species, predicted_species)
print('Confusion matrix:\n', matrix)

# Plot confusion matrix
plt.figure()
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y.columns, yticklabels=y.columns)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

acc = accuracy_score(true_species, predicted_species)
print('Test accuracy:', acc)

