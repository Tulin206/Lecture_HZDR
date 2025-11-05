# Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#Create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

## Concatenate the Data Frames

X=pd.concat([X,geography,gender],axis=1)

## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Perform Hyperparameter Optimization

def create_model(layers_list, activation, n_features):
    """
    layers_list : e.g. [40, 20]  -> two hidden layers: 40 neurons then 20 neurons
    activation  : e.g. "relu" or "sigmoid" for hidden layers
    n_features  : number of input features (X_train.shape[1])
    """

    model = keras.Sequential()

    # 1) Tell Keras the input size (modern way: an explicit Input layer)
    model.add(layers.Input(shape=(n_features,)))

    # 2) Hidden layers (loop through your list)
    for nodes in layers_list:
        model.add(layers.Dense(nodes))  # Dense = fully-connected layer
        model.add(layers.Activation(activation))  # your chosen nonlinearity
        model.add(layers.Dropout(0.3))  # turn off 30% units (reduces overfitting)

    # 3) Output layer for binary classification (one probability)
    model.add(layers.Dense(1, kernel_initializer= 'glorot_uniform', activation="sigmoid"))

    # 4) Training setup: optimizer, loss, metric
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

clf = KerasClassifier(
    model=create_model,
    n_features=X_train.shape[1],  # passed into your function
    verbose=0,
)

# candidates
layers_candidates = [[20], [40, 20], [45, 30, 15]]
activations = ['sigmoid', 'relu']

# ✅ Note the model__ prefix for arguments to create_model
param_grid = {
    "model__layers_list": layers_candidates,
    "model__activation": activations,
    "batch_size": [128, 256],
    "epochs": [30],
}

grid = GridSearchCV(estimator=clf, param_grid=param_grid,cv=5)

grid_result = grid.fit(X_train, y_train)
print("Best params:", grid_result.best_params_)
print("Best score:", grid_result.best_score_)

# Evaluate on test set
best_clf = grid.best_estimator_
test_acc = best_clf.score(X_test, y_test)
print("Test accuracy:", test_acc)

pred_y = grid.predict(X_test)
y_pred = (pred_y > 0.5).astype(int)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print("Accuracy:", score)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Accuracy = {score:.2f})')
plt.show()












