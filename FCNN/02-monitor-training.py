"""
Predicting Sunshine Hours Tomorrow with Neural Networks

This script demonstrates a regression task using a weather prediction dataset. The goal is to predict the number of sunshine hours in Basel for the next day using historical weather data and neural networks. The workflow includes:

- Data Loading & Exploration
- Feature Selection
- Data Splitting
- Model Building
- Training & Monitoring
- Evaluation & Visualization
- Baseline Comparison
- Further Techniques
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# Data Loading & Exploration

df = pd.read_csv("https://zenodo.org/record/5071376/files/weather_prediction_dataset.csv")
print(df.describe())
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Unique suffixes:', set({ "_".join(x.split("_")[1:]) for x in df.columns if x not in ["MONTH", "DATE"]  }))

# Feature Selection
nr_rows = 365*3
X_data = df.loc[:nr_rows].drop(columns=["DATE", "MONTH"])
print('X_data shape:', X_data.shape)
print('BASEL columns:', [item for item in df.columns if "BASEL" in item])
y_data = df.loc[1:(nr_rows+1)]["BASEL_sunshine"]
print('y_data dtype:', y_data.dtype)

# Data Splitting
X_train, X_not_train, y_train, y_not_train = train_test_split(X_data, y_data, test_size=.3, random_state=20211013)
X_val, X_test, y_val, y_test = train_test_split(X_not_train, y_not_train, test_size=.5, random_state=20211014)
print('Train shape:', X_train.shape)
print('Test/Val shapes:', X_test.shape, X_val.shape)

# Model Building

def create_nn(nodes1=100, nodes2=50):
    inputs = keras.Input(shape=(X_data.shape[1],), name="input")
    dense1 = keras.layers.Dense(nodes1, 'relu', name="dense1")(inputs)
    dense2 = keras.layers.Dense(nodes2, 'relu', name="dense2")(dense1)
    outputs = keras.layers.Dense(1)(dense2)
    return keras.Model(inputs=inputs, outputs=outputs, name="weather_prediction_model")

model = create_nn()
model.summary()

# Training & Monitoring
model.compile(optimizer="adam", loss="mse", metrics=[keras.metrics.RootMeanSquaredError()])
history = model.fit(X_train, y_train, batch_size=32, epochs=200, verbose=2)
history_df = pd.DataFrame.from_dict(history.history)
print('History columns:', history_df.columns.tolist())

# Plot RMSE
plt.figure()
sns.lineplot(data=history_df['root_mean_squared_error'])
plt.xlabel("epochs")
plt.ylabel("RMSE")
plt.title("Training RMSE")
plt.show()

# Evaluation & Visualization
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].scatter(y_train_predict, y_train, s=10, alpha=0.5, color="teal")
axes[0].set_title("training set")
axes[0].set_xlabel("predicted sunshine hours")
axes[0].set_ylabel("true sunshine hours")
axes[1].scatter(y_test_predict, y_test, s=10, alpha=0.5, color="teal")
axes[1].set_title("test set")
axes[1].set_xlabel("predicted sunshine hours")
axes[1].set_ylabel("true sunshine hours")
plt.show()

loss_train, rmse_train = model.evaluate(X_train, y_train)
loss_test, rmse_test = model.evaluate(X_test, y_test)
print("training set RMSE:", rmse_train)
print("test set RMSE:", rmse_test)

# Baseline Comparison
y_baseline_prediction = X_test["BASEL_sunshine"]
plt.figure(figsize=(5,5), dpi=100)
plt.scatter(y_baseline_prediction, y_test, s=10, alpha=.5)
plt.xlabel("sunshine hours yesterday (baseline)")
plt.ylabel("true sunshine hours")
plt.title("Baseline vs True")
plt.show()

rmse_nn = mean_squared_error(y_test, y_test_predict, squared=False)
rmse_baseline = mean_squared_error(y_test, y_baseline_prediction , squared=False)
print("NN RMSE:", rmse_nn)
print("Baseline RMSE:", rmse_baseline)

# Training with Validation
model = create_nn()
model.compile(optimizer='adam', loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])
history = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_val, y_val), verbose=2)
history_df = pd.DataFrame.from_dict(history.history)
plt.figure()
sns.lineplot(data=history_df[['root_mean_squared_error','val_root_mean_squared_error']])
plt.xlabel("epochs")
plt.ylabel("RMSE")
plt.title("Training & Validation RMSE")
plt.show()

# Counteract Overfitting: Smaller Model
model = create_nn(10,5)
model.summary()
model.compile(optimizer='adam', loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])
history = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_val, y_val), verbose=2)
history_df = pd.DataFrame.from_dict(history.history)
plt.figure()
sns.lineplot(data=history_df[['root_mean_squared_error', 'val_root_mean_squared_error']])
plt.xlabel("epochs")
plt.ylabel("RMSE")
plt.title("Small Model Training & Validation RMSE")
plt.show()

# Larger Model with EarlyStopping
model = create_nn(100, 50)
model.compile(optimizer='adam', loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])
earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
history = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_val, y_val), callbacks=[earlystop], verbose=2)
history_df = pd.DataFrame.from_dict(history.history)
plt.figure()
sns.lineplot(data=history_df[['root_mean_squared_error', 'val_root_mean_squared_error']])
plt.xlabel("epochs")
plt.ylabel("RMSE")
plt.title("Large Model Training & Validation RMSE with EarlyStopping")
plt.show()

print("Further techniques of interest: batch normalization, dropout layers.")

