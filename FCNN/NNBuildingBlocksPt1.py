"""
Project: Training a Neuron to Classify Flowers

Objective: Build a complete neural network from scratch (a small one) that can learn to separate two classes of Iris flowers.
"""

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data definition and visualization
X = np.array([
    [1.4, 0.2], [1.3, 0.2], [1.5, 0.2], [1.4, 0.3], # Iris-Setosa
    [4.7, 1.4], [4.5, 1.5], [4.9, 1.5], [4.0, 1.3]  # Iris-Versicolor
])
y = np.array([0,0,0,0,1,1,1,1])

plt.figure(figsize=(8, 6))
plt.scatter(X[:4, 0], X[:4, 1], color='blue', marker='o', label='Iris-Setosa (Class 0)')
plt.scatter(X[4:, 0], X[4:, 1], color='red', marker='x', label='Iris-Versicolor (Class 1)')
plt.title('Flower Petal Dimensions')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.grid(True)
plt.show()

# Step 2 and 3: Perceptron with Sigmoid Activation

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(42)
weights = np.random.rand(2)  # two features
bias = np.random.rand(1)[0]  # single bias value

print(f"Initial Weights: {weights}")
print(f"Initial Bias: {bias}")

# Step 4: Mean Squared Error Loss

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Step 5: Training
learning_rate = 0.1
epochs = 100
losses = []

for epoch in range(epochs):
    # 1. Make predictions
    weighted_sum = np.dot(X, weights) + bias
    predictions = sigmoid(weighted_sum)

    # 2. Calculate the loss
    loss = mean_squared_error(y, predictions)
    losses.append(loss)

    # 3. Calculate the error and update weights/bias
    error = y - predictions
    d_predictions = predictions * (1 - predictions)  # derivative of sigmoid

    # Gradients
    d_weights = np.dot(X.T, error * d_predictions) / len(X)
    d_bias = np.mean(error * d_predictions)

    # Update parameters
    weights += learning_rate * d_weights
    bias += learning_rate * d_bias

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\n--- Training Finished ---")
print(f"Final Weights: {weights}")
print(f"Final Bias: {bias}")

plt.figure(figsize=(8, 6))
plt.plot(losses)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

# Step 6: Visualizing the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:4, 0], X[:4, 1], color='blue', marker='o', label='Iris-Setosa (Class 0)')
plt.scatter(X[4:, 0], X[4:, 1], color='red', marker='x', label='Iris-Versicolor (Class 1)')

x1_line = np.linspace(1, 5, 100)
x2_line = (-weights[0] / weights[1]) * x1_line - (bias / weights[1])
plt.plot(x1_line, x2_line, 'k-', label='Learned Decision Boundary')
plt.title('Final Classification Result')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.xlim(1.2, 5.1)
plt.ylim(0.1, 1.7)
plt.legend()
plt.grid(True)
plt.show()

