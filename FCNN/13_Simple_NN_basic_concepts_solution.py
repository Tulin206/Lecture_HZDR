#%% md
# #  🧠 A Neural Network for the Iris Dataset (Solution)
# #
# # This notebook builds a neural network to classify the famous **Iris dataset** using only **NumPy**.
# #
# # **Key concepts:**
# # - One-Hot Encoding
# # - Softmax Activation
# # - Cross-Entropy Loss
# # - Data Scaling
# 
#%%
import numpy as np
from sklearn.datasets import load_iris

#%%
# Set random seed for reproducibility
np.random.seed(42)

#%% md
# # Step 1: Load and Preprocess Data
# 
#%%
# Load Data
iris = load_iris()
X_raw = iris.data      # (150 samples, 4 features)
y_raw = iris.target    # (150 samples,) containing 0, 1, 2

#%%
# Define Helper Functions

def to_one_hot(y, num_classes):
    """Converts (150,) array of ints to (150, 3) one-hot matrix"""
    one_hot = np.zeros((y.shape[0], num_classes))
    for i, label in enumerate(y):
        one_hot[i, label] = 1.0
    return one_hot

def min_max_scale(X):
    """Scales features to be between 0 and 1"""
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    return (X - min_val) / (max_val - min_val)

#%%
# Preprocess Data
X = min_max_scale(X_raw)         # Scale features to [0, 1]

# One-hot encode the outputs
num_classes = 3
y = to_one_hot(y_raw, num_classes)

#%%
# Shuffle the data (important because Iris is sorted by class!)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

print("First 5 X samples (scaled):\n", X[:5])
print("\nFirst 5 y samples (one-hot):\n", y[:5])

#%% md
# # Step 2: Advanced Activation & Loss
# 
#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    # Subtract max for numerical stability (prevents blowing up exp)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#%% md
# # Step 3: Network Architecture
# 
#%%
input_neurons = 4
hidden_neurons = 6
output_neurons = 3

learning_rate = 0.1
epochs = 5000

# Initialize weights (standard normal distribution usually works better here)
# We multiply by 0.1 to keep initial weights small

weights_hidden = np.random.randn(input_neurons, hidden_neurons) * 0.1
bias_hidden = np.zeros((1, hidden_neurons))

weights_output = np.random.randn(hidden_neurons, output_neurons) * 0.1
bias_output = np.zeros((1, output_neurons))

#%% md
# # Step 4: The Training Loop
# 
#%%
loss_history = []

for i in range(epochs):
    # 1. FORWARD PASS
    hidden_layer_input = np.dot(X, weights_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_output) + bias_output
    # USE SOFTMAX NOW
    predicted_output = softmax(output_layer_input)

    # 2. LOSS (Cross-Entropy for monitoring)
    # Small epsilon to prevent log(0) errors
    epsilon = 1e-15
    clipped_preds = np.clip(predicted_output, epsilon, 1 - epsilon)
    loss = -np.mean(np.sum(y * np.log(clipped_preds), axis=1))
    loss_history.append(loss)

    # 3. BACKWARD PASS
    # Gradient of Cross-Entropy + Softmax is just (Pred - Actual)
    d_predicted_output = (predicted_output - y) / X.shape[0] # Normalize by batch size

    # Backprop to hidden layer
    error_hidden = d_predicted_output.dot(weights_output.T)
    d_hidden_layer = error_hidden * sigmoid_derivative(hidden_layer_output)

    # 4. UPDATE WEIGHTS
    weights_output -= hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output -= np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_hidden -= X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden -= np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if i % 500 == 0:
        print(f"Epoch {i} Loss: {loss:.4f}")

#%% md
# # Step 5: Testing Accuracy
# 
#%%
# Final forward pass to get predictions
hidden_out = sigmoid(np.dot(X, weights_hidden) + bias_hidden)
final_preds_prob = softmax(np.dot(hidden_out, weights_output) + bias_output)

# Convert probabilities to class labels (0, 1, or 2)
predicted_classes = np.argmax(final_preds_prob, axis=1)
true_classes = np.argmax(y, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f"\nFinal Training Accuracy: {accuracy * 100:.2f}%")

# Show a few examples
print("\nSample Predictions vs True Labels:")
print(f"Predicted: {predicted_classes[:10]}")
print(f"True:      {true_classes[:10]}")
