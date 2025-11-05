# Hyperparameter Tuning for Neural Networks

This implementation demonstrates advanced hyperparameter tuning techniques using both Keras Tuner and scikit-learn's GridSearchCV for optimizing neural networks. The project includes two different approaches:

## Implementations

### 1. Customer Churn Prediction (Current Implementation)
Uses GridSearchCV with KerasClassifier to optimize a neural network for predicting customer churn.

#### Dataset: Churn_Modelling.csv
- Customer information and churn status
- Features include: Credit Score, Geography, Gender, Age, Tenure, Balance, etc.
- Target variable: Customer churn (binary classification)

#### Hyperparameter Search Space
```python
# Network Architecture Options:
- Hidden layer configurations: [[20], [40, 20], [45, 30, 15]]
- Activation functions: ['sigmoid', 'relu']
- Fixed dropout rate: 0.3

# Training Parameters:
- Batch sizes: [128, 256]
- Epochs: 30
- Cross-validation: 5-fold
```

#### Model Architecture
```python
def create_model(layers_list, activation, n_features):
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_features,)))
    
    for nodes in layers_list:
        model.add(layers.Dense(nodes))
        model.add(layers.Activation(activation))
        model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(1, kernel_initializer='glorot_uniform', activation="sigmoid"))
    model.compile(optimizer="adam",
                 loss="binary_crossentropy",
                 metrics=["accuracy"])
    return model
```

## Implementation Details

### 1. Data Processing
```python
# Key steps:
- Load and inspect the Churn_Modelling.csv dataset
- Create dummy variables for categorical features (Geography and Gender)
- Feature scaling using StandardScaler
- Train-test split (80-20 ratio)
```

### 2. Model Training and Tuning
- Uses scikit-learn's GridSearchCV with KerasClassifier wrapper
- 5-fold cross-validation
- Early stopping and dropout for regularization
- Adam optimizer with binary cross-entropy loss

### 3. Results Analysis and Visualization
The notebook provides:
- Best hyperparameter combination
- Model performance metrics
- Confusion matrix visualization using seaborn
- Final accuracy score
- Detailed heatmap visualization of results

## Usage Instructions

**Running the Notebook**
   - Open `Hyperparameter.ipynb` in Jupyter or Google Colab
   - Ensure `Churn_Modelling.csv` is in the working directory
   - Execute cells sequentially

## Best Practices

1. **Data Preparation**
   - Handle categorical variables using dummy encoding
   - Apply feature scaling for better convergence
   - Use proper train-test splitting

2. **Model Training**
   - Use cross-validation for robust evaluation
   - Monitor training progress
   - Apply early stopping to prevent overfitting

3. **Results Analysis**
   - Examine confusion matrix
   - Check accuracy scores
   - Visualize results using heatmaps
