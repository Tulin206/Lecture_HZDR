"""
CIFAR10 Classification with a Custom VGG-like Network

This script guides you through building a VGG-style convolutional neural network from scratch to classify images in the CIFAR10 dataset. The workflow includes:
- Importing libraries
- Loading and preprocessing data
- Visualizing sample images
- Building a VGG-like CNN
- Training and evaluating the model
- Visualizing training progress
- Confusion matrix analysis
"""

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# Load the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define class names for CIFAR10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Display 25 sample images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i])
    plt.axis('off')
    plt.title(class_names[train_labels[i,0]])
plt.show()

# Build a VGG-like model from scratch
def build_vgg_like(input_shape, num_classes):
    model = keras.Sequential()
    # Block 1
    model.add(keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    # Block 2
    model.add(keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    # Block 3
    model.add(keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    # Flatten and Dense layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    return model

# Create the VGG-like model
model = build_vgg_like(train_images.shape[1:], 10)
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=30,
                    batch_size=64,
                    validation_data=(test_images, test_labels),
                    verbose=2)

# Convert training history to DataFrame
history_df = pd.DataFrame(history.history)

# Plot training and validation accuracy
sns.lineplot(data=history_df[['accuracy','val_accuracy']])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.show()

# Plot training and validation loss
sns.lineplot(data=history_df[['loss','val_loss']])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc:.3f}')

# Predict and show confusion matrix
pred_labels = np.argmax(model.predict(test_images), axis=1)
cm = confusion_matrix(test_labels, pred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

