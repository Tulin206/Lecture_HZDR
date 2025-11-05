"""
CIFAR10 Classification with Transfer Learning (ResNet-50 vs VGG-19)

This script demonstrates how to use transfer learning for image classification on the CIFAR10 dataset, comparing two powerful pretrained models: ResNet-50 and VGG-19. The workflow includes:
- Importing libraries
- Loading and preprocessing data
- Visualizing sample images
- Building and training ResNet-50 and VGG-19 models with transfer learning
- Comparing performance and visualizing results
- Confusion matrix analysis
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import gc

# Enable memory growth for GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Load the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Define image size and batch sizes
IMG_SIZE = 224
BATCH_SIZE = 32  # Reduced batch size
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batch_size, img_size, is_training=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_training = is_training
        self.indexes = np.arange(len(self.images))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        i1 = idx * self.batch_size
        i2 = min((idx + 1) * self.batch_size, len(self.images))
        batch_indexes = self.indexes[i1:i2]

        # Process images
        batch_images = np.zeros((len(batch_indexes), self.img_size, self.img_size, 3))
        for i, idx in enumerate(batch_indexes):
            img = self.images[idx]
            img = tf.image.resize(img, [self.img_size, self.img_size])
            img = tf.cast(img, tf.float32) / 255.0
            batch_images[i] = img

        return batch_images, self.labels[batch_indexes]

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

# Create data generators
train_generator = DataGenerator(train_images, train_labels, TRAIN_BATCH_SIZE, IMG_SIZE, True)
test_generator = DataGenerator(test_images, test_labels, TEST_BATCH_SIZE, IMG_SIZE, False)

# Define class names for CIFAR10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Function to create and compile model
def create_model(model_type='resnet'):
    if model_type == 'resnet':
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    else:
        base_model = keras.applications.VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )

    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Train ResNet model
print("Training ResNet-50 model...")
resnet_model = create_model('resnet')
history_resnet = resnet_model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    verbose=1
)

# Clear memory
tf.keras.backend.clear_session()
gc.collect()

# Train VGG model
print("Training VGG-19 model...")
vgg_model = create_model('vgg')
history_vgg = vgg_model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    verbose=1
)

# Convert training histories to DataFrames
history_resnet_df = pd.DataFrame(history_resnet.history)
history_vgg_df = pd.DataFrame(history_vgg.history)

# Plot training and validation accuracy for both models
plt.figure(figsize=(10,5))
sns.lineplot(data=history_resnet_df['val_accuracy'], label='ResNet-50 (val)')
sns.lineplot(data=history_vgg_df['val_accuracy'], label='VGG-19 (val)')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.show()

# Evaluate models using the test generator
resnet_test_loss, resnet_test_acc = resnet_model.evaluate(test_generator, verbose=1)
vgg_test_loss, vgg_test_acc = vgg_model.evaluate(test_generator, verbose=1)
print(f'ResNet-50 Test Accuracy: {resnet_test_acc:.3f}')
print(f'VGG-19 Test Accuracy: {vgg_test_acc:.3f}')

# Show confusion matrix for both models
resnet_pred_labels = np.argmax(resnet_model.predict(test_generator), axis=1)
vgg_pred_labels = np.argmax(vgg_model.predict(test_generator), axis=1)
cm_resnet = confusion_matrix(test_labels.flatten(), resnet_pred_labels)
cm_vgg = confusion_matrix(test_labels.flatten(), vgg_pred_labels)
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('ResNet-50 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.subplot(1,2,2)
sns.heatmap(cm_vgg, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title('VGG-19 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

print("\nSummary:\n")
print("This script compared the performance of ResNet-50 and VGG-19 on the CIFAR10 dataset using transfer learning. You can see the validation accuracy and confusion matrices for both models above. Try adjusting the number of epochs or unfreezing layers for further improvements.")
print("\n---\n")
print("Notebook Workflow Recap:")
print("- Importing Libraries: Essential Python libraries for deep learning, data manipulation, and visualization.")
print("- Loading Data: CIFAR10 dataset with 60,000 color images in 10 classes.")
print("- Preprocessing: Resize images to 224x224 pixels and normalize pixel values.")
print("- Visualization: Display 25 sample images with class names.")
print("- Model Setup: Build ResNet-50 and VGG-19 models with transfer learning and custom classification heads.")
print("- Training: Train both models for 10 epochs.")
print("- Evaluation: Visualize training histories and compare test accuracy.")
print("- Confusion Matrix: Plot confusion matrices for both models.")
print("- Summary: Encourage further experimentation.")
print("\nThis script guides you through applying and comparing state-of-the-art deep learning models on a benchmark dataset, highlighting the power of transfer learning.")
