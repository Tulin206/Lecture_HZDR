"""
Image Classification with Deep Learning (CIFAR10)

This script demonstrates the fundamentals of image classification using deep learning techniques on the CIFAR10 dataset. It covers:
- Data loading and preprocessing
- Visualization of sample images
- Building and comparing different CNN architectures
- Training and monitoring models
- Using pooling and dropout for generalization
"""

from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

n_images = 5000
train_images = train_images[:n_images]
train_labels = train_labels[:n_images]

train_images = train_images / 255.
test_images = test_images / 255.

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Visualize 25 sample images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.axis('off')
    plt.title(class_names[train_labels[i,0]])
plt.show()

# 3. Calculate total number of pixels per image
image_dim = train_images.shape[1]*train_images.shape[2]*train_images.shape[3]
print('Total pixels per image:', image_dim)

# 4. Model 1: Simple CNN

def create_cnn1():
    inputs = keras.Input(shape=train_images.shape[1:])
    conv1 = keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
    conv2 = keras.layers.Conv2D(32, (3,3), activation='relu')(conv1)
    flat = keras.layers.Flatten()(conv2)
    outputs = keras.layers.Dense(10)(flat)
    return keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_small")

model = create_cnn1()
model.summary()

# 5. Model 2: CNN with Pooling

def create_cnn2():
    inputs = keras.Input(shape=train_images.shape[1:])
    conv1 = keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
    pool1 = keras.layers.MaxPool2D((2,2))(conv1)
    conv2 = keras.layers.Conv2D(32, (3,3), activation='relu')(pool1)
    pool2 = keras.layers.MaxPool2D((2,2))(conv2)
    flat = keras.layers.Flatten()(pool2)
    outputs = keras.layers.Dense(10)(flat)
    return keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_small")

model = create_cnn2()
model.summary()

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# 6. Plot training history
history_df = pd.DataFrame.from_dict(history.history)
print('History columns:', history_df.columns.tolist())

sns.lineplot(data=history_df[['accuracy','val_accuracy']])
plt.title('Training and Validation Accuracy')
plt.show()

sns.lineplot(data=history_df[['loss','val_loss']])
plt.title('Training and Validation Loss')
plt.show()

# 7. Model 3: CNN with Dropout

def create_cnn3():
    inputs = keras.Input(shape=train_images.shape[1:])
    conv1 = keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
    pool1 = keras.layers.MaxPool2D((2,2))(conv1)
    conv2 = keras.layers.Conv2D(32, (3,3), activation='relu')(pool1)
    pool2 = keras.layers.MaxPool2D((2,2))(conv2)
    dropped = keras.layers.Dropout(0.2)(pool2)
    flat = keras.layers.Flatten()(dropped)
    outputs = keras.layers.Dense(10)(flat)
    return keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_small_withdropout")

model = create_cnn3()
model.summary()

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

history_df = pd.DataFrame.from_dict(history.history)
history_df['epoch'] = range(1,len(history_df)+1)
history_df = history_df.set_index('epoch')

sns.lineplot(data=history_df[['accuracy', 'val_accuracy']])
plt.title('Training and Validation Accuracy (Dropout)')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('Test accuracy:', test_acc)

sns.lineplot(data=history_df[['loss', 'val_loss']])
plt.title('Training and Validation Loss (Dropout)')
plt.show()

