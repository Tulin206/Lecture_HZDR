import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build AlexNet-like model
model = models.Sequential([
    # 1st Conv Layer
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
    layers.MaxPooling2D((2, 2)),
    # 2nd Conv Layer
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    # 3rd Conv Layer
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    # 4th Conv Layer
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    # 5th Conv Layer
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    # 1st Fully Connected Layer
    layers.Dense(512, activation='relu'),
    # 2nd Fully Connected Layer
    layers.Dense(256, activation='relu'),
    # Softmax Layer
    layers.Dense(10, activation='softmax')
])

# Show model summary
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Print confusion matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Count correct and incorrect predictions
correct = np.sum(y_pred_classes.flatten() == y_test.flatten())
incorrect = len(y_test) - correct
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Visualize some test images with predicted labels
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i])
    plt.title(f"Pred: {class_names[y_pred_classes[i]]}")
    plt.axis('off')
plt.suptitle("Test Images with Predicted Labels")
plt.show()

