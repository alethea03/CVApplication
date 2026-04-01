import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os

print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training samples : {x_train.shape[0]}")
print(f"Test samples     : {x_test.shape[0]}")
print(f"Image dimensions : {x_train.shape[1]}x{x_train.shape[2]} pixels")
print(f"Number of classes: 10 (digits 0-9)")

x_train_norm = x_train.astype('float32') / 255.0
x_test_norm  = x_test.astype('float32')  / 255.0

y_train_ohe = keras.utils.to_categorical(y_train, 10)
y_test_ohe  = keras.utils.to_categorical(y_test,  10)

print("\n========== ANN MODEL ==========")

x_train_flat = x_train_norm.reshape(-1, 784)
x_test_flat  = x_test_norm.reshape(-1, 784)

ann_model = keras.Sequential([
    layers.Input(shape=(784,)),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(10, activation='softmax')
], name='ANN_Model')

ann_model.summary()

ann_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining ANN...")
ann_history = ann_model.fit(
    x_train_flat, y_train_ohe,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

ann_loss, ann_acc = ann_model.evaluate(x_test_flat, y_test_ohe, verbose=0)
print(f"\nANN Test Accuracy : {ann_acc * 100:.2f}%")
print(f"ANN Test Loss     : {ann_loss:.4f}")

ann_model.save('ann_mnist.h5')
print("ANN model saved as ann_mnist.h5")

print("\n========== CNN MODEL ==========")

x_train_cnn = x_train_norm.reshape(-1, 28, 28, 1)
x_test_cnn  = x_test_norm.reshape(-1, 28, 28, 1)

cnn_model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')
], name='CNN_Model')

cnn_model.summary()

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining CNN...")
cnn_history = cnn_model.fit(
    x_train_cnn, y_train_ohe,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

cnn_loss, cnn_acc = cnn_model.evaluate(x_test_cnn, y_test_ohe, verbose=0)
print(f"\nCNN Test Accuracy : {cnn_acc * 100:.2f}%")
print(f"CNN Test Loss     : {cnn_loss:.4f}")

cnn_model.save('cnn_mnist.h5')
print("CNN model saved as cnn_mnist.h5")

print("\n========== COMPARISON ==========")
print(f"ANN Accuracy : {ann_acc * 100:.2f}%  |  Loss: {ann_loss:.4f}")
print(f"CNN Accuracy : {cnn_acc * 100:.2f}%  |  Loss: {cnn_loss:.4f}")
print("\nConclusion: CNN generally outperforms ANN on image tasks")
print("because Conv layers extract spatial features (edges, curves)")
print("that flat ANN layers cannot capture.")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('ANN vs CNN Training History', fontsize=14, fontweight='bold')

# ANN Accuracy
axes[0][0].plot(ann_history.history['accuracy'],    label='Train')
axes[0][0].plot(ann_history.history['val_accuracy'],label='Validation')
axes[0][0].set_title('ANN Accuracy')
axes[0][0].set_xlabel('Epoch')
axes[0][0].legend()

# ANN Loss
axes[0][1].plot(ann_history.history['loss'],    label='Train')
axes[0][1].plot(ann_history.history['val_loss'],label='Validation')
axes[0][1].set_title('ANN Loss')
axes[0][1].set_xlabel('Epoch')
axes[0][1].legend()

# CNN Accuracy
axes[1][0].plot(cnn_history.history['accuracy'],    label='Train')
axes[1][0].plot(cnn_history.history['val_accuracy'],label='Validation')
axes[1][0].set_title('CNN Accuracy')
axes[1][0].set_xlabel('Epoch')
axes[1][0].legend()

# CNN Loss
axes[1][1].plot(cnn_history.history['loss'],    label='Train')
axes[1][1].plot(cnn_history.history['val_loss'],label='Validation')
axes[1][1].set_title('CNN Loss')
axes[1][1].set_xlabel('Epoch')
axes[1][1].legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()
print("Training plot saved as training_history.png")