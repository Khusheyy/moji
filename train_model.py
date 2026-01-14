import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import tensorflow as tf

print("=" * 60)
print("STEP 1: Loading MNIST Dataset")
print("=" * 60)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Image shape: {x_train.shape[1:]}")

print("\n" + "=" * 60)
print("STEP 2: Preprocessing Data")
print("=" * 60)

x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32')
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0
print(f"Normalized pixel range: {x_train.min():.2f}-{x_train.max():.2f}")

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(f"Labels shape: {y_train.shape}")

print("\n" + "=" * 60)
print("STEP 3: Building CNN Model")
print("=" * 60)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

model.summary()

print("\n" + "=" * 60)
print("STEP 4: Compiling Model")
print("=" * 60)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("✓ Model configured for training")

print("\n" + "=" * 60)
print("STEP 5: Training Model")
print("=" * 60)
print("Training for 15 epochs with batch size 128...\n")

history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=1
)

print("\n" + "=" * 60)
print("STEP 6: Evaluating Model")
print("=" * 60)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"✓ Test Loss: {test_loss:.4f}")

print("\n" + "=" * 60)
print("STEP 7: Saving Model")
print("=" * 60)

model.save('digit_model.h5')
print("✓ Model saved as 'digit_model.h5'")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nIMPORTANT - Use this in your prediction code (backend.py):")
print("-" * 60)
print("image_array = np.array(image)")
print("image_array = 255 - image_array  # INVERT THE IMAGE")
print("image_array = image_array / 255.0")
print("image_array = image_array.reshape(1, 28, 28, 1)")
print("prediction = model.predict(image_array)")
print("-" * 60)
print("\nWhy invert? MNIST has WHITE digits on BLACK background.")
print("Your canvas has BLACK digits on WHITE background.")
print("Inverting fixes this mismatch!")
print("=" * 60)
