# Task 1: Edge AI Prototype
#
# This script demonstrates how to train a lightweight image classification model,
# convert it to TensorFlow Lite, and test it for Edge AI deployment.
# Use case: Recognizing recyclable items (simulated with CIFAR-10 classes).

# 1. Setup and Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
print('TensorFlow version:', tf.__version__)

# 2. Load and Prepare Dataset
# We'll use CIFAR-10 as a stand-in for recyclable item classification.
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# For demonstration, let's use only 3 classes (e.g., 'automobile', 'ship', 'truck') as recyclable items
recyclable_classes = [1, 8, 9]

def filter_classes(x, y, classes):
    mask = np.isin(y, classes).flatten()
    x_filt = x[mask]
    y_filt = y[mask]
    # Map to 0,1,2
    y_filt = np.array([classes.index(label[0]) for label in y_filt]).reshape(-1, 1)
    return x_filt, y_filt

x_train_filt, y_train_filt = filter_classes(x_train, y_train, recyclable_classes)
x_test_filt, y_test_filt = filter_classes(x_test, y_test, recyclable_classes)
print('Filtered train shape:', x_train_filt.shape)

# 3. Build and Train a Simple CNN Model
model = keras.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(x_train_filt, y_train_filt, epochs=5, validation_data=(x_test_filt, y_test_filt))

# 4. Evaluate Model Performance
test_loss, test_acc = model.evaluate(x_test_filt, y_test_filt, verbose=2)
print('Test accuracy:', test_acc)

# 5. Convert Model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('recyclable_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
print('TFLite model saved as recyclable_classifier.tflite')

# 6. Test TFLite Model Inference
# We'll use the TensorFlow Lite Interpreter to run inference on a few test images.
interpreter = tf.lite.Interpreter(model_path='recyclable_classifier.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test on 5 random images
for i in range(5):
    idx = np.random.randint(0, x_test_filt.shape[0])
    input_data = x_test_filt[idx:idx+1].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output)
    true = y_test_filt[idx][0]
    print(f'True: {recyclable_classes[true]}, Predicted: {recyclable_classes[pred]}')

# 7. Edge AI Benefits
# Deploying this model on an edge device (e.g., Raspberry Pi, mobile phone, or smart bin)
# enables real-time classification without sending images to the cloud. This reduces latency,
# enhances privacy, and saves bandwidth—making it ideal for applications like smart recycling bins,
# autonomous drones, or mobile apps. 