#Build and train a simple neural network using a framework like TensorFlow or PyTorch

import os
import numpy as np
import tensorflow as tf
from PIL import Image

class SimpleNeuralNetwork:
    def __init__(self, input_shape):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
    
    def train(self, X_train, y_train, X_val, y_val, epochs):
        history = self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 validation_data=(X_val, y_val),
                                 verbose=1)
        return history
    
    def evaluate(self, X, y):
        _, accuracy = self.model.evaluate(X, y, verbose=0)
        return accuracy
    
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

def load_and_prepare_data(folder_path):
    X = []
    y = []
    
    try:
        files = os.listdir(folder_path)
        
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            # Check if the file is a regular file (not a directory)
            if os.path.isfile(file_path):
                try:
                    # Load image and convert to grayscale and then to numpy array
                    img = Image.open(file_path).convert('L')  # Convert to grayscale
                    img = img.resize((64, 64))  # Resize image if needed
                    arr = np.array(img)

                    # Flatten array and normalize
                    features = arr.flatten() / 255.0  # Normalize pixel values

                    # Assuming file names are labeled as 'class_name.xxx.jpg'
                    label = 1 if 'class_name' in file_name else 0

                    X.append(features)
                    y.append(label)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue  # Skip this file and continue with the next one
            else:
                print(f"{file_path} is not a regular file. Skipping.")

    except FileNotFoundError:
        print(f"Directory '{folder_path}' not found.")
    
    return np.array(X), np.array(y)

# Example usage:
train_folder_path = '/home/rekha-habshipuram/Downloads/archive/train_data'
val_folder_path = '/home/rekha-habshipuram/Downloads/archive/val_data'
test_folder_path = '/home/rekha-habshipuram/Downloads/archive/test_data'

# Load and prepare training, validation, and testing data
X_train, y_train = load_and_prepare_data(train_folder_path)
X_val, y_val = load_and_prepare_data(val_folder_path)
X_test, y_test = load_and_prepare_data(test_folder_path)

# Shuffle the training data
shuffle_index = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Initialize and train the neural network
input_shape = X_train.shape[1:]
model = SimpleNeuralNetwork(input_shape)
history = model.train(X_train, y_train, X_val, y_val, epochs=10)

# Evaluate the model on training, validation, and testing data
train_accuracy = model.evaluate(X_train, y_train)
val_accuracy = model.evaluate(X_val, y_val)
test_accuracy = model.evaluate(X_test, y_test)

# Print accuracies as integers
print(f"Training Accuracy: {train_accuracy * 100:.0f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.0f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.0f}%")

# Individual class accuracies
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

train_class_accuracy = np.mean(y_train == y_train_pred.flatten())
val_class_accuracy = np.mean(y_val == y_val_pred.flatten())
test_class_accuracy = np.mean(y_test == y_test_pred.flatten())

print(f"Training Class Accuracy: {train_class_accuracy * 100:.0f}%")
print(f"Validation Class Accuracy: {val_class_accuracy * 100:.0f}%")
print(f"Testing Class Accuracy: {test_class_accuracy * 100:.0f}%")
