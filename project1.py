#implement a perceptron

import os
import numpy as np
from PIL import Image

class Perceptron:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features + 1)  # +1 for the bias
        self.learning_rate = 0.1
    
    def predict(self, features):
        # Add bias term (x0 = 1)
        activation = np.dot(features, self.weights[1:]) + self.weights[0]
        return 1 if activation >= 0 else 0
    
    def train(self, X_train, y_train, X_val, y_val, epochs):
        best_val_accuracy = 0
        best_weights = np.copy(self.weights)
        
        for epoch in range(epochs):
            for features, label in zip(X_train, y_train):
                prediction = self.predict(features)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * features
                self.weights[0] += self.learning_rate * error
            
            # Evaluate on validation set
            val_accuracy = self.evaluate(X_val, y_val)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_weights = np.copy(self.weights)
        
        # Set the best weights found during training
        self.weights = best_weights
    
    def evaluate(self, X, y):
        correct = 0
        for features, label in zip(X, y):
            prediction = self.predict(features)
            if prediction == label:
                correct += 1
        accuracy = correct / len(y)
        return accuracy

def load_and_prepare_data(folder_path):
    files = os.listdir(folder_path)
    X = []
    y = []
    
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        
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

# Initialize and train the perceptron
perceptron = Perceptron(num_features=X_train.shape[1])
perceptron.train(X_train, y_train, X_val, y_val, epochs=10)

# Evaluate the perceptron on training, validation, and testing data
train_accuracy = perceptron.evaluate(X_train, y_train)
val_accuracy = perceptron.evaluate(X_val, y_val)
test_accuracy = perceptron.evaluate(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.0%}")
print(f"Validation Accuracy: {val_accuracy:.0%}")
print(f"Testing Accuracy: {test_accuracy:.0%}")
