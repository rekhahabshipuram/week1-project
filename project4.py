#Compare performance with various optimization algorithms

import os
import numpy as np
from PIL import Image

class Perceptron:
    def __init__(self, num_features, optimizer='gradient_descent', learning_rate=0.1, batch_size=None):
        self.weights = np.zeros(num_features + 1)  # +1 for the bias
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
    
    def predict(self, features):
        # Add bias term (x0 = 1)
        activation = np.dot(features, self.weights[1:]) + self.weights[0]
        return np.where(activation >= 0, 1, 0)
    
    def train(self, X_train, y_train, X_val, y_val, epochs):
        best_val_accuracy = 0
        best_weights = np.copy(self.weights)
        
        for epoch in range(epochs):
            if self.optimizer == 'gradient_descent':
                self.gradient_descent(X_train, y_train)
            elif self.optimizer == 'sgd':
                self.stochastic_gradient_descent(X_train, y_train)
            elif self.optimizer == 'mini_batch':
                self.mini_batch_gradient_descent(X_train, y_train)
            
            # Evaluate on validation set
            val_accuracy = self.evaluate(X_val, y_val)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_weights = np.copy(self.weights)
        
        # Set the best weights found during training
        self.weights = best_weights
    
    def gradient_descent(self, X_train, y_train):
        errors = y_train - self.predict(X_train)
        self.weights[1:] += self.learning_rate * np.dot(errors.T, X_train)
        self.weights[0] += self.learning_rate * np.sum(errors)
    
    def stochastic_gradient_descent(self, X_train, y_train):
        for features, label in zip(X_train, y_train):
            prediction = self.predict(features)
            error = label - prediction
            self.weights[1:] += self.learning_rate * error * features
            self.weights[0] += self.learning_rate * error
    
    def mini_batch_gradient_descent(self, X_train, y_train):
        if self.batch_size is None:
            self.batch_size = len(X_train)
        
        for i in range(0, len(X_train), self.batch_size):
            X_batch = X_train[i:i+self.batch_size]
            y_batch = y_train[i:i+self.batch_size]
            
            errors = y_batch - self.predict(X_batch)
            self.weights[1:] += self.learning_rate * np.dot(errors.T, X_batch)
            self.weights[0] += self.learning_rate * np.sum(errors)
    
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
        
        if os.path.isfile(file_path):  # Check if the path is a file
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

# Initialize and train the perceptrons with different optimizers
perceptron_gd = Perceptron(num_features=X_train.shape[1], optimizer='gradient_descent', learning_rate=0.1)
perceptron_gd.train(X_train, y_train, X_val, y_val, epochs=10)

perceptron_sgd = Perceptron(num_features=X_train.shape[1], optimizer='sgd', learning_rate=0.1)
perceptron_sgd.train(X_train, y_train, X_val, y_val, epochs=10)

perceptron_mini_batch = Perceptron(num_features=X_train.shape[1], optimizer='mini_batch', learning_rate=0.1, batch_size=32)
perceptron_mini_batch.train(X_train, y_train, X_val, y_val, epochs=10)

# Evaluate the perceptrons on training, validation, and testing data
train_accuracy_gd = perceptron_gd.evaluate(X_train, y_train)
val_accuracy_gd = perceptron_gd.evaluate(X_val, y_val)
test_accuracy_gd = perceptron_gd.evaluate(X_test, y_test)

train_accuracy_sgd = perceptron_sgd.evaluate(X_train, y_train)
val_accuracy_sgd = perceptron_sgd.evaluate(X_val, y_val)
test_accuracy_sgd = perceptron_sgd.evaluate(X_test, y_test)

train_accuracy_mini_batch = perceptron_mini_batch.evaluate(X_train, y_train)
val_accuracy_mini_batch = perceptron_mini_batch.evaluate(X_val, y_val)
test_accuracy_mini_batch = perceptron_mini_batch.evaluate(X_test, y_test)

# Print results
print("Performance with Gradient Descent:")
print(f"Training Accuracy: {train_accuracy_gd:.0%}")
print(f"Validation Accuracy: {val_accuracy_gd:.0%}")
print(f"Testing Accuracy: {test_accuracy_gd:.0%}")

print("\nPerformance with Stochastic Gradient Descent:")
print(f"Training Accuracy: {train_accuracy_sgd:.0%}")
print(f"Validation Accuracy: {val_accuracy_sgd:.0%}")
print(f"Testing Accuracy: {test_accuracy_sgd:.0%}")

print("\nPerformance with Mini-batch Gradient Descent (batch size = 42):")
print(f"Training Accuracy: {train_accuracy_mini_batch:.0%}")
print(f"Validation Accuracy: {val_accuracy_mini_batch:.0%}")
print(f"Testing Accuracy: {test_accuracy_mini_batch:.0%}")
