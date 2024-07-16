#Create a multi-layer perceptron (MLP) for digit classification (MNIST dataset)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define transforms for preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load data from folders
train_data = datasets.MNIST('/home/rekha-habshipuram/Downloads/archive/train_data', download=True, train=True, transform=transform)
validation_data = datasets.MNIST('/home/rekha-habshipuram/Downloads/archive/test_dat', download=True, train=False, transform=transform)
test_data = datasets.MNIST('/home/rekha-habshipuram/Downloads/archive/val_data', download=True, train=False, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=False)

# Multi-layer perceptron (MLP) model
mlp = nn.Sequential(
    nn.Linear(784, 128),  # input layer (28x28) -> hidden layer (128)
    nn.ReLU(),
    nn.Linear(128, 64),  # hidden layer (128) -> hidden layer (64)
    nn.ReLU(),
    nn.Linear(64, 10)  # hidden layer (64) -> output layer (10)
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)

# Train the network
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.view(-1, 784)
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/i}')

# Evaluate on training set
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        inputs, labels = data
        inputs = inputs.view(-1, 784)
        outputs = mlp(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Training Accuracy: {100 * correct / total} %')

# Evaluate on test set
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.view(-1, 784)
        outputs = mlp(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {100 * correct / total} %')

# Evaluate on validation set
correct = 0
total = 0
with torch.no_grad():
    for data in validation_loader:
        inputs, labels = data
        inputs = inputs.view(-1, 784)
        outputs = mlp(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Validation Accuracy: {100 * correct / total} %')