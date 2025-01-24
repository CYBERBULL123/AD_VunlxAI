# models/risk_prediction.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from config import LEARNING_RATE, EPOCHS, PATIENCE
import numpy as np

class AdvancedRiskPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(AdvancedRiskPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

def predict_vulnerability(data):
    """
    Predict vulnerability likelihood using a Random Forest model.
    """
    # Mock implementation (replace with actual trained model)
    return np.random.rand(len(data))

def predict_exploitation_likelihood(data):
    """
    Predict exploitation likelihood using a neural network.
    """
    # Mock implementation (replace with actual trained model)
    return np.random.rand(len(data))

def train_model(data, labels):
    """
    Train the PyTorch model.
    """
    input_size = data.shape[1]  # Dynamically set input size based on data
    model = AdvancedRiskPredictionModel(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return model