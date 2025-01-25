import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
from config import LEARNING_RATE, EPOCHS, PATIENCE , MODEL_SAVE_PATH

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

def preprocess_data(data, labels=None):
    """
    Preprocess the data: normalize features and split into train/validation sets.
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if labels is not None:
        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val, scaler
    return data, scaler

def predict_vulnerability(model, data, scaler):
    """
    Predict vulnerability likelihood using the trained model.
    """
    model.eval()
    with torch.no_grad():
        data = scaler.transform(data)  # Ensure the data is scaled
        data = torch.tensor(data, dtype=torch.float32)
        predictions = model(data)
        return predictions.numpy()

def predict_exploitation_likelihood(model, data, scaler):
    """
    Predict exploitation likelihood using the trained model.
    """
    return predict_vulnerability(model, data, scaler)  # Reuse the same prediction logic

#============================= MOCK ====================================#

def predict_vulnerability_mock(data):
    """
    Predict vulnerability likelihood using a Random Forest model.
    """
    # Mock implementation (replace with actual trained model)
    return np.random.rand(len(data))

def predict_exploitation_likelihood_mock(data):
    """
    Predict exploitation likelihood using a neural network.
    """
    # Mock implementation (replace with actual trained model)
    return np.random.rand(len(data))

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model on validation data.
    """
    model.eval()
    with torch.no_grad():
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        predictions = model(X_val)
        predictions = (predictions > 0.5).float()  # Convert probabilities to binary predictions

        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(y_val, predictions)
        recall = recall_score(y_val, predictions)
        f1 = f1_score(y_val, predictions)

        print(f"Validation Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

def train_model(data, labels):
    """
    Train the PyTorch model with early stopping and validation.
    """
    # Preprocess data
    X_train, X_val, y_train, y_val, scaler = preprocess_data(data, labels)

    input_size = X_train.shape[1]  # Dynamically set input size based on data
    model = AdvancedRiskPredictionModel(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        # Early stopping
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Load the best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    return model, scaler

def save_model(model, path):
    """
    Save the trained model to a file.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(input_size, path):
    """
    Load a trained model from a file.
    """
    model = AdvancedRiskPredictionModel(input_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model