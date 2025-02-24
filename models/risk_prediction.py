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
        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)
        
        # Main layers with residual connections
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
        # Improved activation and regularization
        self.dropout = nn.Dropout(0.4)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initial normalization
        x = self.batch_norm1(x)
        
        # First block with residual
        identity = x
        x = self.fc1(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Second block
        x = self.fc2(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Final layers
        x = self.fc3(x)
        x = self.leaky_relu(x)
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
    Enhanced training function with improved optimization and monitoring
    """
    X_train, X_val, y_train, y_val, scaler = preprocess_data(data, labels)
    
    input_size = X_train.shape[1]
    model = AdvancedRiskPredictionModel(input_size)
    
    # Use weighted BCE loss for imbalanced datasets
    pos_weight = torch.tensor([1.0])  # Adjust based on class distribution
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Advanced optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
        amsgrad=True
    )
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    best_val_f1 = 0.0
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': []
    }

    for epoch in range(EPOCHS):
        model.train()
        # Mini-batch training
        batch_size = 32
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            
            # Calculate F1 score
            val_preds = (val_outputs > 0.5).float()
            val_f1 = f1_score(y_val_tensor.numpy(), val_preds.numpy())
            
            # Store metrics
            training_history['train_loss'].append(loss.item())
            training_history['val_loss'].append(val_loss.item())
            training_history['val_f1'].append(val_f1)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        print(f"Val F1: {val_f1:.4f}")

        # Improved early stopping based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'scaler': scaler
            }, MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break
        
        scheduler.step()

    # Load best model
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Return validation data as numpy arrays
    return model, scaler, training_history, X_val, y_val


def predict_with_uncertainty(model, data, scaler, num_samples=10):
    """
    Make predictions with uncertainty estimation using Monte Carlo Dropout
    """
    model.train()  # Enable dropout for uncertainty estimation
    predictions = []
    
    with torch.no_grad():
        data = scaler.transform(data)
        data = torch.tensor(data, dtype=torch.float32)
        
        # Multiple forward passes with dropout
        for _ in range(num_samples):
            pred = model(data)
            predictions.append(pred.numpy())
    
    # Calculate mean and standard deviation
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred


def evaluate_model_comprehensive(model, X_val, y_val, scaler):
    """
    Comprehensive model evaluation with multiple metrics
    """
    model.eval()
    with torch.no_grad():
        X_val_scaled = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        
        # Get predictions
        predictions = model(X_val_scaled)
        pred_binary = (predictions > 0.5).float()
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, pred_binary)
        precision = precision_score(y_val, pred_binary)
        recall = recall_score(y_val, pred_binary)
        f1 = f1_score(y_val, pred_binary)
        
        # ROC-AUC and PR-AUC
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc_auc = roc_auc_score(y_val, predictions.numpy())
        pr_auc = average_precision_score(y_val, predictions.numpy())
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
        return metrics


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