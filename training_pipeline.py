import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from infra.feature_store import FeatureStore
from infra.model_registry import ModelRegistry
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

class AQIDataset(Dataset):
    def __init__(self, data, sequence_length=7, prediction_length=3):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # Select features for input
        feature_columns = [' pm25', ' co', ' no2', 'tavg', 'prcp', 'wspd', 'pres', 'wdir', 
                         'day_of_week', 'month', ' o3', ' so2']
        self.data = data[feature_columns].values
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(self.data) - sequence_length - prediction_length + 1):
            seq = self.data[i:i+sequence_length]
            target = self.data[i+sequence_length:i+sequence_length+prediction_length, 0]  # Only predict PM2.5
            self.sequences.append(seq)
            self.targets.append(target)
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

def prepare_data(city: str, batch_size=32, feature_store=None, start_date=None, end_date=None):
    """Prepare training data from feature store.
    
    Args:
        city (str): City name
        batch_size (int): Batch size for data loaders
        feature_store (FeatureStore, optional): Feature store instance
        start_date (datetime, optional): Start date for training data
        end_date (datetime, optional): End date for training data
    
    Returns:
        tuple: Training and validation data loaders
    """
    if feature_store is None:
        feature_store = FeatureStore()
    
    # Define feature columns
    FEATURE_COLUMNS = [" pm25", " co", " no2", "tavg", "prcp", "wspd", "pres", "wdir", 
                      "day_of_week", "month", " o3", " so2"]
    
    # Get features from store
    features_df = feature_store.get_features(city, start_date=start_date, end_date=end_date, feature_names=FEATURE_COLUMNS)
    
    if features_df is None:
        raise ValueError(f"No features found for {city}")
    
    # Scale all numeric features
    scaler = MinMaxScaler()
    features_df[FEATURE_COLUMNS] = scaler.fit_transform(features_df[FEATURE_COLUMNS])
    
    # Store feature names with scaler
    scaler.feature_names_ = FEATURE_COLUMNS
    
    # Split into train and validation sets (80-20)
    train_size = int(0.8 * len(features_df))
    train_data = features_df[:train_size]
    val_data = features_df[train_size:]
    
    # Create datasets
    train_dataset = AQIDataset(train_data)
    val_dataset = AQIDataset(val_data)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler

class LSTMModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        last_time_step = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(last_time_step)
        
        return out.squeeze(-1)  # Remove last dimension if size 1

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze(-1))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze(-1))
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_predictions(model, loader, scaler):
    """Evaluate model predictions.
    
    Args:
        model (LSTMModel): Model to evaluate
        loader (DataLoader): Data loader with validation data
        scaler (MinMaxScaler): Scaler used to transform features
        
    Returns:
        dict: Dictionary containing evaluation metrics (mse, mae, rmse)
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X, y in loader:
            # Forward pass
            y_pred = model(X)
            
            # Store predictions and actuals
            predictions.extend(y_pred.numpy())
            actuals.extend(y.numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    
    print("\nModel Performance Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

def plot_results(train_losses, val_losses, predictions, actuals):
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot 2: Predictions vs Actuals
    plt.subplot(1, 2, 2)
    # Plot first 50 predictions for visibility
    n_samples = 50
    x = np.arange(n_samples)
    plt.plot(x, actuals[:n_samples, 0, 0], label='Actual', marker='o')
    plt.plot(x, predictions[:n_samples, 0, 0], label='Predicted', marker='x')
    plt.title('Predictions vs Actuals (First day prediction)')
    plt.xlabel('Sample')
    plt.ylabel('PM2.5')
    plt.legend()
    
    plt.tight_layout()

if __name__ == "__main__":
    import pickle
    import os
    
    # Train for both cities
    for CITY in ["seattle", "bellevue"]:
        print(f"\n{'='*50}")
        print(f"Training model for {CITY}")
        print(f"{'='*50}\n")
        
        MODEL_NAME = f"aqi_lstm_{CITY}"
        VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data preparation
        print(f"Preparing data for {CITY}...")
        train_loader, val_loader, scaler = prepare_data(CITY, start_date=datetime(2019, 1, 1), end_date=datetime(2023, 12, 31))
        
        # Model initialization
        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        print("Training model...")
        train_losses, val_losses = train_model(train_loader, val_loader, model, criterion, optimizer)
        
        # Evaluation
        print("Evaluating model...")
        metrics = evaluate_predictions(model, val_loader, scaler)
        
        # Calculate metrics for model registry
        metrics = {
            "mse": float(metrics['mse']),
            "mae": float(metrics['mae']),
            "rmse": float(metrics['rmse']),
            "val_loss": float(val_losses[-1])
        }
        
        # Save model and scaler
        os.makedirs("models/tmp", exist_ok=True)
        model_path = f"models/tmp/model_{VERSION}.pth"
        scaler_path = f"models/tmp/scaler_{VERSION}.pkl"
        
        torch.save(model.state_dict(), model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        
        # Register model
        print("Registering model...")
        registry = ModelRegistry()
        model_id = registry.register_model(
            model_path=model_path,
            scaler_path=scaler_path,
            model_name=MODEL_NAME,
            version=VERSION,
            metrics=metrics,
            stage="dev"  # New models start in dev stage
        )
        
        print(f"Model registered with ID: {model_id}")
        print(f"Metrics: {metrics}")
        
        # Save plot with city name
        plt.savefig(f'training_results_{CITY}.png')
        plt.close()
