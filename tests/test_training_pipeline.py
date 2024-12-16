import pytest
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from training_pipeline import (
    AQIDataset, LSTMModel, prepare_data, train_model, evaluate_predictions
)
from infra.feature_store import FeatureStore
from infra.model_registry import ModelRegistry

@pytest.fixture
def mock_model_registry(tmp_path):
    """Create a temporary model registry for testing."""
    return ModelRegistry(registry_path=str(tmp_path / "model_registry"))

@pytest.fixture
def mock_feature_store(tmp_path):
    """Create a temporary feature store for testing."""
    return FeatureStore(store_path=str(tmp_path / "feature_store"))

@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    dates = pd.date_range(start='2024-01-01', periods=30)
    data = pd.DataFrame({
        'date': dates,
        ' pm25': np.random.normal(20, 5, 30),
        ' co': np.random.normal(0.5, 0.1, 30),
        ' no2': np.random.normal(20, 5, 30),
        'tavg': np.random.normal(20, 3, 30),
        'prcp': np.random.uniform(0, 1, 30),
        'wspd': np.random.normal(5, 1, 30),
        'pres': np.random.normal(1013, 2, 30),
        'wdir': np.random.uniform(0, 360, 30),
        'day_of_week': [d.weekday() for d in dates],
        'month': [d.month for d in dates],
        ' o3': np.random.normal(30, 5, 30),
        ' so2': np.random.normal(5, 1, 30)
    })
    return data

def test_aqi_dataset(sample_training_data):
    """Test AQI dataset creation and functionality."""
    dataset = AQIDataset(sample_training_data, sequence_length=7, prediction_length=3)
    
    # Check dataset size
    expected_sequences = len(sample_training_data) - dataset.sequence_length - dataset.prediction_length + 1
    assert len(dataset) == expected_sequences
    
    # Check sequence shape
    sequence, target = dataset[0]
    assert sequence.shape == (7, 12)  # 7 days, 12 features
    assert target.shape == (3,)  # 3 days of PM2.5 predictions
    
    # Check that sequences are continuous
    assert torch.all(sequence[1:] == dataset[1][0][:-1])

def test_lstm_model():
    """Test LSTM model architecture and forward pass."""
    model = LSTMModel(input_size=12, hidden_size=64, num_layers=2, output_size=3)
    
    # Test model architecture
    assert isinstance(model.lstm, torch.nn.LSTM)
    assert model.lstm.input_size == 12
    assert model.lstm.hidden_size == 64
    assert model.lstm.num_layers == 2
    
    # Test forward pass
    batch_size = 16
    seq_length = 7
    x = torch.randn(batch_size, seq_length, 12)
    output = model(x)
    assert output.shape == (batch_size, 3)  # 3 days of predictions

def test_prepare_data(mock_feature_store, sample_training_data):
    """Test data preparation function."""
    city = "test_city"
    
    # Save sample data to feature store
    mock_feature_store.save_features(city, sample_training_data)
    
    # Update feature store with data in the correct date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 30)
    mock_feature_store.save_features(city, sample_training_data)
    
    # Prepare data
    train_loader, val_loader, scaler = prepare_data(
        city,
        batch_size=32,
        feature_store=mock_feature_store,
        start_date=start_date,
        end_date=end_date
    )
    
    # Check data loaders
    assert isinstance(train_loader.dataset, AQIDataset)
    assert isinstance(val_loader.dataset, AQIDataset)
    assert scaler is not None
    
    # Check batch shapes
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.dim() == 3  # (batch_size, sequence_length, features)
    assert y_batch.dim() == 2  # (batch_size, prediction_length)

def test_model_training(mock_model_registry, sample_training_data):
    """Test model training process."""
    # Create dataset and data loader
    dataset = AQIDataset(sample_training_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # Initialize model and training components
    model = LSTMModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train for a few epochs
    train_losses, val_losses = train_model(
        train_loader, val_loader, model, criterion, optimizer, num_epochs=2
    )
    
    # Check training results
    assert len(train_losses) == 2  # One loss per epoch
    assert len(val_losses) == 2
    assert all(isinstance(loss, float) for loss in train_losses)
    assert all(isinstance(loss, float) for loss in val_losses)

def test_model_evaluation(sample_training_data):
    """Test model evaluation metrics."""
    # Create a simple model and dataset
    model = LSTMModel()
    dataset = AQIDataset(sample_training_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # Create a dummy scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = sample_training_data[[' pm25', ' co', ' no2', 'tavg', 'prcp', 'wspd',
                                   'pres', 'wdir', 'day_of_week', 'month', ' o3', ' so2']]
    scaler.fit(features)
    
    # Evaluate predictions
    metrics = evaluate_predictions(model, loader, scaler)
    
    # Check metrics - metrics should be a dictionary
    assert isinstance(metrics, dict)
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'rmse' in metrics
    
    # Check metric values are reasonable
    assert metrics['mse'] >= 0
    assert metrics['mae'] >= 0
    assert metrics['rmse'] >= 0
