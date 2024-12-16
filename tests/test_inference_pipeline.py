import pytest
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from inference_pipeline import AQIPredictor
from infra.feature_store import FeatureStore
from infra.prediction_store import PredictionStore
from infra.model_registry import ModelRegistry
from training_pipeline import LSTMModel

@pytest.fixture
def mock_stores(tmp_path):
    """Create temporary stores for testing."""
    feature_store = FeatureStore(store_path=str(tmp_path / "feature_store"))
    prediction_store = PredictionStore(store_path=str(tmp_path / "prediction_store"))
    model_registry = ModelRegistry(registry_path=str(tmp_path / "model_registry"))
    return feature_store, prediction_store, model_registry

@pytest.fixture
def sample_model():
    """Create and save a sample model."""
    model = LSTMModel()
    return model

@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    dates = pd.date_range(end=datetime.now(), periods=10)
    data = pd.DataFrame({
        'date': dates,
        ' pm25': np.random.normal(20, 5, 10),
        ' co': np.random.normal(0.5, 0.1, 10),
        ' no2': np.random.normal(20, 5, 10),
        'tavg': np.random.normal(20, 3, 10),
        'prcp': np.random.uniform(0, 1, 10),
        'wspd': np.random.normal(5, 1, 10),
        'pres': np.random.normal(1013, 2, 10),
        'wdir': np.random.uniform(0, 360, 10),
        'day_of_week': [d.weekday() for d in dates],
        'month': [d.month for d in dates],
        ' o3': np.random.normal(30, 5, 10),
        ' so2': np.random.normal(5, 1, 10)
    })
    return data

def setup_predictor(mock_stores, sample_model, sample_features, tmp_path):
    """Helper function to set up a predictor with test data."""
    feature_store, prediction_store, model_registry = mock_stores
    city = "test_city"
    
    # Save features
    feature_store.save_features(city, sample_features)
    
    # Save model and scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = sample_features[[' pm25', ' co', ' no2', 'tavg', 'prcp', 'wspd', 
                              'pres', 'wdir', 'day_of_week', 'month', ' o3', ' so2']]
    scaler.fit(features)
    
    model_path = tmp_path / "model.pt"
    scaler_path = tmp_path / "scaler.pkl"
    
    # Save model and scaler
    torch.save(sample_model.state_dict(), model_path)
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Register model
    model_id = model_registry.register_model(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        model_name=f"aqi_lstm_{city}",
        version="test",
        metrics={"test_metric": 0.5},
        stage="prod"
    )
    
    # Create predictor
    predictor = AQIPredictor(
        city=city,
        feature_store=feature_store,
        prediction_store=prediction_store,
        model_registry=model_registry
    )
    
    return predictor, model_id

def test_predictor_initialization(mock_stores, sample_model, sample_features, tmp_path):
    """Test predictor initialization."""
    predictor, model_id = setup_predictor(mock_stores, sample_model, sample_features, tmp_path)
    
    assert predictor.city == "test_city"
    assert predictor.model_id == model_id
    assert isinstance(predictor.model, LSTMModel)
    assert predictor.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_get_latest_features(mock_stores, sample_model, sample_features, tmp_path):
    """Test feature retrieval."""
    predictor, _ = setup_predictor(mock_stores, sample_model, sample_features, tmp_path)
    
    features, start_date, end_date = predictor.get_latest_features()
    
    assert isinstance(features, pd.DataFrame)
    assert len(features) == predictor.sequence_length
    assert isinstance(start_date, datetime)
    assert isinstance(end_date, datetime)
    assert end_date > start_date
    assert (end_date - start_date).days == predictor.sequence_length - 1

def test_prepare_input(mock_stores, sample_model, sample_features, tmp_path):
    """Test input preparation."""
    predictor, _ = setup_predictor(mock_stores, sample_model, sample_features, tmp_path)
    
    features, _, _ = predictor.get_latest_features()
    x = predictor.prepare_input(features)
    
    assert isinstance(x, torch.Tensor)
    assert x.dim() == 3  # (batch_size, sequence_length, features)
    assert x.shape[0] == 1  # batch size
    assert x.shape[1] == predictor.sequence_length
    assert x.shape[2] == 12  # number of features

def test_prediction_workflow(mock_stores, sample_model, sample_features, tmp_path):
    """Test end-to-end prediction workflow."""
    predictor, _ = setup_predictor(mock_stores, sample_model, sample_features, tmp_path)
    
    # Make prediction
    result = predictor.predict()
    
    # Check prediction results
    assert "results" in result
    assert "dates" in result["results"]
    assert "pm25_predictions" in result["results"]
    assert len(result["results"]["dates"]) == len(result["results"]["pm25_predictions"])
    assert all(isinstance(pred, float) for pred in result["results"]["pm25_predictions"])
    
    # Check metadata
    assert "metadata" in result
    metadata = result["metadata"]
    assert metadata is not None
    assert metadata["city"] == predictor.city
    assert metadata["model_id"] == predictor.model_id
    assert isinstance(metadata["feature_date"], str)
    assert isinstance(metadata["prediction_date"], str)

def test_prediction_store_integration(mock_stores, sample_model, sample_features, tmp_path):
    """Test prediction storage and retrieval."""
    predictor, _ = setup_predictor(mock_stores, sample_model, sample_features, tmp_path)
    
    # Make and store prediction
    result = predictor.predict()
    
    # Retrieve stored prediction
    feature_store, prediction_store, _ = mock_stores
    stored_prediction = prediction_store.get_prediction(result["prediction_key"])
    
    assert stored_prediction is not None
    assert "metadata" in stored_prediction
    assert "predictions" in stored_prediction
    assert len(stored_prediction["predictions"]) == len(result["results"]["pm25_predictions"])
