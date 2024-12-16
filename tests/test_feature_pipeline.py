import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from feature_pipeline import process_features, fetch_weather_data
from infra.feature_store import FeatureStore

@pytest.fixture
def mock_feature_store(tmp_path):
    """Create a temporary feature store for testing."""
    return FeatureStore(store_path=str(tmp_path / "feature_store"))

@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing."""
    return pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=3),
        'tavg': [20.0, 21.0, 22.0],
        'prcp': [0.0, 0.1, 0.0],
        'wspd': [5.0, 6.0, 5.5],
        'pres': [1013.0, 1012.0, 1014.0],
        'wdir': [180, 190, 185]
    })

@pytest.fixture
def sample_aqi_data():
    """Create sample AQI data for testing."""
    return pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=3),
        ' pm25': [10.0, 12.0, 11.0],
        ' co': [0.5, 0.6, 0.5],
        ' no2': [20.0, 22.0, 21.0],
        ' o3': [30.0, 32.0, 31.0],
        ' so2': [5.0, 5.5, 5.2]
    })

def test_process_features(sample_aqi_data):
    """Test feature processing logic."""
    geo = [47.6062, -122.3321]  # Seattle coordinates
    features = process_features(sample_aqi_data, geo)
    
    # Check that all expected columns are present
    expected_columns = [
        'date', ' pm25', ' co', ' no2', 'tavg', 'prcp', 'wspd',
        'pres', 'wdir', 'day_of_week', 'month', ' o3', ' so2'
    ]
    assert all(col in features.columns for col in expected_columns)
    
    # Check that derived features are correct
    assert len(features) == 3
    assert 'day_of_week' in features.columns
    assert 'month' in features.columns
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(features['date'])
    assert pd.api.types.is_numeric_dtype(features[' pm25'])
    assert pd.api.types.is_numeric_dtype(features['day_of_week'])
    assert pd.api.types.is_numeric_dtype(features['month'])

def test_feature_store_integration(mock_feature_store, sample_aqi_data):
    """Test feature store integration."""
    city = "test_city"
    geo = [47.6062, -122.3321]  # Seattle coordinates
    features = process_features(sample_aqi_data, geo)
    
    # Ensure date column is datetime
    features["date"] = pd.to_datetime(features["date"])
    
    # Save features
    mock_feature_store.save_features(city, features)
    
    # Retrieve features
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 3)
    retrieved_features = mock_feature_store.get_features(
        city,
        start_date=start_date,
        end_date=end_date
    )
    
    # Ensure retrieved features have datetime index
    retrieved_features["date"] = pd.to_datetime(retrieved_features["date"])
    
    # Sort both dataframes by date for comparison
    features = features.sort_values("date").reset_index(drop=True)
    retrieved_features = retrieved_features.sort_values("date").reset_index(drop=True)
    
    # Check that retrieved features match saved features
    pd.testing.assert_frame_equal(
        features,
        retrieved_features,
        check_dtype=False  # Ignore dtype differences from storage
    )

def test_fetch_weather_data():
    """Test weather data fetching."""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 3)
    geo = [47.6062, -122.3321]  # Seattle coordinates
    
    weather_data = fetch_weather_data(start_date, end_date, geo)
    
    assert isinstance(weather_data, pd.DataFrame)
    assert not weather_data.empty
    assert all(col in weather_data.columns for col in ['tavg', 'prcp', 'wspd', 'pres', 'wdir'])
    assert pd.api.types.is_datetime64_any_dtype(weather_data.index)
