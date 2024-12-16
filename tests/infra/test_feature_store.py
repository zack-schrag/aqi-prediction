"""Unit tests for the feature store implementation."""
import os
import shutil
import tempfile
from datetime import datetime, timedelta
import pandas as pd
import pytest
from infra.feature_store import FeatureStore


class TestFeatureStore:
    @pytest.fixture
    def feature_store(self):
        """Create a temporary feature store for testing."""
        temp_dir = tempfile.mkdtemp()
        store = FeatureStore(store_path=temp_dir)
        yield store
        # Cleanup after tests
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def sample_features(self):
        """Create sample features spanning multiple days."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-03', freq='D')
        return pd.DataFrame({
            'date': dates.repeat(3),
            'temperature': [20.5, 21.0, 19.8] * 3,
            'humidity': [65, 70, 68] * 3,
            'wind_speed': [5.5, 4.8, 6.2] * 3,
            'pm25': [15.2, 18.4, 12.9] * 3
        })

    def test_init_creates_directories(self):
        """Test that initialization creates the store directories."""
        temp_dir = tempfile.mkdtemp()
        test_path = os.path.join(temp_dir, "new_store")
        store = FeatureStore(store_path=test_path)
        
        assert os.path.exists(os.path.join(test_path, "online"))
        assert os.path.exists(os.path.join(test_path, "offline"))
        shutil.rmtree(temp_dir)

    def test_save_features_offline(self, feature_store, sample_features):
        """Test saving features to offline store."""
        city = "test_city"
        saved_paths = feature_store.save_features(city, sample_features, is_online=False)
        
        # Should create one file per day
        assert len(saved_paths) == 3
        
        # Verify file structure and content
        for path in saved_paths:
            assert os.path.exists(path)
            assert path.startswith(os.path.join(feature_store.offline_path, city))
            
            with open(path, 'r') as f:
                import json
                data = json.load(f)
                assert "metadata" in data
                assert "features" in data
                assert data["metadata"]["city"] == city
                assert len(data["features"]) == 3  # 3 records per day

    def test_save_features_online(self, feature_store, sample_features):
        """Test saving features to online store."""
        city = "test_city"
        saved_paths = feature_store.save_features(city, sample_features, is_online=True)
        
        # Verify files are in online store
        for path in saved_paths:
            assert path.startswith(os.path.join(feature_store.online_path, city))
            assert os.path.exists(path)

    def test_get_features_date_range(self, feature_store, sample_features):
        """Test retrieving features for a date range."""
        city = "test_city"
        feature_store.save_features(city, sample_features, is_online=True)
        
        # Get features for first two days
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        features = feature_store.get_features(city, start_date, end_date)
        
        assert features is not None
        assert len(features) == 6  # 2 days * 3 records per day
        
        # Verify date range
        dates = pd.to_datetime(features['date'])
        assert dates.min().date() == start_date.date()
        assert dates.max().date() == end_date.date()

    def test_get_features_with_selection(self, feature_store, sample_features):
        """Test retrieving specific features."""
        city = "test_city"
        feature_store.save_features(city, sample_features, is_online=True)
        
        # Get only temperature and humidity
        features = feature_store.get_features(
            city,
            datetime(2024, 1, 1),
            datetime(2024, 1, 3),
            feature_names=['temperature', 'humidity']
        )
        
        assert features is not None
        assert set(features.columns) == {'temperature', 'humidity'}

    def test_get_nonexistent_features(self, feature_store):
        """Test retrieving features for nonexistent city/dates."""
        features = feature_store.get_features(
            "nonexistent_city",
            datetime(2024, 1, 1),
            datetime(2024, 1, 1)
        )
        assert features is None

    def test_get_latest_features(self, feature_store, sample_features):
        """Test retrieving latest features."""
        city = "test_city"
        
        # Save some older data offline
        older_features = sample_features.copy()
        older_features['date'] = older_features['date'] - timedelta(days=7)
        feature_store.save_features(city, older_features, is_online=False)
        
        # Save newer data online
        feature_store.save_features(city, sample_features, is_online=True)
        
        # Get latest features
        latest = feature_store.get_latest_features(city)
        assert latest is not None
        
        # Should get the most recent date from online store
        dates = pd.to_datetime(latest['date'])
        assert dates.max().date() == datetime(2024, 1, 3).date()

    def test_get_latest_features_specific_columns(self, feature_store, sample_features):
        """Test retrieving latest features with specific columns."""
        city = "test_city"
        feature_store.save_features(city, sample_features, is_online=True)
        
        latest = feature_store.get_latest_features(
            city,
            feature_names=['temperature', 'humidity']
        )
        assert latest is not None
        assert set(latest.columns) == {'temperature', 'humidity'}

    def test_online_offline_precedence(self, feature_store, sample_features):
        """Test that online store takes precedence over offline."""
        city = "test_city"
        
        # Save same date range to both stores with different values
        offline_features = sample_features.copy()
        offline_features['temperature'] += 100  # Make offline values distinctly different
        
        feature_store.save_features(city, offline_features, is_online=False)
        feature_store.save_features(city, sample_features, is_online=True)
        
        # Get features for a date that exists in both stores
        features = feature_store.get_features(
            city,
            datetime(2024, 1, 1),
            datetime(2024, 1, 1)
        )
        
        # Should get online values
        assert features is not None
        assert features['temperature'].iloc[0] == sample_features['temperature'].iloc[0]
