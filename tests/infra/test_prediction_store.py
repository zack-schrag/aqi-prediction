"""Unit tests for the prediction store implementation."""
import os
import shutil
import tempfile
from datetime import datetime
import numpy as np
import pytest
from infra.prediction_store import PredictionStore

class TestPredictionStore:
    @pytest.fixture
    def prediction_store(self):
        """Create a temporary prediction store for testing."""
        temp_dir = tempfile.mkdtemp()
        store = PredictionStore(store_path=temp_dir)
        yield store
        # Cleanup after tests
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def sample_prediction(self):
        """Create sample prediction data for testing."""
        return np.array([45.2, 52.1, 48.7])  # AQI predictions
        
    @pytest.fixture
    def prediction_metadata(self):
        """Create sample metadata for predictions."""
        return {
            "feature_key": "test_city_2024-01-01T00:00:00.json",
            "model_version": "v1.0.0"
        }

    def test_init_creates_directory(self):
        """Test that initialization creates the store directory."""
        temp_dir = tempfile.mkdtemp()
        test_path = os.path.join(temp_dir, "new_store")
        PredictionStore(store_path=test_path)
        assert os.path.exists(test_path)
        shutil.rmtree(temp_dir)

    def test_save_prediction(self, prediction_store, sample_prediction, prediction_metadata):
        """Test saving predictions to the store."""
        city = "test_city"
        prediction_key = prediction_store.save_prediction(
            city=city,
            prediction=sample_prediction,
            feature_key=prediction_metadata["feature_key"],
            model_version=prediction_metadata["model_version"]
        )
        
        # Verify key format
        assert city in prediction_key
        assert prediction_key.endswith(".json")
        
        # Verify file exists
        file_path = os.path.join(prediction_store.store_path, prediction_key)
        assert os.path.exists(file_path)

    def test_get_prediction(self, prediction_store, sample_prediction, prediction_metadata):
        """Test retrieving predictions by key."""
        city = "test_city"
        prediction_key = prediction_store.save_prediction(
            city=city,
            prediction=sample_prediction,
            feature_key=prediction_metadata["feature_key"],
            model_version=prediction_metadata["model_version"]
        )
        
        retrieved_prediction = prediction_store.get_prediction(prediction_key)
        
        # Verify metadata
        assert retrieved_prediction["metadata"]["city"] == city
        assert retrieved_prediction["metadata"]["feature_key"] == prediction_metadata["feature_key"]
        assert retrieved_prediction["metadata"]["model_version"] == prediction_metadata["model_version"]
        
        # Verify predictions
        np.testing.assert_array_almost_equal(
            np.array(retrieved_prediction["predictions"]),
            sample_prediction
        )

    def test_get_nonexistent_prediction(self, prediction_store):
        """Test retrieving prediction with invalid key."""
        assert prediction_store.get_prediction("nonexistent_key") is None

    def test_get_latest_prediction(self, prediction_store, sample_prediction, prediction_metadata):
        """Test retrieving latest prediction for a city."""
        city = "test_city"
        # Save predictions multiple times
        prediction_store.save_prediction(
            city=city,
            prediction=sample_prediction,
            feature_key=prediction_metadata["feature_key"],
            model_version=prediction_metadata["model_version"]
        )
        
        # Wait a moment to ensure different timestamps
        import time
        time.sleep(0.1)
        
        # Create newer prediction with different values
        newer_prediction = sample_prediction + 5.0  # Increase all predictions by 5
        prediction_store.save_prediction(
            city=city,
            prediction=newer_prediction,
            feature_key=prediction_metadata["feature_key"],
            model_version=prediction_metadata["model_version"]
        )
        
        latest_prediction = prediction_store.get_latest_prediction(city)
        np.testing.assert_array_almost_equal(
            np.array(latest_prediction["predictions"]),
            newer_prediction
        )

    def test_get_latest_prediction_no_data(self, prediction_store):
        """Test retrieving latest prediction for city with no data."""
        assert prediction_store.get_latest_prediction("nonexistent_city") is None

    def test_prediction_metadata_saved(self, prediction_store, sample_prediction, prediction_metadata):
        """Test that prediction metadata is correctly saved."""
        import json
        
        city = "test_city"
        prediction_key = prediction_store.save_prediction(
            city=city,
            prediction=sample_prediction,
            feature_key=prediction_metadata["feature_key"],
            model_version=prediction_metadata["model_version"]
        )
        
        file_path = os.path.join(prediction_store.store_path, prediction_key)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert data["metadata"]["city"] == city
        assert data["metadata"]["feature_key"] == prediction_metadata["feature_key"]
        assert data["metadata"]["model_version"] == prediction_metadata["model_version"]
        assert len(data["predictions"]) == len(sample_prediction)
