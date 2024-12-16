"""
Prediction Store implementation.
Stores predictions with references to input features.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

class PredictionStore:
    def __init__(self, store_path: str = "data/prediction_store"):
        """Initialize the prediction store.
        
        Args:
            store_path: Directory to store predictions
        """
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        
    def _get_prediction_key(self, city: str, timestamp: datetime) -> str:
        """Generate a unique key for prediction reference."""
        return f"{city}_{timestamp.isoformat()}.json"
    
    def save_prediction(self, city: str, prediction: np.ndarray, feature_key: Dict[str, str], model_version: str) -> str:
        """Save prediction to store and return reference key.
        
        Args:
            city: City name
            prediction: Numpy array of predictions
            feature_key: Dictionary containing feature window metadata (start_date, end_date, window_size)
            model_version: Model version used for prediction
            
        Returns:
            Key to reference this prediction
        """
        timestamp = datetime.now()
        key = self._get_prediction_key(city, timestamp)
        
        # Create prediction data
        data = {
            "metadata": {
                "city": city,
                "timestamp": timestamp.isoformat(),
                "feature_key": feature_key,
                "model_version": model_version
            },
            "predictions": prediction.tolist()
        }
        
        # Save to JSON file
        file_path = os.path.join(self.store_path, key)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        return key
    
    def get_prediction(self, prediction_key: str) -> Optional[Dict]:
        """Retrieve prediction by key.
        
        Args:
            prediction_key: Prediction reference key
            
        Returns:
            Dictionary containing prediction data or None if not found
        """
        file_path = os.path.join(self.store_path, prediction_key)
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, 'r') as f:
            return json.load(f, parse_float=lambda x: float(x))
    
    def get_latest_prediction(self, city: str) -> Optional[Dict]:
        """Get the most recent prediction for a city.
        
        Args:
            city: City name
            
        Returns:
            Most recent prediction or None if not found
        """
        # List all prediction files for the city
        files = [f for f in os.listdir(self.store_path) if f.startswith(city)]
        if not files:
            return None
            
        # Get most recent file
        latest_file = max(files)
        with open(os.path.join(self.store_path, latest_file), 'r') as f:
            return json.load(f, parse_float=lambda x: float(x))
