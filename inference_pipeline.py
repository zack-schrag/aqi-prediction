import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import pickle

from infra.feature_store import FeatureStore
from infra.prediction_store import PredictionStore
from infra.model_registry import ModelRegistry
from training_pipeline import LSTMModel  # Import the model class


class AQIPredictor:
    def __init__(
        self,
        city: str,
        model_stage: str = "prod",
        feature_store: Optional[FeatureStore] = None,
        prediction_store: Optional[PredictionStore] = None,
        model_registry: Optional[ModelRegistry] = None,
        sequence_length: int = 7
    ):
        """Initialize the AQI predictor.
        
        Args:
            city: City to make predictions for
            model_stage: Model stage to use (prod, staging, dev)
            feature_store: Optional feature store instance
            prediction_store: Optional prediction store instance
            model_registry: Optional model registry instance
            sequence_length: Number of days of features to use for prediction
        """
        self.city = city
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        
        # Initialize stores
        self.feature_store = feature_store or FeatureStore()
        self.prediction_store = prediction_store or PredictionStore()
        self.model_registry = model_registry or ModelRegistry()
        
        # Get latest model for city and stage
        model_name = f"aqi_lstm_{city}"
        model_id = self.model_registry.get_latest_model(model_name, stage=model_stage)
        if not model_id:
            raise ValueError(f"No {model_stage} model found for {city}")
            
        # Load model artifacts
        artifact_paths = self.model_registry.get_model_path(model_id)
        if not artifact_paths:
            raise ValueError(f"Model artifacts not found for {model_id}")
            
        # Initialize and load model
        self.model = LSTMModel()
        self.model.load_state_dict(
            torch.load(artifact_paths["model"], map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler
        with open(artifact_paths["scaler"], "rb") as f:
            self.scaler = pickle.load(f)
            
        self.model_id = model_id
        
    def get_latest_features(self) -> Tuple[pd.DataFrame, datetime, datetime]:
        """Get the most recent features for prediction.
        
        Returns:
            Tuple of (features DataFrame, start_date, end_date) where dates
            indicate the feature window used for prediction.
        """
        # Get features from the last 30 days to ensure we have enough data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Get feature names from scaler
        feature_names = getattr(self.scaler, 'feature_names_', [
            " pm25", " co", " no2", "tavg", "prcp", "wspd", 
            "pres", "wdir", "day_of_week", "month", " o3", " so2"
        ])
        
        # Add date column to features to fetch
        features_to_fetch = ["date"] + feature_names
        
        features = self.feature_store.get_features(
            self.city,
            start_date=start_date,
            end_date=end_date,
            feature_names=features_to_fetch
        )
        
        if len(features) < self.sequence_length:
            raise ValueError(
                f"Not enough features available. Need {self.sequence_length} days "
                f"but only found {len(features)} days."
            )
        
        # Ensure date column is datetime
        features["date"] = pd.to_datetime(features["date"])
        
        # Get the last sequence_length days of features
        features = features.sort_values("date").tail(self.sequence_length)
        feature_start_date = pd.to_datetime(features.date.min())
        feature_end_date = pd.to_datetime(features.date.max())
        
        return features, feature_start_date, feature_end_date
        
    def prepare_input(self, features: pd.DataFrame) -> torch.Tensor:
        """Prepare input data for model prediction."""
        # Get feature names from scaler
        feature_names = getattr(self.scaler, 'feature_names_', [
            " pm25", " co", " no2", "tavg", "prcp", "wspd", 
            "pres", "wdir", "day_of_week", "month", " o3", " so2"
        ])
        
        # Create DataFrame with proper feature names for scaling
        features_for_scaling = pd.DataFrame(
            features[feature_names].values,
            columns=feature_names
        )
        
        # Scale features using the same column order as training
        scaled_data = self.scaler.transform(features_for_scaling)
        
        # Convert to tensor
        x = torch.FloatTensor(scaled_data).unsqueeze(0)  # Add batch dimension
        x = x.to(self.device)
        return x
        
    def predict(self) -> Dict:
        """Make prediction using latest features.
        
        Returns:
            Dictionary containing prediction results and metadata
        """
        # Get latest features
        features, feature_start_date, feature_end_date = self.get_latest_features()
        
        # Prepare input
        x = self.prepare_input(features)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(x)
            prediction = prediction.cpu().numpy().reshape(-1)
        
        # Unscale predictions (only PM2.5)
        dummy = np.zeros((len(prediction), 12))  # 12 features
        dummy[:, 0] = prediction  # PM2.5 is first feature
        pred_unscaled = self.scaler.inverse_transform(dummy)[:, 0]
        
        # Generate prediction dates
        start_date = feature_end_date.date() + timedelta(days=1)
        prediction_dates = [
            (start_date + timedelta(days=i)).isoformat()
            for i in range(len(prediction))
        ]
        
        # Create prediction results
        results = {
            "dates": prediction_dates,
            "pm25_predictions": pred_unscaled.tolist()
        }
        
        # Save prediction with feature window metadata
        prediction_key = self.prediction_store.save_prediction(
            city=self.city,
            prediction=pred_unscaled,
            feature_key={
                "start_date": feature_start_date.isoformat(),
                "end_date": feature_end_date.isoformat(),
                "window_size": self.sequence_length
            },
            model_version=self.model_id
        )
        
        return {
            "prediction_key": prediction_key,
            "results": results,
            "metadata": {
                "city": self.city,
                "model_id": self.model_id,
                "feature_date": feature_end_date.isoformat(),
                "prediction_date": datetime.now().isoformat()
            }
        }


def make_predictions():
    """Make predictions for all cities using production models."""
    for city in ["seattle", "bellevue"]:
        print(f"\nMaking predictions for {city}")
        try:
            predictor = AQIPredictor(city, model_stage="dev")  # Use dev for now
            results = predictor.predict()
            
            print(f"Predictions saved with key: {results['prediction_key']}")
            print("\nPM2.5 Predictions:")
            for date, value in zip(results["results"]["dates"], 
                                 results["results"]["pm25_predictions"]):
                print(f"{date}: {value:.2f}")
                
        except Exception as e:
            print(f"Error making predictions for {city}: {str(e)}")


if __name__ == "__main__":
    make_predictions()
