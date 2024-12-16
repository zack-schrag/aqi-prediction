"""
Feature Store implementation.
Stores and manages feature data with versioning and timestamps.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd


class FeatureStore:
    def __init__(self, store_path: str = "data/feature_store"):
        """Initialize the feature store.
        
        Args:
            store_path: Directory to store feature data
        """
        self.store_path = store_path
        self.offline_path = os.path.join(store_path, "offline")
        self.online_path = os.path.join(store_path, "online")
        os.makedirs(self.offline_path, exist_ok=True)
        os.makedirs(self.online_path, exist_ok=True)
        
    def _get_feature_path(self, city: str, date: datetime, is_online: bool = False) -> str:
        """Generate path for feature storage."""
        base_path = self.online_path if is_online else self.offline_path
        year_month = date.strftime("%Y-%m")
        return os.path.join(base_path, city, year_month, f"{date.strftime('%Y-%m-%d')}.json")
        
    def save_features(
        self,
        city: str,
        features: pd.DataFrame,
        is_online: bool = False
    ) -> List[str]:
        """Save features to store.
        
        Args:
            city: City name
            features: DataFrame containing feature values
            is_online: Whether to save to online store
            
        Returns:
            List of paths where features were saved
        """
        saved_paths = []
        features = features.copy()
        
        # Group features by date
        features['date'] = pd.to_datetime(features['date'])
        for date, day_features in features.groupby(features['date'].dt.date):
            # Convert datetime to midnight UTC for consistent storage
            storage_date = datetime.combine(date, datetime.min.time())
            feature_path = self._get_feature_path(city, storage_date, is_online)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
            
            # Convert datetime columns to strings
            day_features_copy = day_features.copy()
            for col in day_features_copy.select_dtypes(include=['datetime64']):
                day_features_copy[col] = day_features_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
            feature_dict = {
                "metadata": {
                    "city": city,
                    "date": storage_date.isoformat(),
                    "feature_names": list(day_features.columns),
                    "last_updated": datetime.now().isoformat()
                },
                "features": day_features_copy.to_dict(orient='records')
            }
            
            with open(feature_path, 'w') as f:
                json.dump(feature_dict, f, indent=2)
            saved_paths.append(feature_path)
            
        return saved_paths
    
    def get_features(
        self,
        city: str,
        start_date: datetime,
        end_date: datetime,
        feature_names: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """Get features for a city within a date range.
        
        Args:
            city: City name
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            feature_names: Optional list of specific features to retrieve
            
        Returns:
            DataFrame containing features or None if not found
        """
        features_list = []
        current_date = start_date
        
        while current_date <= end_date:
            # Try online store first, then offline
            for is_online in [True, False]:
                feature_path = self._get_feature_path(city, current_date, is_online)
                if os.path.exists(feature_path):
                    with open(feature_path, 'r') as f:
                        feature_dict = json.load(f)
                        day_features = pd.DataFrame.from_records(feature_dict["features"])
                        features_list.append(day_features)
                    break
            current_date += timedelta(days=1)
            
        if not features_list:
            return None
            
        # Combine all features and select requested columns
        features = pd.concat(features_list, ignore_index=True)
        if feature_names:
            features = features[feature_names]
            
        return features
    
    def get_latest_features(
        self,
        city: str,
        feature_names: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """Get most recent features for a city.
        
        Args:
            city: City name
            feature_names: Optional list of specific features to retrieve
            
        Returns:
            DataFrame containing latest features or None if not found
        """
        # Look in online store first
        online_city_path = os.path.join(self.online_path, city)
        if os.path.exists(online_city_path):
            # Find most recent date
            all_dates = []
            for year_month in os.listdir(online_city_path):
                month_path = os.path.join(online_city_path, year_month)
                if os.path.isdir(month_path):
                    all_dates.extend(
                        [f.replace('.json', '') for f in os.listdir(month_path)]
                    )
            
            if all_dates:
                latest_date = datetime.strptime(max(all_dates), '%Y-%m-%d')
                return self.get_features(
                    city,
                    latest_date,
                    latest_date,
                    feature_names
                )
        
        return None
