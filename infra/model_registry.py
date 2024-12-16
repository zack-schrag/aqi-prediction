import os
import json
from datetime import datetime
from typing import Dict, Optional, List
import shutil


class ModelRegistry:
    def __init__(self, registry_path: str = "models/registry"):
        """Initialize model registry.

        Args:
            registry_path: Base directory for model registry
        """
        self.registry_path = registry_path
        self.metadata_file = os.path.join(registry_path, "metadata.json")

        # Create registry directory if it doesn't exist
        os.makedirs(registry_path, exist_ok=True)

        # Initialize or load metadata
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def register_model(
        self,
        model_path: str,
        scaler_path: str,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        stage: str = "dev",
    ) -> str:
        """Register a new model version.

        Args:
            model_path: Path to model weights
            scaler_path: Path to scaler
            model_name: Name of the model
            version: Version string (e.g., "1.0.0")
            metrics: Dictionary of evaluation metrics
            stage: One of ["dev", "staging", "prod"]

        Returns:
            Model ID string
        """
        # Generate model ID
        model_id = f"{model_name}_{version}"

        # Create model directory
        model_dir = os.path.join(self.registry_path, model_id)
        os.makedirs(model_dir, exist_ok=True)

        # Copy model artifacts
        shutil.copy2(model_path, os.path.join(model_dir, "model.pth"))
        shutil.copy2(scaler_path, os.path.join(model_dir, "scaler.pkl"))

        # Update metadata
        self.metadata[model_id] = {
            "name": model_name,
            "version": version,
            "stage": stage,
            "metrics": metrics,
            "registered_at": datetime.now().isoformat(),
            "artifacts": {"model": "model.pth", "scaler": "scaler.pkl"},
        }

        # Save metadata
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        return model_id

    def get_model_path(self, model_id: str) -> Optional[Dict[str, str]]:
        """Get paths to model artifacts.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with paths to model artifacts
        """
        if model_id not in self.metadata:
            return None

        model_dir = os.path.join(self.registry_path, model_id)
        return {
            "model": os.path.join(
                model_dir, self.metadata[model_id]["artifacts"]["model"]
            ),
            "scaler": os.path.join(
                model_dir, self.metadata[model_id]["artifacts"]["scaler"]
            ),
        }

    def get_latest_model(self, model_name: str, stage: str = "prod") -> Optional[str]:
        """Get the latest model version for a given stage.

        Args:
            model_name: Name of the model
            stage: One of ["dev", "staging", "prod"]

        Returns:
            Model ID of latest version
        """
        matching_models = [
            model_id
            for model_id, meta in self.metadata.items()
            if meta["name"] == model_name and meta["stage"] == stage
        ]

        if not matching_models:
            return None

        # Sort by registration time and return latest
        return max(matching_models, key=lambda x: self.metadata[x]["registered_at"])

    def transition_stage(self, model_id: str, new_stage: str):
        """Transition a model to a new stage.

        Args:
            model_id: Model identifier
            new_stage: New stage for the model
        """
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")

        if new_stage not in ["dev", "staging", "prod"]:
            raise ValueError(f"Invalid stage: {new_stage}")

        self.metadata[model_id]["stage"] = new_stage

        # Save metadata
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def list_models(
        self, model_name: Optional[str] = None, stage: Optional[str] = None
    ) -> List[str]:
        """List models filtered by name and/or stage.

        Args:
            model_name: Optional filter by model name
            stage: Optional filter by stage

        Returns:
            List of matching model IDs
        """
        matching_models = []

        for model_id, meta in self.metadata.items():
            if model_name and meta["name"] != model_name:
                continue
            if stage and meta["stage"] != stage:
                continue
            matching_models.append(model_id)

        return matching_models
