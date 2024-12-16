import pytest
import os
import shutil
import tempfile
import json
from infra.model_registry import ModelRegistry

@pytest.fixture
def registry_setup():
    """Setup test fixtures."""
    # Create a temporary directory for the registry
    test_dir = tempfile.mkdtemp()
    registry = ModelRegistry(registry_path=test_dir)
    
    # Create temporary model files
    model_file = os.path.join(test_dir, "temp_model.pth")
    scaler_file = os.path.join(test_dir, "temp_scaler.pkl")
    
    # Create dummy files
    with open(model_file, 'w') as f:
        f.write("dummy model")
    with open(scaler_file, 'w') as f:
        f.write("dummy scaler")
        
    yield registry, test_dir, model_file, scaler_file
    
    # Cleanup
    shutil.rmtree(test_dir)

def test_register_model(registry_setup):
    """Test registering a new model."""
    registry, test_dir, model_file, scaler_file = registry_setup
    
    model_id = registry.register_model(
        model_path=model_file,
        scaler_path=scaler_file,
        model_name="test_model",
        version="1.0.0",
        metrics={"accuracy": 0.95},
        stage="dev"
    )
    
    # Check model ID format
    assert model_id == "test_model_1.0.0"
    
    # Check metadata was saved
    assert os.path.exists(os.path.join(test_dir, "metadata.json"))
    
    # Check model files were copied
    model_dir = os.path.join(test_dir, model_id)
    assert os.path.exists(os.path.join(model_dir, "model.pth"))
    assert os.path.exists(os.path.join(model_dir, "scaler.pkl"))
    
    # Check metadata content
    with open(os.path.join(test_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
        
    assert model_id in metadata
    assert metadata[model_id]["name"] == "test_model"
    assert metadata[model_id]["version"] == "1.0.0"
    assert metadata[model_id]["stage"] == "dev"
    assert metadata[model_id]["metrics"] == {"accuracy": 0.95}

def test_get_model_path(registry_setup):
    """Test retrieving model artifact paths."""
    registry, test_dir, model_file, scaler_file = registry_setup
    
    # Register a model
    model_id = registry.register_model(
        model_path=model_file,
        scaler_path=scaler_file,
        model_name="test_model",
        version="1.0.0",
        metrics={"accuracy": 0.95}
    )
    
    # Get paths
    paths = registry.get_model_path(model_id)
    
    # Check paths exist and are correct
    assert paths is not None
    assert os.path.exists(paths["model"])
    assert os.path.exists(paths["scaler"])
    
    # Test non-existent model
    paths = registry.get_model_path("non_existent_model")
    assert paths is None

def test_get_latest_model(registry_setup):
    """Test getting the latest model version."""
    registry, test_dir, model_file, scaler_file = registry_setup
    
    # Register multiple versions
    versions = ["1.0.0", "1.1.0", "2.0.0"]
    for version in versions:
        registry.register_model(
            model_path=model_file,
            scaler_path=scaler_file,
            model_name="test_model",
            version=version,
            metrics={"accuracy": 0.95},
            stage="prod"
        )
        
    # Get latest
    latest = registry.get_latest_model("test_model", stage="prod")
    
    # Should return the most recently registered version
    assert latest == "test_model_2.0.0"
    
    # Test non-existent model/stage
    latest = registry.get_latest_model("non_existent_model")
    assert latest is None

def test_transition_stage(registry_setup):
    """Test transitioning model between stages."""
    registry, test_dir, model_file, scaler_file = registry_setup
    
    # Register a model
    model_id = registry.register_model(
        model_path=model_file,
        scaler_path=scaler_file,
        model_name="test_model",
        version="1.0.0",
        metrics={"accuracy": 0.95},
        stage="dev"
    )
    
    # Transition to staging
    registry.transition_stage(model_id, "staging")
    
    # Check stage was updated
    with open(os.path.join(test_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    assert metadata[model_id]["stage"] == "staging"
    
    # Test invalid stage
    with pytest.raises(ValueError):
        registry.transition_stage(model_id, "invalid_stage")
        
    # Test non-existent model
    with pytest.raises(ValueError):
        registry.transition_stage("non_existent_model", "prod")

def test_list_models(registry_setup):
    """Test listing models with filters."""
    registry, test_dir, model_file, scaler_file = registry_setup
    
    # Register multiple models
    models = [
        ("model1", "1.0.0", "dev"),
        ("model1", "1.1.0", "staging"),
        ("model2", "1.0.0", "prod")
    ]
    
    for name, version, stage in models:
        registry.register_model(
            model_path=model_file,
            scaler_path=scaler_file,
            model_name=name,
            version=version,
            metrics={"accuracy": 0.95},
            stage=stage
        )
        
    # Test listing all models
    all_models = registry.list_models()
    assert len(all_models) == 3
    
    # Test filtering by name
    model1_versions = registry.list_models(model_name="model1")
    assert len(model1_versions) == 2
    
    # Test filtering by stage
    prod_models = registry.list_models(stage="prod")
    assert len(prod_models) == 1
    assert prod_models[0] == "model2_1.0.0"
    
    # Test filtering by both name and stage
    filtered = registry.list_models(model_name="model1", stage="staging")
    assert len(filtered) == 1
    assert filtered[0] == "model1_1.1.0"
