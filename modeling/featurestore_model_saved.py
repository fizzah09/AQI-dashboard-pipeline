"""
Model Registry integration for storing trained models
"""
import hopsworks
import joblib
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import json

log = logging.getLogger(__name__)


def register_model_to_hopsworks(
    model_path: str,
    model_name: str,
    metrics: Dict[str, Any],
    feature_names: list,
    target_name: str,
    description: Optional[str] = None,
    api_key: Optional[str] = None
) -> None:
    """
    Register a trained model to Hopsworks Model Registry
    
    Args:
        model_path: Path to the saved .pkl model file
        model_name: Name for the model in registry
        metrics: Dictionary of evaluation metrics (rmse, mae, r2, etc.)
        feature_names: List of feature column names
        target_name: Name of the target variable
        description: Optional model description
        api_key: Hopsworks API key (optional if using config)
    """
    try:
        log.info(f"Connecting to Hopsworks...")
        
        # Login to Hopsworks
        if api_key:
            project = hopsworks.login(api_key_value=api_key)
        else:
            project = hopsworks.login()
        
        log.info(f"Connected to project: {project.name}")
        
        # Get Model Registry
        mr = project.get_model_registry()
        
        # Load the model to get metadata
        model = joblib.load(model_path)
        
        # Create input/output schema
        from hsml.schema import Schema
        from hsml.model_schema import ModelSchema
        
        input_schema = Schema(feature_names)
        output_schema = Schema([target_name])
        
        model_schema = ModelSchema(
            input_schema=input_schema,
            output_schema=output_schema
        )
        
        # Prepare model description
        if description is None:
            description = f"XGBoost model for {target_name} prediction trained on {len(feature_names)} features"
        
        # Create model in registry
        log.info(f"Registering model: {model_name}")
        
        registered_model = mr.python.create_model(
            name=model_name,
            metrics=metrics,
            model_schema=model_schema,
            description=description,
            input_example=None  # Optional: add a sample input
        )
        
        # Save model artifacts
        model_dir = Path(model_path).parent
        registered_model.save(str(model_dir))
        
        log.info(f"✓ Model registered successfully!")
        log.info(f"  Model name: {model_name}")
        log.info(f"  Version: {registered_model.version}")
        log.info(f"  Metrics: {metrics}")
        
        return registered_model
        
    except Exception as e:
        log.error(f"Failed to register model: {e}")
        raise


def load_model_from_registry(
    model_name: str,
    version: Optional[int] = None
):
    """
    Load a model from Hopsworks Model Registry
    
    Args:
        model_name: Name of the model in registry
        version: Specific version to load (None = latest)
    
    Returns:
        Loaded model object
    """
    try:
        log.info(f"Loading model from registry: {model_name}")
        
        project = hopsworks.login()
        mr = project.get_model_registry()
        
        # Get model
        if version:
            model = mr.get_model(model_name, version=version)
        else:
            model = mr.get_model(model_name)
        
        # Download model directory
        model_dir = model.download()
        
        # Load the actual model file
        model_path = Path(model_dir) / f"{model_name}.pkl"
        loaded_model = joblib.load(model_path)
        
        log.info(f"✓ Model loaded: {model_name} v{model.version}")
        
        return loaded_model, model
        
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise


def list_registered_models():
    """List all models in the Model Registry"""
    try:
        project = hopsworks.login()
        mr = project.get_model_registry()
        
        models = mr.get_models()
        
        print("\n" + "="*70)
        print("REGISTERED MODELS")
        print("="*70)
        
        for model in models:
            print(f"\nModel: {model.name}")
            print(f"  Version: {model.version}")
            print(f"  Created: {model.created}")
            if hasattr(model, 'metrics'):
                print(f"  Metrics: {model.metrics}")
        
        print("="*70)
        
        return models
        
    except Exception as e:
        log.error(f"Failed to list models: {e}")
        raise