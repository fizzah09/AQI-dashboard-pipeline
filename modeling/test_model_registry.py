"""
Example script to test model registry functionality
"""
import logging
from modeling.model_registry import list_registered_models, load_model_from_registry

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def test_list_models():
    """Test listing all registered models"""
    log.info("Testing model listing...")
    try:
        models = list_registered_models()
        log.info(f"Found {len(models)} models in registry")
    except Exception as e:
        log.error(f"Error listing models: {e}")


def test_load_model(model_name: str, version: int = None):
    """Test loading a specific model"""
    log.info(f"Testing model loading: {model_name}")
    try:
        model, metadata = load_model_from_registry(model_name, version)
        log.info(f"Successfully loaded model: {model_name}")
        log.info(f"Model version: {metadata.version}")
        log.info(f"Model type: {type(model)}")
        return model, metadata
    except Exception as e:
        log.error(f"Error loading model: {e}")
        return None, None


if __name__ == "__main__":
    print("="*70)
    print("MODEL REGISTRY TEST")
    print("="*70)
    
    # Test 1: List all models
    print("\n1. Listing all registered models...")
    test_list_models()
    
    # Test 2: Load a specific model (example)
    print("\n2. Loading a specific model...")
    model_name = "aqi_pollutant_aqi_regression"  # Example model name
    
    try:
        model, metadata = test_load_model(model_name)
        if model:
            print(f"\n✓ Model loaded successfully!")
            print(f"  Name: {metadata.name}")
            print(f"  Version: {metadata.version}")
    except Exception as e:
        print(f"\n⚠ Note: To load a model, first train and register one using:")
        print(f"  python modeling/train_pipeline.py --data data/ml_training_data_1year.csv")
    
    print("\n" + "="*70)
