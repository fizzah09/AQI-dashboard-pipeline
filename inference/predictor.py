"""
Model Inference Pipeline
Loads trained models and makes predictions
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

log = logging.getLogger(__name__)


class AQIInferenceEngine:
    """
    Inference engine for AQI predictions
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model file (.pkl)
        """
        self.model = None
        self.model_path = model_path
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from file"""
        try:
            log.info(f"Loading model from: {model_path}")
            
            # Load the predictor object
            self.model = joblib.load(model_path)
            
            # Get feature names if available
            if hasattr(self.model, 'feature_names_'):
                self.feature_names = self.model.feature_names_
            
            log.info(f"✓ Model loaded successfully")
            
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise
    
    def load_from_registry(self, model_name: str, version: Optional[int] = None):
        """Load model from Hopsworks Model Registry"""
        try:
            from modeling.model_registry import load_model_from_registry
            
            log.info(f"Loading model from registry: {model_name}")
            self.model, metadata = load_model_from_registry(model_name, version)
            
            log.info(f"✓ Model loaded from registry: v{metadata.version}")
            
        except Exception as e:
            log.error(f"Failed to load from registry: {e}")
            raise
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            predictions = self.model.predict(features)
            return predictions
            
        except Exception as e:
            log.error(f"Prediction failed: {e}")
            raise
    
    def predict_single(self, feature_dict: Dict[str, Any]) -> float:
        """
        Make single prediction from dictionary
        
        Args:
            feature_dict: Dictionary of feature name -> value
            
        Returns:
            Single prediction value
        """
        df = pd.DataFrame([feature_dict])
        prediction = self.predict(df)
        return float(prediction[0])
    
    def predict_with_uncertainty(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimates
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        predictions = self.predict(features)
        
        # TODO: Implement proper uncertainty quantification
        # For now, use simple heuristic based on model performance
        uncertainty = np.abs(predictions) * 0.1  # 10% uncertainty
        
        return {
            'predictions': predictions,
            'lower_bound': predictions - uncertainty,
            'upper_bound': predictions + uncertainty,
            'uncertainty': uncertainty
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if hasattr(self.model, 'get_feature_importance'):
            return self.model.get_feature_importance()
        else:
            raise NotImplementedError("Model doesn't support feature importance")


def categorize_aqi(aqi_value: float) -> Dict[str, Any]:
    """
    Categorize AQI value and provide health recommendations
    
    Args:
        aqi_value: AQI value
        
    Returns:
        Dictionary with category, color, and health message
    """
    if aqi_value <= 50:
        return {
            'category': 'Good',
            'level': 1,
            'color': '#00e400',
            'message': 'Air quality is satisfactory, and air pollution poses little or no risk.',
            'is_hazardous': False
        }
    elif aqi_value <= 100:
        return {
            'category': 'Moderate',
            'level': 2,
            'color': '#ffff00',
            'message': 'Air quality is acceptable. However, there may be a risk for some people.',
            'is_hazardous': False
        }
    elif aqi_value <= 150:
        return {
            'category': 'Unhealthy for Sensitive Groups',
            'level': 3,
            'color': '#ff7e00',
            'message': 'Members of sensitive groups may experience health effects.',
            'is_hazardous': False
        }
    elif aqi_value <= 200:
        return {
            'category': 'Unhealthy',
            'level': 4,
            'color': '#ff0000',
            'message': 'Some members of the general public may experience health effects.',
            'is_hazardous': True
        }
    elif aqi_value <= 300:
        return {
            'category': 'Very Unhealthy',
            'level': 5,
            'color': '#8f3f97',
            'message': 'Health alert: The risk of health effects is increased for everyone.',
            'is_hazardous': True
        }
    else:
        return {
            'category': 'Hazardous',
            'level': 6,
            'color': '#7e0023',
            'message': '⚠️ HEALTH WARNING: Everyone may experience serious health effects.',
            'is_hazardous': True
        }


def generate_aqi_alert(aqi_value: float) -> Optional[str]:
    """
    Generate alert message for hazardous AQI levels
    
    Args:
        aqi_value: AQI value
        
    Returns:
        Alert message if hazardous, None otherwise
    """
    info = categorize_aqi(aqi_value)
    
    if info['is_hazardous']:
        return f"🚨 ALERT: Air quality is {info['category']} (AQI: {aqi_value:.0f}). {info['message']}"
    
    return None
