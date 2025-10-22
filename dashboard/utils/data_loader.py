"""
Data loading utilities for the dashboard
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference.predictor import AQIInferenceEngine


@st.cache_resource
def load_model():
    """
    Load the trained XGBoost model
    
    Returns:
        AQIInferenceEngine: Loaded model engine or None if loading fails
    """
    try:
        model_dir = Path(__file__).parent.parent.parent / "modeling" / "models"
        model_files = list(model_dir.glob("*_xgboost.pkl"))
        
        if model_files:
            # Get the most recently modified model
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            engine = AQIInferenceEngine(str(latest_model))
            return engine
        else:
            st.error("No trained model found. Please train a model first.")
            return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_data
def load_data():
    """
    Load historical training data for analysis
    
    Returns:
        pd.DataFrame: Historical data or None if loading fails
    """
    try:
        data_path = Path(__file__).parent.parent.parent / "data" / "ml_training_data_1year.csv"
        
        if not data_path.exists():
            st.warning("No historical data found.")
            return None
        
        # Load CSV
        df = pd.read_csv(data_path)
        
        # Convert timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            # Create synthetic timestamps if none exist
            df['timestamp'] = pd.date_range(
                end=datetime.now(), 
                periods=len(df), 
                freq='H'
            )
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None


def get_target_column(df):
    """
    Identify the target column (AQI) in the dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        str: Name of the target column
    """
    if 'pollutant_aqi' in df.columns:
        return 'pollutant_aqi'
    else:
        # Fallback to last column
        return df.columns[-1]
