"""
Prediction Page - Interactive interface for making AQI predictions
"""
import streamlit as st
from dashboard.components.metrics import display_prediction_result


def show_prediction_interface(engine):
    """
    Display the prediction interface page
    
    This page allows users to:
    - Input weather features (temperature, humidity, pressure, wind speed)
    - Input pollutant features (PM2.5, PM10, NO2, O3, SO2, CO)
    - Get real-time AQI predictions
    - View AQI categorization and health alerts
    
    Args:
        engine (AQIInferenceEngine): Loaded model engine
    """
    st.header("🔮 Make Predictions")
    
    # Create three columns for input sliders
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Weather Features
    with col1:
        st.subheader("Weather Features")
        temp = st.slider("Temperature (°C)", -20.0, 50.0, 25.0, 0.1)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, 1.0)
        pressure = st.slider("Pressure (hPa)", 900.0, 1100.0, 1013.0, 1.0)
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0, 0.1)
    
    # Column 2: Primary Pollutants
    with col2:
        st.subheader("Pollutant Features")
        pm2_5 = st.slider("PM2.5 (μg/m³)", 0.0, 500.0, 50.0, 1.0)
        pm10 = st.slider("PM10 (μg/m³)", 0.0, 600.0, 80.0, 1.0)
        no2 = st.slider("NO2 (μg/m³)", 0.0, 400.0, 40.0, 1.0)
    
    # Column 3: Additional Pollutants
    with col3:
        st.subheader("Additional Pollutants")
        o3 = st.slider("O3 (μg/m³)", 0.0, 300.0, 60.0, 1.0)
        so2 = st.slider("SO2 (μg/m³)", 0.0, 200.0, 20.0, 1.0)
        co = st.slider("CO (μg/m³)", 0.0, 10000.0, 1000.0, 10.0)
    
    # Prediction button
    if st.button("🚀 Predict AQI", type="primary", use_container_width=True):
        # Prepare feature dictionary
        features = {
            'weather_temp': temp,
            'weather_humidity': humidity,
            'weather_pressure': pressure,
            'weather_wind_speed': wind_speed,
            'pollutant_pm2_5': pm2_5,
            'pollutant_pm10': pm10,
            'pollutant_no2': no2,
            'pollutant_o3': o3,
            'pollutant_so2': so2,
            'pollutant_co': co
        }
        
        try:
            # Check if engine is valid
            if engine is None:
                st.error("❌ Model not loaded. Please check model file exists.")
                return
            
            # Check if engine has predict_single method
            if not hasattr(engine, 'predict_single'):
                st.error("❌ Model engine does not have predict_single method")
                st.info(f"Engine type: {type(engine)}")
                return
            
            # Make prediction
            prediction = engine.predict_single(features)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            display_prediction_result(prediction)
            
        except AttributeError as e:
            st.error(f"❌ Model error: {str(e)}")
            st.info(
                "The model might not be loaded correctly. "
                "Please check:\n"
                "1. Model file exists in modeling/models/\n"
                "2. Model is a valid XGBoost model\n"
                "3. Run `python run_training.py` to retrain"
            )
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Please ensure all features are provided correctly.")
