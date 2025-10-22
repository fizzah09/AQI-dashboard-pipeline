import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from dashboard.utils.config import apply_page_config, apply_custom_css
from dashboard.utils.data_loader import load_model, load_data
from dashboard.pages.dashboard_overview import show_dashboard_overview
from dashboard.pages.prediction_page import show_prediction_interface
from dashboard.pages.eda_page import show_eda_analysis
from dashboard.pages.explainability_page import show_explainability
def main():
    apply_page_config()
    apply_custom_css()
    st.markdown(
        '<h1 class="main-header">🌍 AQI Prediction Dashboard</h1>', 
        unsafe_allow_html=True
    )
    st.markdown("**Real-time Air Quality Index Monitoring and Prediction System**")
    
    # Load model and data with spinner
    with st.spinner("Loading model and data..."):
        engine = load_model()
        df = load_data()
    
    # Check if model is loaded
    if engine is None:
        st.error(
            "⚠️ Model not loaded. Please train a model first using "
            "`python run_training.py`"
        )
        st.info("Run: `python run_training.py` to train a model")
        return
    
    # Check if data is loaded
    if df is None:
        st.warning("⚠️ No historical data available for analysis")
        st.info("Some features may be limited without historical data")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Overview", 
        "🔮 Make Predictions", 
        "📈 EDA & Trends", 
        "🔍 Model Explainability"
    ])
    
    with tab1:
        show_dashboard_overview(df)
    
    with tab2:
        show_prediction_interface(engine)
    
    with tab3:
        show_eda_analysis(df)
    
    with tab4:
        show_explainability(engine, df)


if __name__ == "__main__":
    main()

