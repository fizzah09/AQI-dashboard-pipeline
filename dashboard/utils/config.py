"""
Dashboard configuration and styling
"""
import streamlit as st

# Page configuration
PAGE_CONFIG = {
    "page_title": "AQI Dashboard",
    "page_icon": "⛅",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Custom CSS
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .alert-danger {
        background-color: #ff4444;
        color: white;
    }
    .alert-warning {
        background-color: #ffaa00;
        color: black;
    }
    .alert-success {
        background-color: #00cc44;
        color: white;
    }
</style>
"""

# AQI thresholds for visualizations
AQI_THRESHOLDS = [
    (50, 'Good', '#00e400'),
    (100, 'Moderate', '#ffff00'),
    (150, 'Unhealthy (Sensitive)', '#ff7e00'),
    (200, 'Unhealthy', '#ff0000'),
    (300, 'Very Unhealthy', '#8f3f97')
]


def apply_page_config():
    """Apply Streamlit page configuration"""
    st.set_page_config(**PAGE_CONFIG)


def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
