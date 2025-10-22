"""
Dashboard Overview Page - Main landing page with key metrics and visualizations
"""
import streamlit as st
from dashboard.components.charts import plot_aqi_trend, plot_pollutant_bars
from dashboard.components.metrics import display_aqi_metrics
from dashboard.utils.data_loader import get_target_column


def show_dashboard_overview(df):
    """
    Display the main dashboard overview page
    
    This page shows:
    - Key AQI metrics (average, max, min, hazardous days)
    - AQI trend over time with threshold lines
    - Average pollutant concentrations
    
    Args:
        df (pd.DataFrame): Historical data for visualization
    """
    st.header("📊 Dashboard Overview")
    
    if df is None:
        st.warning("No data available for display")
        return
    
    # Get the target column name
    target_col = get_target_column(df)
    
    # Display key metrics
    display_aqi_metrics(df, target_col)
    
    st.markdown("---")
    
    # AQI Trend Chart
    st.subheader("📈 AQI Trend Over Time")
    fig_trend = plot_aqi_trend(df, target_col)
    if fig_trend:
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Pollutant Concentrations Bar Chart
    st.subheader("📊 Pollutant Concentrations")
    fig_bars = plot_pollutant_bars(df)
    if fig_bars:
        st.plotly_chart(fig_bars, use_container_width=True)
