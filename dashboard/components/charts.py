import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard.utils.config import AQI_THRESHOLDS
def plot_aqi_trend(df, target_col='pollutant_aqi'):
    if 'timestamp' not in df.columns:
        st.warning("No timestamp column found for trend analysis")
        return None
    
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp')
    
    # Create figure
    fig = go.Figure()
    
    # Add AQI line with area fill
    fig.add_trace(go.Scatter(
        x=df_sorted['timestamp'],
        y=df_sorted[target_col],
        mode='lines',
        name='AQI',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    # Add threshold reference lines
    for threshold, label, color in AQI_THRESHOLDS:
        fig.add_hline(
            y=threshold, 
            line_dash="dash", 
            line_color=color,
            annotation_text=label,
            annotation_position="right"
        )
    
    # Update layout
    fig.update_layout(
        title="AQI Trend Over Time",
        xaxis_title="Date",
        yaxis_title="AQI Value",
        hovermode='x unified',
        height=500,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig


def plot_pollutant_bars(df):
    """
    Create bar chart of average pollutant concentrations
    
    Args:
        df (pd.DataFrame): Input dataframe with pollutant columns
        
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    # Find pollutant columns
    pollutant_cols = [
        col for col in df.columns 
        if col.startswith('pollutant_') and col != 'pollutant_aqi'
    ]
    
    if not pollutant_cols:
        st.warning("No pollutant columns found")
        return None
    
    # Select only numeric pollutant columns
    try:
        numeric_pollutant_cols = df[pollutant_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_pollutant_cols:
            st.warning("No numeric pollutant columns found")
            return None
        
        # Calculate mean values and sort
        mean_values = df[numeric_pollutant_cols].mean().sort_values(ascending=False)
        
    except Exception as e:
        st.error(f"Error calculating pollutant means: {e}")
        return None
    
    # Clean column names for display
    display_names = [
        col.replace('pollutant_', '').replace('_', ' ').upper() 
        for col in mean_values.index
    ]
    
    # Create bar chart with color gradient
    fig = go.Figure(data=[
        go.Bar(
            x=display_names,
            y=mean_values.values,
            marker=dict(
                color=mean_values.values,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Concentration")
            ),
            text=[f"{val:.2f}" for val in mean_values.values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Average Pollutant Concentrations",
        xaxis_title="Pollutant Type",
        yaxis_title="Concentration (μg/m³)",
        height=500,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig
    
    fig.update_layout(
        title="Average Pollutant Concentrations",
        xaxis_title="Pollutant Type",
        yaxis_title="Concentration (μg/m³)",
        height=500,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def plot_scatter_analysis(df, x_col, y_col, color_col='pollutant_aqi'):

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_continuous_scale='RdYlGn_r',
        title=f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}",
        labels={
            x_col: x_col.replace('_', ' ').title(),
            y_col: y_col.replace('_', ' ').title(),
            color_col: 'AQI'
        },
        hover_data=[color_col],
        trendline="ols",  # Ordinary Least Squares trendline
        height=500
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig


def create_correlation_heatmap(df):
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove timestamp-related columns
    numeric_cols = [
        col for col in numeric_cols 
        if 'timestamp' not in col.lower()
    ]
    
    # Calculate correlation matrix
    corr = df[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=600,
        template='plotly_white'
    )
    
    return fig


def plot_distribution(df, column):

    fig = px.histogram(
        df,
        x=column,
        nbins=50,
        title=f"{column.replace('_', ' ').title()} Distribution",
        labels={column: column.replace('_', ' ').title()},
        color_discrete_sequence=['#1f77b4']
    )
    
    fig.update_layout(template='plotly_white', height=400)
    
    return fig
