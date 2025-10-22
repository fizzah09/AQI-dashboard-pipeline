"""
Explainability Page - Model interpretability using SHAP and LIME
"""
import streamlit as st
import plotly.express as px


def show_explainability(engine, df):
    """
    Display the model explainability page
    
    This page provides:
    - SHAP (SHapley Additive exPlanations) analysis
    - LIME (Local Interpretable Model-agnostic Explanations) analysis
    - Feature importance visualization
    
    Args:
        engine (AQIInferenceEngine): Loaded model engine
        df (pd.DataFrame): Historical data
    """
    st.header("🔍 Model Explainability")
    
    st.info(
        "Model explainability helps understand **why** the model makes certain predictions. "
        "SHAP and LIME are two popular techniques for interpreting machine learning models."
    )
    
    # Create tabs for different explainability methods
    tab1, tab2 = st.tabs(["SHAP Analysis", "LIME Analysis"])
    
    # Tab 1: SHAP Analysis
    with tab1:
        st.subheader("SHAP (SHapley Additive exPlanations)")
        
        st.markdown("""
        **What is SHAP?**
        
        SHAP values show how each feature contributes to the prediction:
        - **Positive SHAP value**: Feature pushes prediction higher
        - **Negative SHAP value**: Feature pushes prediction lower
        - **Magnitude**: How much impact the feature has
        """)
        
        try:
            # Check if engine has model and it's an XGBoost model
            if hasattr(engine, 'model') and hasattr(engine.model, 'get_booster'):
                # XGBoost feature importance
                import xgboost as xgb
                booster = engine.model.get_booster()
                importance_dict = booster.get_score(importance_type='weight')
                
                # Convert to dataframe
                import pandas as pd
                importance_df = pd.DataFrame([
                    {'feature': k, 'importance': v} 
                    for k, v in importance_dict.items()
                ]).sort_values('importance', ascending=False)
                
                # Create horizontal bar plot
                fig = px.bar(
                    importance_df.head(20),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 20 Features by Importance",
                    labels={'importance': 'Feature Importance', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    height=600, 
                    showlegend=False, 
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top features table
                st.markdown("#### Top 10 Most Important Features")
                st.dataframe(
                    importance_df.head(10),
                    use_container_width=True,
                    hide_index=True
                )
                
            else:
                raise AttributeError("Model does not support feature importance extraction")
            
        except Exception as e:
            st.warning(f"⚠️ Feature importance not available: {str(e)}")
            st.info(
                "**To enable SHAP analysis:**\n\n"
                "1. Ensure your model is properly trained\n"
                "2. Install SHAP: `pip install shap`\n"
                "3. Model must be XGBoost or support feature importance"
            )
    
    # Tab 2: LIME Analysis
    with tab2:
        st.subheader("LIME (Local Interpretable Model-agnostic Explanations)")
        
        st.markdown("""
        **What is LIME?**
        
        LIME explains individual predictions by:
        1. Creating perturbed samples around the prediction
        2. Training a simple interpretable model locally
        3. Showing which features contributed most to that specific prediction
        
        **Use Case**: Understanding why a specific prediction was made
        """)
        
        st.info(
            "LIME explanations are generated for individual predictions. "
            "Go to the **Predictions** page to make a prediction, "
            "then return here to see LIME explanations for that prediction."
        )
        
        # Installation instructions
        st.markdown("#### Setup Instructions")
        st.code("pip install lime", language="bash")
        
        # Example usage
        with st.expander("📖 How to use LIME"):
            st.markdown("""
            1. Navigate to the **Predictions** page
            2. Input feature values and click "Predict AQI"
            3. The LIME explainer will show:
               - Which features increased/decreased the prediction
               - By how much each feature contributed
               - A local approximation of the model's behavior
            """)
