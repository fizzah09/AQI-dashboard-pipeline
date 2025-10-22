
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)


class ModelExplainer:
    def __init__(self, model, feature_names: List[str]):

        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = None
        self.lime_explainer = None
    
    def initialize_shap(self, X_background: pd.DataFrame, max_samples: int = 100):
        
        try:
            import shap
            
            log.info("Initializing SHAP explainer...")
            
            # Sample background data if too large
            if len(X_background) > max_samples:
                X_background = X_background.sample(n=max_samples, random_state=42)
            
            # Create SHAP explainer (works with XGBoost)
            self.shap_explainer = shap.TreeExplainer(self.model.model)
            
            log.info("✓ SHAP explainer initialized")
            
        except ImportError:
            log.warning("SHAP not installed. Run: pip install shap")
        except Exception as e:
            log.error(f"Failed to initialize SHAP: {e}")
    
    def initialize_lime(self, X_train: pd.DataFrame):
       
        try:
            from lime import lime_tabular
            
            log.info("Initializing LIME explainer...")
            
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=self.feature_names,
                mode='regression',
                random_state=42
            )
            
            log.info("LIME explainer initialized")
            
        except ImportError:
            log.warning("LIME not installed. Run: pip install lime")
        except Exception as e:
            log.error(f"Failed to initialize LIME: {e}")
    
    def explain_prediction_shap(self, X: pd.DataFrame, max_display: int = 20) -> Dict[str, Any]:
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_shap() first.")
        
        try:
            import shap
            
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(X)
            
            # Convert to DataFrame for easier handling
            if len(X) == 1:
                # Single prediction
                shap_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'shap_value': shap_values[0],
                    'feature_value': X.iloc[0].values
                })
                shap_df['abs_shap'] = shap_df['shap_value'].abs()
                shap_df = shap_df.sort_values('abs_shap', ascending=False).head(max_display)
            else:
                # Multiple predictions
                shap_df = pd.DataFrame(
                    shap_values,
                    columns=self.feature_names
                )
            
            return {
                'shap_values': shap_values,
                'shap_df': shap_df,
                'base_value': self.shap_explainer.expected_value,
                'feature_names': self.feature_names
            }
            
        except Exception as e:
            log.error(f"SHAP explanation failed: {e}")
            raise
    
    def explain_prediction_lime(self, X: pd.DataFrame, num_features: int = 10) -> Dict[str, Any]:
        """
        Explain prediction using LIME
        
        Args:
            X: Input features (single row only)
            num_features: Number of top features to show
            
        Returns:
            Dictionary with LIME explanation
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Call initialize_lime() first.")
        
        if len(X) != 1:
            raise ValueError("LIME explanation works with single instance only")
        
        try:
            # Get LIME explanation
            exp = self.lime_explainer.explain_instance(
                data_row=X.iloc[0].values,
                predict_fn=lambda x: self.model.predict(pd.DataFrame(x, columns=self.feature_names)),
                num_features=num_features
            )
            
            # Extract feature importance
            lime_values = exp.as_list()
            lime_df = pd.DataFrame(lime_values, columns=['feature', 'importance'])
            
            return {
                'explanation': exp,
                'lime_df': lime_df,
                'prediction': exp.predicted_value,
                'local_pred': exp.local_pred
            }
            
        except Exception as e:
            log.error(f"LIME explanation failed: {e}")
            raise
    
    def plot_shap_waterfall(self, shap_result: Dict[str, Any], instance_idx: int = 0, 
                           output_path: Optional[str] = None):
        """
        Create SHAP waterfall plot
        
        Args:
            shap_result: Result from explain_prediction_shap()
            instance_idx: Index of instance to plot
            output_path: Path to save plot
        """
        try:
            import shap
            
            plt.figure(figsize=(10, 6))
            
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_result['shap_values'][instance_idx],
                    base_values=shap_result['base_value'],
                    data=shap_result['shap_df'].iloc[instance_idx] if len(shap_result['shap_df']) > 1 else shap_result['shap_df']['feature_value'],
                    feature_names=self.feature_names
                ),
                show=False
            )
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                log.info(f"Waterfall plot saved: {output_path}")
            
            return plt.gcf()
            
        except Exception as e:
            log.error(f"Failed to create waterfall plot: {e}")
            raise
    
    def plot_shap_bar(self, shap_result: Dict[str, Any], output_path: Optional[str] = None):
        """
        Create SHAP bar plot showing feature importance
        
        Args:
            shap_result: Result from explain_prediction_shap()
            output_path: Path to save plot
        """
        try:
            shap_df = shap_result['shap_df']
            
            plt.figure(figsize=(10, 8))
            
            # Plot horizontal bar chart
            plt.barh(
                shap_df['feature'][::-1], 
                shap_df['shap_value'][::-1],
                color=['red' if x < 0 else 'green' for x in shap_df['shap_value'][::-1]]
            )
            
            plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                log.info(f"SHAP bar plot saved: {output_path}")
            
            return plt.gcf()
            
        except Exception as e:
            log.error(f"Failed to create SHAP bar plot: {e}")
            raise
    
    def plot_lime_explanation(self, lime_result: Dict[str, Any], output_path: Optional[str] = None):
        """
        Create LIME explanation plot
        
        Args:
            lime_result: Result from explain_prediction_lime()
            output_path: Path to save plot
        """
        try:
            lime_df = lime_result['lime_df']
            
            plt.figure(figsize=(10, 6))
            
            colors = ['green' if x > 0 else 'red' for x in lime_df['importance']]
            
            plt.barh(
                range(len(lime_df)),
                lime_df['importance'],
                color=colors
            )
            
            plt.yticks(range(len(lime_df)), lime_df['feature'])
            plt.xlabel('LIME Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title('LIME Feature Importance', fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                log.info(f"LIME plot saved: {output_path}")
            
            return plt.gcf()
            
        except Exception as e:
            log.error(f"Failed to create LIME plot: {e}")
            raise


def create_feature_importance_comparison(
    xgboost_importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
    output_path: Optional[str] = None
):
    """
    Compare XGBoost built-in importance with SHAP importance
    
    Args:
        xgboost_importance: XGBoost feature importance DataFrame
        shap_importance: SHAP importance DataFrame
        output_path: Path to save plot
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: XGBoost Importance
        top_features_xgb = xgboost_importance.head(15)
        axes[0].barh(
            top_features_xgb['feature'][::-1],
            top_features_xgb['importance'][::-1],
            color='steelblue'
        )
        axes[0].set_xlabel('XGBoost Importance', fontsize=12)
        axes[0].set_ylabel('Feature', fontsize=12)
        axes[0].set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Plot 2: SHAP Importance
        top_features_shap = shap_importance.head(15)
        axes[1].barh(
            top_features_shap['feature'][::-1],
            top_features_shap['abs_shap'][::-1],
            color='coral'
        )
        axes[1].set_xlabel('SHAP Importance (Mean |SHAP|)', fontsize=12)
        axes[1].set_title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            log.info(f"Comparison plot saved: {output_path}")
        
        return fig
        
    except Exception as e:
        log.error(f"Failed to create comparison plot: {e}")
        raise
