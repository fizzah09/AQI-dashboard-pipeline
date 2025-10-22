import logging
import sys
import pandas as pd
import argparse
from pathlib import Path

# Import modeling functions
from modeling.data_loader import load_training_data, prepare_features_targets
from modeling.train_model import AQIPredictor, split_train_test
from modeling.evaluate import evaluate_model, plot_feature_importance

# Add this at the very top of main() function to fix Windows encoding
def setup_logging():
    """Setup logging with UTF-8 encoding for Windows"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Fix Windows console encoding
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')


log = logging.getLogger(__name__)


def main():
    setup_logging()  # Add this as first line
    
    parser = argparse.ArgumentParser(description="Train AQI prediction model")
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/ml_training_data_1year.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--target", 
        type=str, 
        default="pollutant_aqi",
        choices=["pollutant_aqi", "pollutant_pm2_5", "pollutant_pm10"],
        help="Target variable to predict"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="regression",
        choices=["regression", "classification"],
        help="Task type"
    )
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Test set proportion"
    )
    parser.add_argument(
        "--val-size", 
        type=float, 
        default=0.1,
        help="Validation set proportion"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="modeling",
        help="Output directory for models and evaluation"
    )
    
    args = parser.parse_args()
    
    log.info("="*70)
    log.info("AQI PREDICTION MODEL TRAINING PIPELINE")
    log.info("="*70)
    log.info("Configuration:")
    log.info("  Data:       %s", args.data)
    log.info("  Target:     %s", args.target)
    log.info("  Task:       %s", args.task)
    log.info("  Test size:  %.1f%%", args.test_size * 100)
    log.info("  Val size:   %.1f%%", args.val_size * 100)
    log.info("")
    
    # STEP 1: Load data
    log.info("STEP 1/4: Loading training data...")
    df = load_training_data(args.data)
    
    # STEP 2: Prepare features and target
    log.info("\nSTEP 2/4: Preparing features and target...")
    X, y = prepare_features_targets(df, target_col=args.target)
    
    log.info("Target distribution:")
    print(y.value_counts().sort_index())
    print(f"\nTarget statistics:")
    print(y.describe())
    
    # STEP 3: Split data
    log.info("\nSTEP 3/4: Splitting data into train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(
        X, y, 
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # STEP 4: Train model
    log.info("\nSTEP 4/4: Training XGBoost model...")
    predictor = AQIPredictor(task=args.task)
    
    train_history = predictor.train(
        X_train, y_train,
        X_val, y_val,
        scale_features=True
    )
    
    log.info("✓ Training complete")
    
    # STEP 5: Evaluate model
    log.info("\n" + "="*70)
    log.info("MODEL EVALUATION")
    log.info("="*70)
    
    eval_dir = f"{args.output_dir}/evaluation"
    results = evaluate_model(
        predictor, 
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        output_dir=eval_dir
    )
    
    # STEP 6: Feature importance
    log.info("\nAnalyzing feature importance...")
    importance_df = predictor.get_feature_importance()
    
    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))
    
    plot_feature_importance(importance_df, eval_dir, top_n=20)
    
    # Save importance to CSV
    importance_df.to_csv(f"{eval_dir}/feature_importance.csv", index=False)
    log.info("✓ Feature importance saved")
    
    # STEP 7: Save model locally
    model_dir = f"{args.output_dir}/models"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    model_filename = f"{args.target}_{args.task}_xgboost.pkl"
    model_path = f"{model_dir}/{model_filename}"
    
    predictor.save(model_path)
    log.info(f"Model saved locally: {model_path}")
    
    # STEP 8: Register to Model Registry
    log.info("\n" + "="*70)
    log.info("REGISTERING MODEL TO HOPSWORKS")
    log.info("="*70)
    
    try:
        from modeling.model_registry import register_model_to_hopsworks
        
        log.info("Registering model to Hopsworks Model Registry...")
        
        # Prepare metrics for registry
        registry_metrics = {
            'test_rmse': float(results['test']['rmse']),
            'test_mae': float(results['test']['mae']),
            'test_r2': float(results['test']['r2']),
            'test_mape': float(results['test']['mape']),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test))
        }
        
        # Register model
        registered_model = register_model_to_hopsworks(
            model_path=model_path,
            model_name=f"aqi_{args.target}_{args.task}",
            metrics=registry_metrics,
            feature_names=list(X_train.columns),
            target_name=args.target,
            description=f"XGBoost {args.task} model for {args.target} prediction"
        )
        
        log.info(f"✓ Model registered to Hopsworks Model Registry!")
        log.info(f"  Registry Name: {registered_model.name}")
        log.info(f"  Version: {registered_model.version}")
        
    except ImportError:
        log.warning("Hopsworks not available. Model saved locally only.")
    except Exception as e:
        log.error(f"Failed to register model to registry: {e}")
        log.warning("Model saved locally but not registered.")
    
    # FINAL SUMMARY
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"✓ Model saved:      {model_path}")
    print(f"✓ Evaluation plots: {eval_dir}/")
    print(f"✓ Metrics summary:  {eval_dir}/metrics_summary.csv")
    print(f"\nFinal Test Performance:")
    print(f"  RMSE: {results['test']['rmse']:.4f}")
    print(f"  MAE:  {results['test']['mae']:.4f}")
    print(f"  R²:   {results['test']['r2']:.4f}")
    print("="*70)
    
    log.info("[SUCCESS] Training pipeline completed!")


if __name__ == "__main__":
    main()
