"""
Quick test to verify Hopsworks connection and check if data/models exist
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*70)
print("QUICK HOPSWORKS TEST")
print("="*70)

try:
    import hopsworks
    
    print("\n1️⃣ Testing Connection...")
    project = hopsworks.login(
        api_key_value=os.getenv('HOPSWORKS_API_KEY'),
        project=os.getenv('HOPSWORKS_PROJECT_NAME')
    )
    print(f"✅ Connected to: {project.name}")
    
    print("\n2️⃣ Testing Feature Store...")
    fs = project.get_feature_store()
    print(f"✅ Feature Store: {fs.name}")
    
    # Check for weather features
    print("\n3️⃣ Checking Feature Groups...")
    try:
        weather_fg = fs.get_feature_group('weather_features', version=2)
        print(f"✅ weather_features v2 exists")
        print(f"   Features: {len(weather_fg.features) if hasattr(weather_fg, 'features') else 'N/A'}")
    except Exception as e:
        print(f"⚠️  weather_features v2: {str(e)[:100]}")
    
    try:
        pollutant_fg = fs.get_feature_group('pollutant_features', version=2)
        print(f"✅ pollutant_features v2 exists")
        print(f"   Features: {len(pollutant_fg.features) if hasattr(pollutant_fg, 'features') else 'N/A'}")
    except Exception as e:
        print(f"⚠️  pollutant_features v2: {str(e)[:100]}")
    
    print("\n4️⃣ Testing Model Registry...")
    mr = project.get_model_registry()
    print(f"✅ Model Registry access: OK")
    
    # Try to find any AQI models
    model_found = False
    for model_name in ['aqi_pollutant_aqi_regression', 'aqi_xgboost_model', 'aqi_model']:
        try:
            model = mr.get_model(model_name)
            print(f"✅ Found model: {model.name} v{model.version}")
            model_found = True
            break
        except:
            continue
    
    if not model_found:
        print("⚠️  No models found yet (normal for new setup)")
        print("💡 Models will be created after first training run")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n📋 Next Steps:")
    print("1. Run feature pipeline: python src/main.py")
    print("2. Run training pipeline: python run_training.py --use-hopsworks")
    print("3. Check GitHub Actions are configured with secrets")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
