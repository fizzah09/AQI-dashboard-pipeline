"""
Test script to verify FastAPI and Streamlit setup
"""
import sys
from pathlib import Path
import requests
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_fastapi_running():
    """Test if FastAPI is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI is running")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"⚠️  FastAPI returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ FastAPI is not running")
        print("   Start it with: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error connecting to FastAPI: {e}")
        return False


def test_streamlit_running():
    """Test if Streamlit is running"""
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit is running")
            return True
        else:
            print(f"⚠️  Streamlit returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Streamlit is not running")
        print("   Start it with: streamlit run dashboard/app.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Streamlit: {e}")
        return False


def test_model_exists():
    """Test if trained model exists"""
    model_dir = Path("modeling/models")
    if not model_dir.exists():
        print("❌ Model directory not found")
        return False
    
    model_files = list(model_dir.glob("*_xgboost.pkl"))
    if model_files:
        print(f"✅ Model found: {model_files[0].name}")
        return True
    else:
        print("❌ No trained model found")
        print("   Train a model with: python run_training.py")
        return False


def test_data_exists():
    """Test if training data exists"""
    data_path = Path("data/ml_training_data_1year.csv")
    if data_path.exists():
        print(f"✅ Training data found: {data_path}")
        return True
    else:
        print(f"❌ Training data not found: {data_path}")
        return False


def test_prediction_api():
    """Test prediction endpoint"""
    try:
        payload = {
            "weather_temp": 25.0,
            "weather_humidity": 60.0,
            "weather_pressure": 1013.0,
            "weather_wind_speed": 5.0,
            "pollutant_pm2_5": 50.0,
            "pollutant_pm10": 80.0,
            "pollutant_no2": 40.0,
            "pollutant_o3": 60.0,
            "pollutant_so2": 20.0,
            "pollutant_co": 1000.0
        }
        
        response = requests.post(
            "http://localhost:8000/predict",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction API works")
            print(f"   AQI Prediction: {result['aqi_prediction']:.1f}")
            print(f"   Category: {result['category']}")
            return True
        else:
            print(f"⚠️  Prediction API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to prediction API (FastAPI not running)")
        return False
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("AQI DASHBOARD & API - SYSTEM CHECK")
    print("="*60)
    print()
    
    print("1. Checking File System...")
    print("-" * 60)
    model_ok = test_model_exists()
    data_ok = test_data_exists()
    print()
    
    print("2. Checking Services...")
    print("-" * 60)
    fastapi_ok = test_fastapi_running()
    streamlit_ok = test_streamlit_running()
    print()
    
    if fastapi_ok:
        print("3. Testing API Endpoints...")
        print("-" * 60)
        prediction_ok = test_prediction_api()
        print()
    else:
        prediction_ok = False
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    all_checks = [
        ("Model Available", model_ok),
        ("Training Data Available", data_ok),
        ("FastAPI Running", fastapi_ok),
        ("Streamlit Running", streamlit_ok),
        ("Prediction API Working", prediction_ok),
    ]
    
    for check_name, status in all_checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}")
    
    print()
    
    total_passed = sum(1 for _, status in all_checks if status)
    total_checks = len(all_checks)
    
    if total_passed == total_checks:
        print(f"🎉 All checks passed! ({total_passed}/{total_checks})")
        print()
        print("🌐 Access URLs:")
        print("   FastAPI Docs:  http://localhost:8000/docs")
        print("   Streamlit UI:  http://localhost:8501")
    else:
        print(f"⚠️  {total_checks - total_passed} checks failed")
        print()
        print("📋 Next Steps:")
        if not model_ok:
            print("   1. Train model: python run_training.py")
        if not fastapi_ok:
            print("   2. Start FastAPI: python -m uvicorn api.main:app --port 8000")
        if not streamlit_ok:
            print("   3. Start Streamlit: streamlit run dashboard/app.py")
    
    print("="*60)


if __name__ == "__main__":
    main()
