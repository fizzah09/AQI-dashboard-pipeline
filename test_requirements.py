"""
Test if all required packages are installed correctly
"""
import sys

def test_imports():
    """Test importing all required packages"""
    
    print("="*60)
    print("TESTING PACKAGE IMPORTS")
    print("="*60)
    print()
    
    packages_to_test = {
        # Core
        "dotenv": "python-dotenv",
        
        # Data Processing
        "pandas": "pandas",
        "numpy": "numpy",
        "yaml": "pyyaml",
        
        # ML
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "joblib": "joblib",
        
        # Web
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "streamlit": "streamlit",
        
        # API
        "requests": "requests",
        "httpx": "httpx",
        "pydantic": "pydantic",
        
        # Visualization
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "plotly": "plotly",
        
        # Explainability
        "shap": "shap",
        "lime": "lime",
        
        # Hopsworks
        "hopsworks": "hopsworks",
        "hsml": "hsml",
        "pyarrow": "pyarrow",
        
        # Testing
        "pytest": "pytest",
        
        # Utilities
        "tqdm": "tqdm",
    }
    
    failed = []
    passed = []
    
    for module_name, package_name in packages_to_test.items():
        try:
            __import__(module_name)
            print(f"✅ {package_name:<25} - OK")
            passed.append(package_name)
        except ImportError as e:
            print(f"❌ {package_name:<25} - MISSING")
            failed.append(package_name)
    
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Passed: {len(passed)}/{len(packages_to_test)}")
    print(f"❌ Failed: {len(failed)}/{len(packages_to_test)}")
    
    if failed:
        print()
        print("Missing packages:")
        for pkg in failed:
            print(f"  - {pkg}")
        print()
        print("Install missing packages with:")
        print("  pip install " + " ".join(failed))
        return False
    else:
        print()
        print("🎉 All packages installed successfully!")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
