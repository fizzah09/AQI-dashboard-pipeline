"""
Comprehensive Hopsworks Verification Script
Checks Feature Store and Model Registry health
"""
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hopsworks
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def verify_connection():
    """Verify Hopsworks connection"""
    print("🔌 Testing Hopsworks Connection...")
    print("="*70)
    
    try:
        project = hopsworks.login(
            api_key_value=os.getenv('HOPSWORKS_API_KEY'),
            project=os.getenv('HOPSWORKS_PROJECT_NAME')
        )
        
        print(f"✅ Connected Successfully!")
        print(f"   Project: {project.name}")
        print(f"   Project ID: {project.id if hasattr(project, 'id') else 'N/A'}")
        
        return project, True
        
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return None, False


def verify_feature_store(project):
    """Verify Feature Store and Feature Groups"""
    print("\n📦 Verifying Feature Store...")
    print("="*70)
    
    results = {
        'status': 'unknown',
        'feature_groups': [],
        'errors': []
    }
    
    try:
        fs = project.get_feature_store()
        print(f"✅ Feature Store: {fs.name}")
        
        # Check specific AQI feature groups
        print("\n🔍 Checking AQI-specific Feature Groups...")
        
        for fg_name, version in [('weather_features', 2), ('pollutant_features', 2)]:
            try:
                fg = fs.get_feature_group(fg_name, version=version)
                print(f"\n  ✅ {fg_name} v{version} - Found")
                
                fg_info = {
                    'name': fg.name,
                    'version': fg.version,
                    'features_count': len(fg.features) if hasattr(fg, 'features') else 0,
                    'status': 'found'
                }
                
                print(f"     Features: {len(fg.features) if hasattr(fg, 'features') else 'N/A'}")
                
                # Try to read sample
                try:
                    df = fg.read(limit=5)
                    print(f"     Sample rows: {len(df)}")
                    print(f"     Columns: {list(df.columns)[:10]}")
                    fg_info['status'] = 'readable'
                    fg_info['row_count'] = len(df)
                except Exception as read_err:
                    print(f"     ⚠️  Read Error: {str(read_err)[:50]}")
                    fg_info['status'] = 'found_but_not_readable'
                    fg_info['error'] = str(read_err)
                
                results['feature_groups'].append(fg_info)
                
            except Exception as e:
                print(f"   ❌ {fg_name} v{version} - Not Found: {str(e)[:100]}")
                results['errors'].append(f"Missing: {fg_name} v{version}")
                results['feature_groups'].append({
                    'name': fg_name,
                    'version': version,
                    'status': 'not_found',
                    'error': str(e)
                })
        
        # Determine overall status
        if len(results['errors']) == 0:
            results['status'] = 'success'
        elif len(results['feature_groups']) > 0:
            results['status'] = 'partial'
        else:
            results['status'] = 'error'
        
    except Exception as e:
        print(f"❌ Feature Store Error: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'error'
        results['errors'].append(str(e))
    
    return results


def verify_model_registry(project):
    """Verify Model Registry and Models"""
    print("\n🤖 Verifying Model Registry...")
    print("="*70)
    
    results = {
        'status': 'unknown',
        'models': [],
        'errors': []
    }
    
    try:
        mr = project.get_model_registry()
        print(f"✅ Model Registry Access: OK")
        
        # Try to list models - check if any exist
        print("\n� Searching for models...")
        
        # Try to get specific AQI models
        model_names_to_check = [
            'aqi_pollutant_aqi_regression',
            'aqi_xgboost_model',
            'aqi_model'
        ]
        
        found_any = False
        
        for model_name in model_names_to_check:
            try:
                # Try to get the model
                model = mr.get_model(model_name)
                
                if model:
                    found_any = True
                    model_info = {
                        'name': model.name,
                        'version': model.version,
                        'created': str(model.created) if hasattr(model, 'created') else 'N/A'
                    }
                    
                    print(f"\n  ✅ Found: {model.name} (v{model.version})")
                    print(f"     Created: {model.created if hasattr(model, 'created') else 'N/A'}")
                    
                    # Get metrics if available
                    if hasattr(model, 'training_metrics') and model.training_metrics:
                        print(f"     Metrics: {model.training_metrics}")
                        model_info['metrics'] = model.training_metrics
                    
                    # Try to download
                    try:
                        model_dir = model.download()
                        print(f"     Download: ✅ Success")
                        model_info['downloadable'] = True
                        
                        # Check files
                        files = list(Path(model_dir).rglob('*'))
                        print(f"     Files: {len(files)} files")
                        model_info['files_count'] = len(files)
                        
                    except Exception as e:
                        print(f"     Download: ⚠️  {str(e)[:50]}")
                        model_info['downloadable'] = False
                        model_info['error'] = str(e)
                    
                    results['models'].append(model_info)
                    
            except Exception as e:
                # Model doesn't exist, continue
                continue
        
        if not found_any:
            print("⚠️  No models found in registry")
            print("💡 This is normal if training hasn't run yet")
            results['status'] = 'empty'
            results['errors'].append("No models in registry (normal for new setup)")
        else:
            print(f"\n✅ Found {len(results['models'])} model(s)")
            results['status'] = 'success'
        
    except Exception as e:
        print(f"❌ Model Registry Error: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'error'
        results['errors'].append(str(e))
    
    return results


def generate_report(connection_ok, fs_results, mr_results):
    """Generate comprehensive verification report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'connection': {
            'status': 'success' if connection_ok else 'failed'
        },
        'feature_store': fs_results,
        'model_registry': mr_results,
        'overall_status': 'healthy' if (
            connection_ok and 
            fs_results['status'] in ['success', 'partial'] and 
            mr_results['status'] in ['success', 'partial']
        ) else 'unhealthy'
    }
    
    # Save report
    with open('hopsworks_verification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*70)
    print("📊 VERIFICATION SUMMARY")
    print("="*70)
    print(f"Overall Status: {'✅ HEALTHY' if report['overall_status'] == 'healthy' else '❌ UNHEALTHY'}")
    print(f"Connection: {'✅' if connection_ok else '❌'}")
    print(f"Feature Store: {fs_results['status'].upper()}")
    print(f"  - Feature Groups: {len(fs_results.get('feature_groups', []))}")
    print(f"  - Errors: {len(fs_results.get('errors', []))}")
    print(f"Model Registry: {mr_results['status'].upper()}")
    print(f"  - Models: {len(mr_results.get('models', []))}")
    print(f"  - Errors: {len(mr_results.get('errors', []))}")
    print("="*70)
    print(f"📄 Report saved: hopsworks_verification_report.json")
    
    return report


def main():
    print("\n" + "="*70)
    print("HOPSWORKS INTEGRATION VERIFICATION")
    print("="*70)
    print(f"Time: {datetime.now()}")
    print(f"Project: {os.getenv('HOPSWORKS_PROJECT_NAME')}")
    print("="*70)
    
    # Step 1: Verify connection
    project, connection_ok = verify_connection()
    
    fs_results = {'status': 'skipped', 'feature_groups': [], 'errors': []}
    mr_results = {'status': 'skipped', 'models': [], 'errors': []}
    
    if connection_ok and project:
        # Step 2: Verify Feature Store
        fs_results = verify_feature_store(project)
        
        # Step 3: Verify Model Registry
        mr_results = verify_model_registry(project)
    
    # Step 4: Generate report
    report = generate_report(connection_ok, fs_results, mr_results)
    
    # Exit with error code if unhealthy
    if report['overall_status'] != 'healthy':
        print("\n⚠️  System is not fully healthy. Check errors above.")
        sys.exit(1)
    else:
        print("\n✅ All systems operational!")
        sys.exit(0)


if __name__ == "__main__":
    main()
