
import sys
import os
from pathlib import Path
import subprocess

def main():
    # Get project root
    project_root = Path(__file__).parent.resolve()
    
    # Add to Python path
    sys.path.insert(0, str(project_root))
    os.environ['PYTHONPATH'] = str(project_root)
    
    # Dashboard path
    dashboard_path = project_root / "dashboard" / "app.py"
    
    print("=" * 70)
    print("🚀 Starting AQI Dashboard")
    print("=" * 70)
    print(f"Project Root: {project_root}")
    print(f"Dashboard:    {dashboard_path}")
    print(f"PYTHONPATH:   {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"URL:          http://localhost:8501")
    print("=" * 70)
    print()
    
    # Verify dashboard file exists
    if not dashboard_path.exists():
        print(f"❌ ERROR: Dashboard file not found at {dashboard_path}")
        return 1
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Dashboard failed to start: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
