
import logging
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.backfill.combined_backfill import run_combined_backfill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("backfill.log"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)

def main():
    log.info("="*70)
    log.info("STARTING 1-YEAR COMBINED WEATHER + AIR QUALITY BACKFILL")
    log.info("="*70)
    
    cfg_dict = load_config()
    
    log.info("Configuration:")
    log.info("  Location: (%s, %s) - %s", 
             cfg_dict["location"]["lat"], 
             cfg_dict["location"]["lon"],
             cfg_dict["location"]["city"])
    log.info("  Date range: 2024-10-21 to 2025-10-20")
    log.info("  Expected records: 365 days")
    log.info("  Weather source: Open-Meteo (FREE, bulk)")
    log.info("  Pollutant source: OpenWeather (FREE, historical)")
    log.info("  Output: data/ml_training_data_1year.csv")
    log.info("")
    
    df = run_combined_backfill(
        openweather_api_key=cfg_dict["openweather"]["api_key"],
        pollution_base_url=cfg_dict["openweather"]["air_pollution_url"],
        lat=cfg_dict["location"]["lat"],
        lon=cfg_dict["location"]["lon"],
        start_date="2024-10-21",
        end_date="2025-10-20",
        output_csv="data/ml_training_data_1year.csv"
    )
    
    if df.empty:
        log.error("❌ Backfill failed - no data collected")
        return
    
    log.info("="*70)
    log.info("✓ BACKFILL COMPLETE")
    log.info("="*70)
    log.info("Records collected: %d", len(df))
    log.info("Columns: %s", list(df.columns))
    log.info("Date range: %s to %s", df['timestamp'].min(), df['timestamp'].max())
    log.info("")
    log.info("Data summary:")
    print(df.describe())
    log.info("")
    log.info("Sample records:")
    print(df.head(10))
    log.info("")
    log.info("Missing values:")
    print(df.isnull().sum())
    log.info("")
    log.info("AQI distribution:")
    if 'aqi_category' in df.columns:
        print(df['aqi_category'].value_counts())
    
    log.info("\n✓ Training data ready for ML modeling at: data/ml_training_data_1year.csv")

if __name__ == "__main__":
    main()
