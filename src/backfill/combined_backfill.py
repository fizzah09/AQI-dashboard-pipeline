"""
Complete 1-year backfill combining:
- Weather data from Open-Meteo (FREE, bulk download)
- Pollutant data from OpenWeather (FREE, historical API)
"""
import time
import logging
from typing import List, Dict, Any
import pandas as pd

from src.api_client.fetch_weather_openmeteo import fetch_openmeteo_bulk
from src.api_client.fetch_pollutant_historical import fetch_pollutant_historical
from src.feature_engineering.weather_features import compute_weather_features
from src.feature_engineering.pollutant_features import compute_pollutant_features
from src.feature_engineering.cleaning import clean_weather_record, clean_pollutant_record

log = logging.getLogger(__name__)


def run_combined_backfill(
    openweather_api_key: str,
    pollution_base_url: str,
    lat: float,
    lon: float,
    start_date: str,  # "YYYY-MM-DD"
    end_date: str,    # "YYYY-MM-DD"
    output_csv: str,
) -> pd.DataFrame:
    """
    Backfill 1 year of combined weather + pollutant data.
    
    Strategy:
    1. Bulk download weather data from Open-Meteo (1 API call for entire year)
    2. Fetch pollutant data day-by-day from OpenWeather (365 calls with rate limiting)
    3. Merge on timestamp
    4. Clean, compute features, save to CSV
    """
    log.info("="*70)
    log.info("COMBINED BACKFILL: Weather (Open-Meteo) + Pollutant (OpenWeather)")
    log.info("="*70)
    log.info("Location: (%s, %s)", lat, lon)
    log.info("Date range: %s to %s", start_date, end_date)
    log.info("")
    
    # STEP 1: Get all weather data in ONE call
    log.info("Step 1/3: Fetching weather data from Open-Meteo...")
    try:
        weather_records = fetch_openmeteo_bulk(lat, lon, start_date, end_date)
        log.info("✓ Got %d days of weather data", len(weather_records))
    except Exception as e:
        log.error("Failed to fetch weather data: %s", e)
        return pd.DataFrame()
    
    # Create timestamp -> weather mapping
    weather_by_ts: Dict[int, Dict[str, Any]] = {}
    for rec in weather_records:
        ts = rec["timestamp"]
        weather_by_ts[ts] = rec
    
    # STEP 2: Fetch pollutant data day by day
    log.info("\nStep 2/3: Fetching pollutant data from OpenWeather...")
    log.info("This will take ~%d minutes with rate limiting...", len(weather_records) // 50)
    
    pollutant_records: List[Dict[str, Any]] = []
    last_pollutant_raws: List[Dict[str, Any]] = []
    success = 0
    failed = 0
    
    for i, weather_rec in enumerate(weather_records, 1):
        target_ts = weather_rec["timestamp"]
        
        try:
            raw_p = fetch_pollutant_historical(
                api_key=openweather_api_key,
                base_url=pollution_base_url,
                lat=lat,
                lon=lon,
                dt_unix=target_ts,
            )
            
            if raw_p and raw_p.get("timestamp"):
                pollutant_records.append(raw_p)
                last_pollutant_raws.append(raw_p)
                success += 1
            else:
                failed += 1
                
        except Exception as e:
            log.warning("Pollutant fetch failed for ts=%s: %s", target_ts, str(e)[:80])
            failed += 1
        
        # Progress updates
        if i % 50 == 0:
            log.info("Progress: %d/%d (%.1f%%) - Success: %d, Failed: %d",
                    i, len(weather_records), (i/len(weather_records))*100, success, failed)
        
        # Rate limiting (60 calls/minute max)
        time.sleep(1.2)
    
    log.info("✓ Pollutant data: %d success, %d failed", success, failed)
    
    # Create timestamp -> pollutant mapping
    pollutant_by_ts: Dict[int, Dict[str, Any]] = {}
    for rec in pollutant_records:
        # Match to nearest weather timestamp
        p_ts = rec["timestamp"]
        closest_ts = min(weather_by_ts.keys(), key=lambda t: abs(t - p_ts))
        pollutant_by_ts[closest_ts] = rec
    
    # STEP 3: Merge, clean, compute features
    log.info("\nStep 3/3: Merging data and computing features...")
    
    combined_rows: List[Dict[str, Any]] = []
    
    for i, (ts, weather_raw) in enumerate(sorted(weather_by_ts.items())):
        # Clean raw data
        weather_clean = clean_weather_record(weather_raw)
        
        pollutant_raw = pollutant_by_ts.get(ts)
        if pollutant_raw:
            pollutant_clean = clean_pollutant_record(pollutant_raw)
        else:
            pollutant_clean = {}
        
        # Compute features
        weather_features = compute_weather_features(weather_clean)
        
        if pollutant_clean:
            # Get historical context for change rates
            hist_index = i
            hist_slice = pollutant_records[max(0, hist_index-10):hist_index] if hist_index > 0 else []
            pollutant_features = compute_pollutant_features(pollutant_clean, historical_data=hist_slice)
        else:
            pollutant_features = {}
        
        # Merge features
        merged = {**weather_features}
        for k, v in pollutant_features.items():
            if k != "timestamp":  # Avoid duplicate timestamp
                merged[f"pollutant_{k}"] = v
        
        combined_rows.append(merged)
    
    # Create DataFrame
    df = pd.DataFrame(combined_rows)
    
    # Sort and deduplicate
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    
    log.info("✓ Final dataset: %d rows, %d columns", len(df), len(df.columns))
    
    # Save to CSV
    try:
        df.to_csv(output_csv, index=False)
        log.info("Saved to: %s", output_csv)
    except Exception as e:
        log.error("Failed to save CSV: %s", e)
    
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    
    from src.config import load_config
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("combined_backfill.log"),
            logging.StreamHandler()
        ]
    )
    
    cfg = load_config()
    
    df = run_combined_backfill(
        openweather_api_key=cfg["openweather"]["api_key"],
        pollution_base_url=cfg["openweather"]["air_pollution_url"],
        lat=cfg["location"]["lat"],
        lon=cfg["location"]["lon"],
        start_date="2024-10-21",
        end_date="2025-10-20",
        output_csv="data/ml_training_data_1year.csv"
    )
    
    print("\n" + "="*70)
    print("BACKFILL COMPLETE")
    print("="*70)
    print(f"Collected: {len(df)} days")
    print(f"Columns: {len(df.columns)}")
    print(f"\nData Summary:")
    print(df.describe())
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nFirst 5 rows:")
    print(df.head())
