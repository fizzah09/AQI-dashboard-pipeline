import logging
import sys
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import load_config
from src.api_client.weather_client import WeatherClient
from src.api_client.pollutant_client import PollutantClient
from src.feature_engineering.weather_features import compute_weather_features
from src.feature_engineering.pollutant_features import compute_pollutant_features
from src.feature_store.store_manager import StoreManager
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def main():
    parser = ArgumentParser(description="Weather/Pollutant feature pipeline")
    parser.add_argument("--backfill", action="store_true", help="Run historical backfill")
    parser.add_argument("--start", type=str, help="Backfill start ISO date, e.g., 2024-01-01")
    parser.add_argument("--end", type=str, help="Backfill end ISO date, e.g., 2024-12-31")
    parser.add_argument("--hours", type=int, default=1, help="Backfill step in hours (default 1)")
    parser.add_argument("--out", type=str, help="Optional CSV output path for combined dataset")

    cfg = load_config()
    wc = WeatherClient(api_key=cfg["openweather"]["api_key"],
                       base_url=cfg["openweather"]["base_url"])
    pc = PollutantClient(api_key=cfg["openweather"]["api_key"],
                         base_url=cfg["openweather"]["air_pollution_url"])
    lat, lon = cfg["location"]["lat"], cfg["location"]["lon"]

    args = parser.parse_args()
    if args.backfill:
        start = pd.to_datetime(args.start).tz_localize("UTC") if args.start else pd.Timestamp.utcnow().tz_localize("UTC").normalize() - pd.Timedelta(days=365)
        end = pd.to_datetime(args.end).tz_localize("UTC") if args.end else pd.Timestamp.utcnow().tz_localize("UTC")
        bf_cfg = BackfillConfig(
            api_key=cfg["openweather"]["api_key"],
            weather_base_url=cfg["openweather"]["base_url"],
            pollution_base_url=cfg["openweather"]["air_pollution_url"],
            lat=lat,
            lon=lon,
            start=start,
            end=end,
            step_hours=int(args.hours or 1),
            rate_limit_sec=1.0,
            write_csv_path=args.out,
        )
        df = run_backfill(bf_cfg)
        log.info("Backfill produced %d rows", len(df))
        return

    raw_w = wc.fetch_weather_data(lat=lat, lon=lon)
    raw_p = pc.fetch_pollutant_data(lat=lat, lon=lon)

    wf = compute_weather_features(raw_w)
    pf = compute_pollutant_features(raw_p, historical_data=None)

    # Try to store in Hopsworks, but continue if unavailable
    try:
        sm = StoreManager(api_key=cfg["hopsworks"]["api_key"],
                          project_name=cfg["hopsworks"]["project_name"])
        sm.store_features(wf, cfg["feature_groups"]["weather"]["name"],
                          version=cfg["feature_groups"]["weather"]["version"],
                          primary_key=cfg["feature_groups"]["weather"]["primary_key"],
                          event_time=cfg["feature_groups"]["weather"]["event_time"])
        sm.store_features(pf, cfg["feature_groups"]["pollutant"]["name"],
                          version=cfg["feature_groups"]["pollutant"]["version"],
                          primary_key=cfg["feature_groups"]["pollutant"]["primary_key"],
                          event_time=cfg["feature_groups"]["pollutant"]["event_time"])
    except Exception as e:
        log.warning("Skipping Hopsworks storage: %s", e)
        log.info("Weather features sample: %s", {k: wf.get(k) for k in list(wf)[:6]})
        log.info("Pollutant features sample: %s", {k: pf.get(k) for k in list(pf)[:6]})

if __name__ == "__main__":
    main()