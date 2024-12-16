#!/usr/bin/env python3

import copy
import glob
import time
from typing import List, Tuple
from helium import Text, click, start_chrome, wait_until, Alert
import httpx
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
from meteostat import Point, Daily
from dotenv import load_dotenv
from pandas.errors import EmptyDataError

from infra.feature_store import FeatureStore


# BASE_URL = "http://api.waqi.info/feed"
BASE_URL = "https://aqicn.org/historical"
CITIES = ["seattle", "bellevue"]


def get_station_and_geo(city: str, api_key: str) -> Tuple[str, List[float]]:
    r = httpx.get(f"http://api.waqi.info/feed/{city}/?token={api_key}")
    r.raise_for_status()
    return r.json()["data"]["city"]["name"], r.json()["data"]["city"]["geo"]


def feature_pipeline(api_key: str, city: str) -> None:
    filename = f"data/unprocessed/{city}.csv"
    if not os.path.exists(filename):
        print(f"No new data to process for {city}")
        return

    # Read and sort unprocessed data
    df = pd.read_csv(filename, parse_dates=["date"])  # Parse dates while reading
    df = df.sort_values("date")

    # Save processed raw data for archival purposes
    df.to_csv(filename.replace("unprocessed", "processed"), index=False)
    os.remove(filename)

    # Convert all air quality columns to numeric
    for col in [" pm25", " o3", " no2", " so2", " co"]:
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")
        df[col] = df[col].bfill().ffill()

    # Get geo location and process features
    _, geo = get_station_and_geo(city, api_key)
    df = process_features(df, geo)
    df.sort_values("date", inplace=True, ascending=False)

    # Save to feature store instead of CSV
    feature_store = FeatureStore()
    saved_paths = feature_store.save_features(
        city, df, is_online=False
    )  # Save to offline store
    print(f"Saved historical features for {city} to {len(saved_paths)} daily files")


def fetch_weather_data(start_date, end_date, geo):
    seattle = Point(geo[0], geo[1])

    # Get daily weather data
    data = Daily(seattle, start_date, end_date)
    data = data.fetch()

    # Select relevant features
    selected_features = [
        "tavg",  # Average temperature (°C)
        "prcp",  # Precipitation (mm)
        "wspd",  # Wind speed (km/h)
        "pres",  # Air pressure (hPa) - useful for air quality prediction
        "wdir",  # Wind direction (degrees) - important for pollution transport
    ]

    return data[selected_features]


def process_features(aqi_df, geo: List[float]):
    # Get date range for weather data
    start_date = aqi_df["date"].min()
    end_date = aqi_df["date"].max()

    # Fetch weather data
    weather_df = fetch_weather_data(start_date, end_date, geo)

    # Reset index to make date a column
    weather_df = weather_df.reset_index()
    weather_df = weather_df.rename(columns={"time": "date"})

    # Merge AQI and weather data
    combined_df = pd.merge(aqi_df, weather_df, on="date", how="left")

    # Forward fill missing weather values (if any)
    combined_df = combined_df.bfill().ffill()

    # Add time-based features
    combined_df["day_of_week"] = combined_df["date"].dt.dayofweek
    combined_df["month"] = combined_df["date"].dt.month

    return combined_df


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("AQI_API_KEY")

    if not api_key:
        raise ValueError("AQI_API_KEY is required")

    for city in CITIES:
        print(f"Processing features for {city}")
        feature_pipeline(api_key, city)
