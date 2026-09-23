# -*- coding: utf-8 -*-
"""Dataset loading helpers."""

import glob
import os

import pandas as pd

KAGGLE_DATASET = "anirudhchauhan/retail-store-inventory-forecasting-dataset"


def fetch_kaggle_dataset(dataset: str = KAGGLE_DATASET) -> str:
    """Download the retail dataset from Kaggle via kagglehub and return the CSV path.

    Requires the `kagglehub` package and a configured Kaggle API token
    (~/.kaggle/kaggle.json, or KAGGLE_USERNAME/KAGGLE_KEY env vars).
    """
    try:
        import kagglehub
    except ImportError as e:
        raise ImportError(
            "kagglehub is required to auto-download the dataset. "
            "Install it with: pip install kagglehub"
        ) from e

    print(f"Downloading Kaggle dataset '{dataset}' via kagglehub...")
    download_path = kagglehub.dataset_download(dataset)
    print(f"Dataset downloaded to: {download_path}")

    csv_files = glob.glob(os.path.join(download_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded dataset at '{download_path}'.")

    preferred = [f for f in csv_files if os.path.basename(f) == "retail_store_inventory.csv"]
    csv_path = preferred[0] if preferred else csv_files[0]
    print(f"Using dataset file: {csv_path}")
    return csv_path


def load_retail_dataset(csv_path: str) -> pd.DataFrame:
    """Load the retail inventory CSV dataset.

    Expected columns (spaces/slashes are auto-standardized by the environment):
    Price, Discount, Weather Condition, Holiday/Promotion, Competitor Pricing,
    Seasonality, Category, Region, Inventory Level, Units Sold, Demand Forecast.
    """
    df = pd.read_csv(csv_path)
    print(f"Data loaded! Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}\n")
    return df
