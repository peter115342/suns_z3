"""Data loading utilities using Polars."""

from pathlib import Path
from typing import List

import polars as pl
from PIL import Image
import numpy as np


def load_solar_data(month_folder: Path) -> pl.DataFrame:
    """
    Load solar data from a specific month folder.

    Args:
        month_folder: Path to month folder (e.g., dataset/Solar_data/01)

    Returns:
        Polars DataFrame with solar data
    """
    csv_path = month_folder / "out_data.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pl.read_csv(csv_path)

    df = df.with_columns(
        [pl.col("DateTime").str.split("#").list.get(0).alias("DateTime_clean")]
    )

    month_name = month_folder.name
    df = df.with_columns([pl.lit(month_name).alias("Month")])

    return df


def load_all_months(
    data_dir: Path, months: List[str] = ["01", "04", "07", "10"]
) -> pl.DataFrame:
    """
    Load data from all specified months.

    Args:
        data_dir: Path to Solar_data directory
        months: List of month folders to load

    Returns:
        Combined Polars DataFrame
    """
    dfs = []

    for month in months:
        month_path = data_dir / month
        if month_path.exists():
            print(f"Loading data from month {month}...")
            df = load_solar_data(month_path)
            dfs.append(df)
        else:
            print(f"Warning: Month folder {month} not found")

    if not dfs:
        raise ValueError("No data loaded. Check data directory.")

    combined_df = pl.concat(dfs)

    print(f"\nTotal records loaded: {len(combined_df)}")
    print(f"Months: {combined_df['Month'].unique().sort()}")

    return combined_df


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image and return as numpy array (RGB).

    Args:
        image_path: Path to image file

    Returns:
        Numpy array of shape (H, W, 3)
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def get_image_path(picture_name: str, data_dir: Path, month: str) -> Path:
    """
    Get full path to image file.

    Args:
        picture_name: Name from PictureName column
        data_dir: Path to Solar_data directory
        month: Month folder (e.g., "01")

    Returns:
        Path to image file
    """
    image_path = data_dir / month / "original" / picture_name
    return image_path


def verify_image_existence(df: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    """
    Add a column indicating whether the image file exists.

    Args:
        df: DataFrame with PictureName and Month columns
        data_dir: Path to Solar_data directory

    Returns:
        DataFrame with ImageExists column
    """
    image_exists = []

    for row in df.iter_rows(named=True):
        img_path = get_image_path(row["PictureName"], data_dir, row["Month"])
        image_exists.append(img_path.exists())

    df = df.with_columns([pl.Series("ImageExists", image_exists)])

    return df
