"""Data preprocessing and splitting utilities."""

from pathlib import Path
from typing import Tuple, Dict

import polars as pl


def pair_images_with_metadata(
    df: pl.DataFrame, data_dir: Path, target_column: str = "Irradiance"
) -> Tuple[pl.DataFrame, pl.Series]:
    """
    Ensure images are properly paired with metadata.

    Args:
        df: DataFrame with metadata
        data_dir: Path to Solar_data directory
        target_column: Name of target column

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Verify image existence
    from .loading import verify_image_existence

    df = verify_image_existence(df, data_dir)

    # Filter only rows where images exist
    df_valid = df.filter(pl.col("ImageExists"))

    print(f"Valid image-metadata pairs: {len(df_valid)} / {len(df)}")

    # Separate features and target
    X = df_valid.drop(target_column)
    y = df_valid.select(target_column)

    return X, y


def split_data(
    df: pl.DataFrame,
    target_column: str = "Irradiance",
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, pl.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Args:
        df: DataFrame with all data
        target_column: Name of target column
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for test
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, (
        "train_size + val_size + test_size must equal 1.0"
    )

    total_samples = len(df)
    indices = list(range(total_samples))

    import random

    rng = random.Random(random_state)
    rng.shuffle(indices)

    train_end = int(total_samples * train_size)
    val_end = train_end + int(total_samples * val_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    splits = {
        "train": df[train_indices],
        "val": df[val_indices],
        "test": df[test_indices],
    }

    print(f"\nData split sizes:")
    print(
        f"  Train: {len(splits['train'])} ({len(splits['train']) / len(df) * 100:.1f}%)"
    )
    print(f"  Val:   {len(splits['val'])} ({len(splits['val']) / len(df) * 100:.1f}%)")
    print(
        f"  Test:  {len(splits['test'])} ({len(splits['test']) / len(df) * 100:.1f}%)"
    )

    # Verify target distribution
    print(f"\nTarget ({target_column}) statistics:")
    for split_name, split_df in splits.items():
        mean_val = split_df[target_column].mean()
        std_val = split_df[target_column].std()
        print(f"  {split_name:5s}: mean={mean_val:.2f}, std={std_val:.2f}")

    return splits


def get_feature_columns(df: pl.DataFrame, exclude_columns: list = None) -> list:
    """
    Get list of numeric feature columns, excluding non-feature columns.

    Args:
        df: DataFrame
        exclude_columns: Additional columns to exclude

    Returns:
        List of numeric feature column names
    """
    default_exclude = [
        "PictureName",
        "DateTime",
        "DateTime_clean",
        "Month",
        "ImageExists",
        "Irradiance",
        "IrradianceNotCompensated",
    ]

    if exclude_columns:
        default_exclude.extend(exclude_columns)

    numeric_types = [
        pl.Float32,
        pl.Float64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ]

    feature_cols = [
        col
        for col in df.columns
        if col not in default_exclude and df[col].dtype in numeric_types
    ]

    return feature_cols
