from pathlib import Path
from PIL import Image
import polars as pl
from tqdm import tqdm


def resize_and_save_images(
    df: pl.DataFrame,
    data_dir: Path,
    output_dir: Path,
    image_size: tuple = (224, 224),
    verbose: bool = True,
):
    """
    Resize all images in the dataset and save them to a new directory.

    Args:
        df: DataFrame containing image metadata
        data_dir: Path to the original images
        output_dir: Path where resized images will be saved
        image_size: Target size for resizing (width, height)
        verbose: Whether to show progress bar

    Returns:
        Updated DataFrame with ResizedImagePath column
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    resized_paths = []

    iterator = df.iter_rows(named=True)
    if verbose:
        iterator = tqdm(iterator, total=len(df), desc="Resizing images")

    for row in iterator:
        month_str = str(row["Month"]).zfill(2)
        original_path = data_dir / month_str / "original" / row["PictureName"]

        month_output_dir = output_dir / month_str
        month_output_dir.mkdir(parents=True, exist_ok=True)

        resized_path = month_output_dir / row["PictureName"]

        if not resized_path.exists():
            try:
                with Image.open(original_path) as img:
                    img_rgb = img.convert("RGB")
                    img_resized = img_rgb.resize(image_size, Image.BILINEAR)
                    img_resized.save(resized_path, quality=95)
            except Exception as e:
                if verbose:
                    print(f"\nError processing {original_path}: {e}")
                resized_paths.append(None)
                continue

        resized_paths.append(str(resized_path))

    df_with_paths = df.with_columns(pl.Series("ResizedImagePath", resized_paths))

    return df_with_paths
