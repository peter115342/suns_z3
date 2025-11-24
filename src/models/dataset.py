from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import polars as pl


class SolarDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        data_dir: Path,
        target_column: str = "Irradiance",
        image_size: tuple = (224, 224),
        transform=None,
        normalize=True,
    ):
        self.df = df
        self.data_dir = data_dir
        self.target_column = target_column
        self.image_size = image_size
        self.transform = transform
        self.normalize = normalize

        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)

        month_str = str(row["Month"]).zfill(2)
        image_path = self.data_dir / month_str / "original" / row["PictureName"]

        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size, Image.BILINEAR)

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        if self.normalize:
            image = (image - self.image_mean) / self.image_std

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(float(row[self.target_column]), dtype=torch.float32)

        return image, target


def create_data_loaders(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    data_dir: Path,
    batch_size: int = 32,
    image_size: tuple = (224, 224),
    num_workers: int = 0,
    normalize: bool = True,
):
    train_dataset = SolarDataset(
        train_df,
        data_dir,
        image_size=image_size,
        normalize=normalize,
    )

    val_dataset = SolarDataset(
        val_df,
        data_dir,
        image_size=image_size,
        normalize=normalize,
    )

    test_dataset = SolarDataset(
        test_df,
        data_dir,
        image_size=image_size,
        normalize=normalize,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
