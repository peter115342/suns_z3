"""Feature extraction using pretrained models from ImageNet."""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np
import polars as pl
from PIL import Image

try:
    import intel_extension_for_pytorch as ipex

    IPEX_AVAILABLE = True
except (ImportError, OSError):
    IPEX_AVAILABLE = False


class FeatureExtractor:
    """Extract features from images using pretrained models."""

    def __init__(self, model_name="mobilenet_v2", device=None):
        """
        Initialize feature extractor.

        Args:
            model_name: Name of pretrained model (mobilenet_v2, resnet18, efficientnet_b0)
            device: Device to use for extraction
        """
        if device is None:
            from .utils import get_device

            device = get_device()

        self.device = device
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.model.eval()

        self.normalize = torch.nn.functional.normalize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    def _load_model(self, model_name):
        """Load pretrained model and remove classification head."""
        if model_name == "mobilenet_v2":
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            model = mobilenet_v2(weights=weights)
            self.feature_dim = 1280
            model = nn.Sequential(*list(model.children())[:-1])

        elif model_name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights

            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
            self.feature_dim = 512
            model = nn.Sequential(*list(model.children())[:-1])

        elif model_name == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = efficientnet_b0(weights=weights)
            self.feature_dim = 1280
            model = nn.Sequential(*list(model.children())[:-1])

        else:
            raise ValueError(f"Neznámy model: {model_name}")

        model = model.to(self.device)
        model.eval()

        # IPEX optimize() vypnuté - spôsobuje chybu "could not create a primitive"
        # Model stále beží na XPU, len bez dodatočnej optimalizácie
        # if IPEX_AVAILABLE and str(self.device).startswith("xpu"):
        #     model = ipex.optimize(model, dtype=torch.float32)
        #     print(f"Model {model_name} optimalizovaný s IPEX")

        return model

    def extract_features(self, image_path):
        """
        Extract features from a single image.

        Args:
            image_path: Path to image file

        Returns:
            Feature vector as numpy array
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)

        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        img_tensor = (img_tensor - self.mean) / self.std
        img_tensor = img_tensor.unsqueeze(0)

        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
            features = features.flatten()

        return features.cpu().numpy()

    def extract_features_from_dataframe(
        self,
        df: pl.DataFrame,
        data_dir: Path,
        use_resized: bool = True,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        """
        Extract features from all images in dataframe.

        Args:
            df: DataFrame with image metadata
            data_dir: Path to image directory
            use_resized: Whether to use pre-resized images
            batch_size: Batch size for extraction
            verbose: Show progress bar

        Returns:
            DataFrame with features and metadata
        """
        all_features = []
        all_image_paths = []
        all_targets = []

        iterator = range(0, len(df), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc=f"Extrakcia príznakov ({self.model_name})")

        for i in iterator:
            batch_df = df[i : i + batch_size]
            batch_images = []
            batch_paths = []

            for row in batch_df.iter_rows(named=True):
                if (
                    use_resized
                    and "ResizedImagePath" in row
                    and row["ResizedImagePath"] is not None
                ):
                    img_path = Path(row["ResizedImagePath"])
                else:
                    month_str = str(row["Month"]).zfill(2)
                    img_path = data_dir / month_str / "original" / row["PictureName"]

                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((224, 224), Image.BILINEAR)
                    img_tensor = (
                        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    )
                    img_tensor = (img_tensor - self.mean.cpu()) / self.std.cpu()
                    batch_images.append(img_tensor)
                    batch_paths.append(str(img_path))
                    all_targets.append(row["Irradiance"])
                except Exception as e:
                    if verbose:
                        print(f"\nChyba pri načítaní {img_path}: {e}")
                    continue

            if len(batch_images) == 0:
                continue

            batch_tensor = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                features = self.model(batch_tensor)
                features = features.reshape(features.size(0), -1)  # Flatten

            all_features.append(features.cpu().numpy())
            all_image_paths.extend(batch_paths)

        features_array = np.vstack(all_features)

        feature_cols = {
            f"feature_{i}": features_array[:, i] for i in range(features_array.shape[1])
        }
        feature_cols["image_path"] = all_image_paths
        feature_cols["Irradiance"] = all_targets

        features_df = pl.DataFrame(feature_cols)

        if verbose:
            print(
                f"\n Extrakcia dokončená: {len(features_df)} vzoriek, {features_array.shape[1]} príznakov"
            )

        return features_df

    def get_model_info(self):
        """Get information about the model."""
        param_count = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": self.model_name,
            "feature_dim": self.feature_dim,
            "parameters": param_count,
            "device": str(self.device),
        }
