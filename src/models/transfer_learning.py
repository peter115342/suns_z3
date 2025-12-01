"""Transfer learning with fine-tuning pretrained models."""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except (ImportError, OSError):
    IPEX_AVAILABLE = False


class TransferLearningModel(nn.Module):
    """Transfer learning model with frozen backbone and trainable head."""

    def __init__(self, model_name="mobilenet_v2", hidden_sizes=[256, 128], dropout=0.5):
        super(TransferLearningModel, self).__init__()

        self.model_name = model_name
        self.backbone = self._load_backbone(model_name)

        for param in self.backbone.parameters():
            param.requires_grad = False

        layers = []
        prev_size = self.feature_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.head = nn.Sequential(*layers)

    def _load_backbone(self, model_name):
        """Load pretrained backbone."""
        if model_name == "mobilenet_v2":
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            model = mobilenet_v2(weights=weights)
            self.feature_dim = 1280
            backbone = nn.Sequential(*list(model.children())[:-1])

        elif model_name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights

            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
            self.feature_dim = 512
            backbone = nn.Sequential(*list(model.children())[:-1])

        elif model_name == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = efficientnet_b0(weights=weights)
            self.feature_dim = 1280
            backbone = nn.Sequential(*list(model.children())[:-1])

        else:
            raise ValueError(f"Neznámy model: {model_name}")

        return backbone

    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(1)

        output = self.head(features)

        return output.squeeze(1)

    def unfreeze_last_n_layers(self, n=2):
        """Unfreeze last n layers of backbone for fine-tuning."""
        backbone_modules = list(self.backbone.modules())

        for module in backbone_modules[-n:]:
            for param in module.parameters():
                param.requires_grad = True

        print(f"Uvoľnené posledné {n} vrstvy backbone siete")


def create_transfer_learning_model(
    model_name="mobilenet_v2", hidden_sizes=[256, 128], dropout=0.5, device=None
):
    """
    Create transfer learning model.

    Args:
        model_name: Name of pretrained model
        hidden_sizes: Hidden layer sizes for regression head
        dropout: Dropout rate
        device: Device to use

    Returns:
        Model on device
    """
    if device is None:
        from .utils import get_device

        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    model = TransferLearningModel(model_name, hidden_sizes, dropout)
    model = model.to(device)

    return model


def train_transfer_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    device,
    checkpoint_dir=None,
    early_stopping_patience=None,
    verbose=True,
):
    """
    Train transfer learning model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss criterion
        optimizer: Optimizer
        epochs: Number of epochs
        device: Device to use
        checkpoint_dir: Directory to save checkpoints
        early_stopping_patience: Early stopping patience
        verbose: Print progress

    Returns:
        Training history
    """
    history = {
        "train_loss": [],
        "train_mse": [],
        "train_mae": [],
        "train_rmse": [],
        "train_r2": [],
        "val_loss": [],
        "val_mse": [],
        "val_mae": [],
        "val_rmse": [],
        "val_r2": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    if IPEX_AVAILABLE and str(device).startswith("xpu"):
        model.train()
        model, optimizer = ipex.optimize(
            model, optimizer=optimizer, dtype=torch.float32, level="O1"
        )
        print("Model a optimizer optimalizované s IPEX")

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        pbar = (
            tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            if verbose
            else train_loader
        )

        for data, target in pbar:
            data, target = data.to(device), target.to(device).float()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            train_preds.extend(output.detach().cpu().numpy())
            train_targets.extend(target.cpu().numpy())

            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader.dataset)
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)

        train_mse = np.mean((train_preds - train_targets) ** 2)
        train_mae = np.mean(np.abs(train_preds - train_targets))
        train_rmse = np.sqrt(train_mse)
        train_r2 = 1 - (
            np.sum((train_targets - train_preds) ** 2)
            / np.sum((train_targets - np.mean(train_targets)) ** 2)
        )

        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).float()
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item() * data.size(0)
                val_preds.extend(output.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        val_mse = np.mean((val_preds - val_targets) ** 2)
        val_mae = np.mean(np.abs(val_preds - val_targets))
        val_rmse = np.sqrt(val_mse)
        val_r2 = 1 - (
            np.sum((val_targets - val_preds) ** 2)
            / np.sum((val_targets - np.mean(val_targets)) ** 2)
        )

        history["train_loss"].append(train_loss)
        history["train_mse"].append(train_mse)
        history["train_mae"].append(train_mae)
        history["train_rmse"].append(train_rmse)
        history["train_r2"].append(train_r2)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_r2"].append(val_r2)

        if verbose:
            print(f"Epoch {epoch}/{epochs}:")
            print(
                f"  Train - Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}"
            )
            print(
                f"  Val   - Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}"
            )

        if checkpoint_dir and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint_path = Path(checkpoint_dir) / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

            if verbose:
                print(f"  ✓ Uložený najlepší model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        if early_stopping_patience and patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping po {epoch} epochách")
            break

    return history
