from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

try:
    import intel_extension_for_pytorch as ipex

    IPEX_AVAILABLE = True
except (ImportError, OSError):
    IPEX_AVAILABLE = False


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    predictions = []
    targets = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)

        predictions.extend(output.detach().cpu().numpy())
        targets.extend(target.cpu().numpy())

        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader.dataset)
    predictions = np.array(predictions)
    targets = np.array(targets)

    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        "loss": avg_loss,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions,
        "targets": targets,
    }


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device, dtype=torch.float32)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    predictions = np.array(predictions)
    targets = np.array(targets)

    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        "loss": avg_loss,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions,
        "targets": targets,
    }


def train_model(
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

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_metrics = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_mse"].append(train_metrics["mse"])
        history["train_mae"].append(train_metrics["mae"])
        history["train_rmse"].append(train_metrics["rmse"])
        history["train_r2"].append(train_metrics["r2"])

        history["val_loss"].append(val_metrics["loss"])
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_r2"].append(val_metrics["r2"])

        if verbose:
            print(f"\nEpoch {epoch}/{epochs}:")
            print(
                f"  Train - Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}, "
                f"MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}, R2: {train_metrics['r2']:.4f}"
            )
            print(
                f"  Val   - Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}, "
                f"MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, R2: {val_metrics['r2']:.4f}"
            )

        if checkpoint_dir and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics["loss"],
                checkpoint_dir / "best_model.pth",
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping_patience and patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}")
            break

    return history


def create_optimizer_and_criterion(
    model,
    learning_rate=0.001,
    weight_decay=1e-4,
    optimizer_type="adam",
):
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    criterion = nn.MSELoss()

    if IPEX_AVAILABLE:
        try:
            device = next(model.parameters()).device
            if str(device).startswith("xpu"):
                model.train()
                model, optimizer = ipex.optimize(
                    model, optimizer=optimizer, dtype=torch.float32, level="O1"
                )
                print("Model and optimizer optimized with Intel Extension for PyTorch (O1)")
        except Exception as e:
            print(f"Could not optimize with IPEX: {e}")

    return optimizer, criterion


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        filepath,
    )


def load_checkpoint(model, optimizer, filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]
