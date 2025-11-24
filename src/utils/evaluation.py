import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ("loss", "Loss"),
        ("mse", "MSE"),
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("r2", "R²"),
    ]

    for idx, (metric, label) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        epochs = range(1, len(history[f"train_{metric}"]) + 1)
        ax.plot(epochs, history[f"train_{metric}"], "b-", label="Train", linewidth=2)
        ax.plot(epochs, history[f"val_{metric}"], "r-", label="Validation", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{label} over Epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_residuals(targets, predictions, split_name="Test", save_path=None):
    residuals = targets - predictions

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].scatter(predictions, targets, alpha=0.5, s=20)
    axes[0, 0].plot(
        [targets.min(), targets.max()],
        [targets.min(), targets.max()],
        "r--",
        linewidth=2,
        label="Perfect fit",
    )
    axes[0, 0].set_xlabel("Predicted Irradiance")
    axes[0, 0].set_ylabel("Actual Irradiance")
    axes[0, 0].set_title(f"{split_name}: Predicted vs Actual")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(predictions, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0, 1].set_xlabel("Predicted Irradiance")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title(f"{split_name}: Residual Plot")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[1, 0].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[1, 0].set_xlabel("Residuals")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title(f"{split_name}: Residuals Distribution")
    axes[1, 0].grid(True, alpha=0.3)

    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f"{split_name}: Q-Q Plot")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_predictions_comparison(
    targets, predictions, split_name="Test", n_samples=100, save_path=None
):
    indices = np.random.choice(
        len(targets), min(n_samples, len(targets)), replace=False
    )
    indices = np.sort(indices)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(indices))
    ax.plot(x, targets[indices], "b-o", label="Actual", markersize=4, linewidth=1.5)
    ax.plot(
        x, predictions[indices], "r-s", label="Predicted", markersize=4, linewidth=1.5
    )
    ax.fill_between(x, targets[indices], predictions[indices], alpha=0.3)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Irradiance")
    ax.set_title(
        f"{split_name}: Actual vs Predicted Values (Random {len(indices)} samples)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
