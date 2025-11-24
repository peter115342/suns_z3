import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def set_plotting_style():
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10


def plot_target_distribution(df: pl.DataFrame, target_column: str = "Irradiance"):
    set_plotting_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    values = df[target_column].to_numpy()
    axes[0].hist(values, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel(target_column)
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Distribution of {target_column}")
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(values)
    axes[1].set_ylabel(target_column)
    axes[1].set_title(f"Box Plot of {target_column}")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_vs_target(
    df: pl.DataFrame, feature_column: str, target_column: str = "Irradiance"
):
    set_plotting_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = df[feature_column].to_numpy()
    y = df[target_column].to_numpy()

    ax.scatter(x, y, alpha=0.5, s=20)
    ax.set_xlabel(feature_column)
    ax.set_ylabel(target_column)
    ax.set_title(f"{target_column} vs {feature_column}")
    ax.grid(True, alpha=0.3)

    corr = np.corrcoef(x, y)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {corr:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pl.DataFrame, feature_columns: list[str], target_column: str = "Irradiance"
):
    set_plotting_style()

    cols = feature_columns + [target_column]
    data = df.select(cols).to_numpy()

    corr_matrix = np.corrcoef(data.T)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation", rotation=270, labelpad=20)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    return fig


def plot_monthly_distribution(df: pl.DataFrame, target_column: str = "Irradiance"):
    set_plotting_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    months = df["Month"].unique().sort()

    data_by_month = []
    for month in months:
        month_data = df.filter(pl.col("Month") == month)[target_column].to_numpy()
        data_by_month.append(month_data)

    ax.boxplot(data_by_month, labels=months)
    ax.set_xlabel("Month")
    ax.set_ylabel(target_column)
    ax.set_title(f"{target_column} Distribution by Month")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def display_sample_images(
    df: pl.DataFrame,
    data_dir,
    n_samples: int = 6,
    random_state: int | None = 42,
):
    from data.loading import get_image_path
    from PIL import Image

    if random_state is not None:
        sample_df = df.sample(n=min(n_samples, len(df)), seed=random_state)
    else:
        sample_df = df.sample(n=min(n_samples, len(df)))

    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]

    for idx, row in enumerate(sample_df.iter_rows(named=True)):
        if idx >= n_samples:
            break

        img_path = get_image_path(row["PictureName"], data_dir, row["Month"])

        if img_path.exists():
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(
                f"Irradiance: {row['Irradiance']:.2f}\n"
                f"Month: {row['Month']}, {row['DateTime_clean'][:10]}"
            )
        else:
            axes[idx].text(0.5, 0.5, "Image not found", ha="center", va="center")
            axes[idx].set_title(f"Month: {row['Month']}")

        axes[idx].axis("off")

    for idx in range(n_samples, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig


def plot_feature_importance_vs_target(
    df: pl.DataFrame, feature_columns: list[str], target_column: str = "Irradiance"
):
    set_plotting_style()

    correlations = []
    for feat in feature_columns:
        x = df[feat].to_numpy()
        y = df[target_column].to_numpy()
        corr = np.corrcoef(x, y)[0, 1]
        correlations.append((feat, abs(corr), corr))

    correlations.sort(key=lambda x: x[1], reverse=True)

    features = [c[0] for c in correlations]
    corr_values = [c[2] for c in correlations]

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))

    colors = ["green" if c > 0 else "red" for c in corr_values]
    ax.barh(features, corr_values, color=colors, alpha=0.7)
    ax.set_xlabel(f"Correlation with {target_column}")
    ax.set_title(f"Feature Correlations with {target_column}")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig
