"""Clustering utilities for feature analysis."""

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
from pathlib import Path


def reduce_dimensionality(features, method="pca", n_components=50, random_state=42):
    """
    Reduce dimensionality of features.

    Args:
        features: Feature array
        method: 'pca' or 'tsne'
        n_components: Number of components
        random_state: Random seed

    Returns:
        Reduced features
    """
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(features)
        variance_explained = reducer.explained_variance_ratio_.sum()
        print(
            f"PCA: {n_components} komponentov vysvetľuje {variance_explained:.2%} variancie"
        )
        return reduced, reducer

    elif method == "tsne":
        # TSNE is slow, use PCA first if features are high-dimensional
        if features.shape[1] > 50:
            pca = PCA(n_components=50, random_state=random_state)
            features = pca.fit_transform(features)
            print(
                f"PCA preprocessing: 50 komponentov vysvetľuje {pca.explained_variance_ratio_.sum():.2%}"
            )

        reducer = TSNE(n_components=n_components, random_state=random_state, verbose=1)
        reduced = reducer.fit_transform(features)
        return reduced, reducer

    else:
        raise ValueError(f"Neznáma metóda: {method}")


def cluster_features(features, n_clusters=5, method="kmeans", random_state=42):
    """
    Cluster features using specified method.

    Args:
        features: Feature array
        n_clusters: Number of clusters
        method: 'kmeans', 'hierarchical', or 'dbscan'
        random_state: Random seed

    Returns:
        Cluster labels
    """
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = clusterer.fit_predict(features)

    elif method == "hierarchical":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(features)

    elif method == "dbscan":
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        labels = clusterer.fit_predict(features)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"DBSCAN našiel {n_clusters} zhlukov")

    else:
        raise ValueError(f"Neznáma metóda: {method}")

    return labels, clusterer


def visualize_clusters_2d(features_2d, labels, targets=None, save_path=None):
    """
    Visualize clusters in 2D space.

    Args:
        features_2d: 2D feature array
        labels: Cluster labels
        targets: Optional target values for coloring
        save_path: Path to save figure
    """
    n_plots = 2 if targets is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))

    if n_plots == 1:
        axes = [axes]

    # Plot by cluster
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        axes[0].scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[color],
            label=f"Zhluk {label}",
            alpha=0.6,
            s=30,
        )

    axes[0].set_xlabel("Komponent 1")
    axes[0].set_ylabel("Komponent 2")
    axes[0].set_title("Zhluky v 2D priestore")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Plot by target value if available
    if targets is not None:
        scatter = axes[1].scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=targets,
            cmap="viridis",
            alpha=0.6,
            s=30,
        )
        axes[1].set_xlabel("Komponent 1")
        axes[1].set_ylabel("Komponent 2")
        axes[1].set_title("Vzorky podľa Irradiance")
        plt.colorbar(scatter, ax=axes[1], label="Irradiance")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Graf uložený do {save_path}")

    return fig


def display_cluster_samples(
    df,
    data_dir=None,
    cluster_col="Cluster",
    image_path_col="image_path",
    cluster_id=None,
    n_samples=9,
    save_path=None,
):
    """
    Display sample images from clusters.

    Args:
        df: DataFrame with cluster assignments and image paths
        data_dir: Path to data directory (optional, for constructing paths)
        cluster_col: Name of cluster column
        image_path_col: Name of image path column
        cluster_id: Specific cluster to display (None = all clusters)
        n_samples: Number of samples per cluster
        save_path: Path to save figure
    """
    if cluster_id is not None:
        # Display single cluster
        cluster_df = df.filter(pl.col(cluster_col) == cluster_id)
        unique_clusters = [cluster_id]
        n_clusters = 1
    else:
        unique_clusters = sorted(df[cluster_col].unique().to_list())
        n_clusters = len(unique_clusters)

    fig, axes = plt.subplots(
        n_clusters, n_samples, figsize=(n_samples * 2, n_clusters * 2)
    )

    if n_clusters == 1:
        axes = np.array(axes).reshape(1, -1)

    for cluster_idx, cluster_label in enumerate(unique_clusters):
        cluster_df = df.filter(pl.col(cluster_col) == cluster_label)
        total_in_cluster = len(cluster_df)

        # Sample randomly
        if len(cluster_df) > n_samples:
            cluster_df = cluster_df.sample(n=n_samples, seed=42)

        for img_idx, row in enumerate(cluster_df.head(n_samples).iter_rows(named=True)):
            if img_idx >= n_samples:
                break

            # Try to get image path from image_path column or construct it
            if image_path_col in row and row[image_path_col]:
                img_path = Path(row[image_path_col])
            elif data_dir is not None and "Month" in row and "PictureName" in row:
                month_str = str(row["Month"]).zfill(2)
                img_path = data_dir / month_str / "original" / row["PictureName"]
            else:
                axes[cluster_idx, img_idx].text(
                    0.5, 0.5, "Cesta chýba", ha="center", va="center"
                )
                axes[cluster_idx, img_idx].axis("off")
                continue

            try:
                img = Image.open(img_path)
                # Resize for display
                img.thumbnail((200, 200))
                axes[cluster_idx, img_idx].imshow(img)
                axes[cluster_idx, img_idx].axis("off")

                if img_idx == 0:
                    axes[cluster_idx, img_idx].set_title(
                        f"Zhluk {cluster_label}\n(n={total_in_cluster})",
                        fontsize=10,
                        fontweight="bold",
                    )
            except Exception as e:
                axes[cluster_idx, img_idx].text(
                    0.5, 0.5, "Chyba", ha="center", va="center"
                )
                axes[cluster_idx, img_idx].axis("off")

        # Hide unused subplots
        for img_idx in range(min(len(cluster_df), n_samples), n_samples):
            axes[cluster_idx, img_idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Ukážky zhlukov uložené do {save_path}")

    return fig


def compute_average_images(
    df,
    data_dir=None,
    cluster_col="Cluster",
    image_path_col="image_path",
    save_path=None,
):
    """
    Compute average images for each cluster.

    Args:
        df: DataFrame with cluster assignments and image paths
        data_dir: Path to data directory (optional, for constructing paths)
        cluster_col: Name of cluster column
        image_path_col: Name of image path column
        save_path: Path to save figure

    Returns:
        Dictionary of {cluster_id: average_image}
    """
    unique_clusters = sorted(df[cluster_col].unique().to_list())
    average_images = {}

    for cluster_label in unique_clusters:
        cluster_df = df.filter(pl.col(cluster_col) == cluster_label)

        # Load all images in cluster
        images = []
        for row in cluster_df.iter_rows(named=True):
            # Try to get image path from image_path column or construct it
            if image_path_col in row and row[image_path_col]:
                img_path = Path(row[image_path_col])
            elif data_dir is not None and "Month" in row and "PictureName" in row:
                month_str = str(row["Month"]).zfill(2)
                img_path = data_dir / month_str / "original" / row["PictureName"]
            else:
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((128, 128), Image.BILINEAR)
                images.append(np.array(img))
            except Exception:
                continue

        if len(images) > 0:
            # Compute average
            avg_img = np.mean(images, axis=0).astype(np.uint8)
            average_images[cluster_label] = avg_img

    return average_images


def analyze_clusters(df, cluster_col="Cluster", target_col="Irradiance"):
    """
    Analyze cluster statistics.

    Args:
        df: DataFrame with cluster assignments and targets
        cluster_col: Name of cluster column
        target_col: Name of target column

    Returns:
        Statistics DataFrame
    """
    stats = (
        df.group_by(cluster_col)
        .agg(
            [
                pl.count().alias("count"),
                pl.col(target_col).mean().alias("mean_irradiance"),
                pl.col(target_col).std().alias("std_irradiance"),
                pl.col(target_col).min().alias("min_irradiance"),
                pl.col(target_col).max().alias("max_irradiance"),
            ]
        )
        .sort(cluster_col)
    )

    return stats
