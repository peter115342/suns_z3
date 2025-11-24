import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def visualize_conv_filters(model, layer_idx=0, max_filters=32, save_path=None):
    """
    Vizualizuje konvolučné filtre z konkrétnej vrstvy modelu.

    Args:
        model: Natrénovaný CNN model
        layer_idx: Index konvolučného bloku na vizualizáciu
        max_filters: Maximálny počet filtrov na zobrazenie
        save_path: Cesta pre uloženie obrázku

    Returns:
        matplotlib figure objekt
    """
    conv_block = model.conv_layers[layer_idx]
    conv_layer = None

    for module in conv_block:
        if isinstance(module, torch.nn.Conv2d):
            conv_layer = module
            break

    if conv_layer is None:
        print("Konvolučná vrstva nebola nájdená!")
        return None

    filters = conv_layer.weight.data.cpu()
    num_filters = min(filters.shape[0], max_filters)

    print(
        f"Vrstva {layer_idx}: {filters.shape[0]} filtrov s tvarom {filters.shape[1:]} (zobrazených {num_filters})"
    )

    grid_size = int(np.ceil(np.sqrt(num_filters)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle(f"Konvolučné filtre - Vrstva {layer_idx}", fontsize=16)

    for i in range(grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        ax = axes[row, col] if grid_size > 1 else axes

        if i < num_filters:
            filter_img = filters[i]

            if filter_img.shape[0] == 3:
                filter_img = filter_img.permute(1, 2, 0)
                filter_img = (filter_img - filter_img.min()) / (
                    filter_img.max() - filter_img.min()
                )
                ax.imshow(filter_img.numpy())
            else:
                filter_img = filter_img.mean(dim=0)
                filter_img = (filter_img - filter_img.min()) / (
                    filter_img.max() - filter_img.min()
                )
                ax.imshow(filter_img.numpy(), cmap="viridis")

            ax.set_title(f"Filter {i}", fontsize=8)

        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Filtre uložené do {save_path}")

    return fig


def predict_custom_images(model, images_dir, device, image_size=(224, 224)):
    """
    Predikuje slnečné žiarenie pre vlastné obrázky.

    Args:
        model: Natrénovaný CNN model
        images_dir: Cesta k priečinku s vlastnými obrázkami
        device: Zariadenie pre spustenie predikcií
        image_size: Veľkosť pre zmenu veľkosti obrázkov

    Returns:
        List slovníkov s informáciami o obrázkoch a predikciami
    """
    image_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []

    if images_dir.exists():
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))

    if len(image_files) == 0:
        print(f"Žiadne obrázky neboli nájdené v {images_dir}")
        print("Prosím pridajte aspoň 5 vlastných fotografií oblohy do tohto priečinka.")
        return []

    print(f"Nájdených {len(image_files)} obrázkov v {images_dir}")

    results = []
    model.eval()

    with torch.no_grad():
        for img_path in sorted(image_files):
            try:
                image = Image.open(img_path).convert("RGB")
                original_size = image.size
                image_resized = image.resize(image_size, Image.BILINEAR)

                image_tensor = (
                    torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float()
                    / 255.0
                )
                image_tensor = (image_tensor - image_mean) / image_std
                image_tensor = image_tensor.unsqueeze(0).to(device)

                prediction = model(image_tensor)
                predicted_irradiance = prediction.item()

                results.append(
                    {
                        "filename": img_path.name,
                        "path": img_path,
                        "original_size": original_size,
                        "predicted_irradiance": predicted_irradiance,
                        "image": image_resized,
                    }
                )

                print(f"  {img_path.name}: {predicted_irradiance:.2f} W/m²")

            except Exception as e:
                print(f"  Chyba pri spracovaní {img_path.name}: {e}")

    return results


def visualize_custom_predictions(custom_results, save_path=None):
    """
    Vizualizuje vlastné obrázky s predikciami.

    Args:
        custom_results: List slovníkov s výsledkami predikcií
        save_path: Cesta pre uloženie obrázku

    Returns:
        matplotlib figure objekt
    """
    if len(custom_results) == 0:
        print("Žiadne vlastné obrázky na vizualizáciu.")
        return None

    num_images = len(custom_results)
    cols = min(3, num_images)
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(
        "Vlastné obrázky oblohy - Predikované slnečné žiarenie", fontsize=16, y=1.02
    )

    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes

    for idx, result in enumerate(custom_results):
        ax = axes[idx] if num_images > 1 else axes[0]

        ax.imshow(result["image"])
        ax.set_title(
            f"{result['filename']}\nPredikované: {result['predicted_irradiance']:.2f} W/m²",
            fontsize=10,
        )
        ax.axis("off")

    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nVizualizácia vlastných obrázkov uložená do {save_path}")

    predictions = [r["predicted_irradiance"] for r in custom_results]
    print("\nŠtatistiky predikcií:")
    print(f"  Priemerne žiarenie: {np.mean(predictions):.2f} W/m²")
    print(f"  Min žiarenie:       {np.min(predictions):.2f} W/m²")
    print(f"  Max žiarenie:       {np.max(predictions):.2f} W/m²")
    print(f"  Štandardná odchýlka: {np.std(predictions):.2f} W/m²")

    return fig
