# %%
import sys
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))

from data.loading import load_all_months, verify_image_existence
from data.preprocessing import split_data, get_feature_columns
from utils.visualization import (
    plot_target_distribution,
    plot_correlation_matrix,
    plot_monthly_distribution,
    display_sample_images,
    plot_feature_importance_vs_target,
)

DATA_DIR = project_root / "dataset" / "Solar_data"
MONTHS = ["01", "04", "07", "10"]

print(f"Project root: {project_root}")
print(f"Data directory: {DATA_DIR}")
print(f"Data directory exists: {DATA_DIR.exists()}")

# %% [markdown]
# ## 1. Načítanie dát

# %%
# Načítanie dát
df_all = load_all_months(DATA_DIR, MONTHS)

print("\nPrvých 5 riadkov:")
print(df_all.head())

print("\nInformácie o DataFrame:")
print(f"Počet riadkov: {len(df_all)}")
print(f"Počet stĺpcov: {len(df_all.columns)}")
print(f"\nStĺpce: {df_all.columns}")

# %% [markdown]
# ## 2. Základná štatistika
#
# Pozrime sa na základné štatistické charakteristiky dát.

# %%
print("Štatistický prehľad:")
print(df_all.describe())

print("\n\nChýbajúce hodnoty:")
null_counts = df_all.null_count()
print(null_counts)

# %% [markdown]
# ## 3. Analýza cieľovej premennej: Irradiance
#
# Analyzujeme distribúciu slnečného žiarenia.

# %%
fig_dir = project_root / "figures"
fig_dir.mkdir(exist_ok=True)

target_stats = df_all.select(
    [
        pl.col("Irradiance").min().alias("min"),
        pl.col("Irradiance").max().alias("max"),
        pl.col("Irradiance").mean().alias("mean"),
        pl.col("Irradiance").median().alias("median"),
        pl.col("Irradiance").std().alias("std"),
    ]
)

print("Štatistiky Irradiance:")
print(target_stats)

fig = plot_target_distribution(df_all, "Irradiance")
plt.savefig(fig_dir / "eda_target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Distribúcia žiarenia podľa mesiacov
#
# Porovnáme žiarenie v rôznych mesiacoch.

# %%
monthly_stats = (
    df_all.group_by("Month")
    .agg(
        [
            pl.col("Irradiance").count().alias("count"),
            pl.col("Irradiance").mean().alias("mean"),
            pl.col("Irradiance").std().alias("std"),
            pl.col("Irradiance").min().alias("min"),
            pl.col("Irradiance").max().alias("max"),
        ]
    )
    .sort("Month")
)

print("Štatistiky žiarenia po mesiacoch:")
print(monthly_stats)

fig = plot_monthly_distribution(df_all, "Irradiance")
plt.savefig(fig_dir / "eda_monthly_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Verifikácia existencie obrázkov
#
# Overiť, či všetky obrázky skutočne existujú v dataset.

# %%
df_all = verify_image_existence(df_all, DATA_DIR)

total_records = len(df_all)
existing_images = df_all.filter(pl.col("ImageExists")).shape[0]
missing_images = total_records - existing_images

print(f"Celkový počet záznamov: {total_records}")
print(f"Záznamy s existujúcimi obrázkami: {existing_images}")
print(f"Záznamy s chýbajúcimi obrázkami: {missing_images}")

if missing_images > 0:
    print("\nUpozornenie: Niektoré obrázky chýbajú!")
    missing_df = df_all.filter(~pl.col("ImageExists"))
    print("Chýbajúce obrázky po mesiacoch:")
    print(missing_df.group_by("Month").agg(pl.count().alias("missing_count")))

df_valid = df_all.filter(pl.col("ImageExists"))
print(f"\nPočet validných záznamov: {len(df_valid)}")

# %% [markdown]
# ## 6. Ukážky obrázkov
#
# Zobrazíme náhodné vzorky obrázkov s rôznymi hodnotami žiarenia.

# %%
fig = display_sample_images(df_valid, DATA_DIR, n_samples=6, random_state=42)
plt.savefig(fig_dir / "eda_sample_images.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nObrázky s najnižším žiarením:")
low_irradiance = df_valid.sort("Irradiance").head(3)
fig = display_sample_images(low_irradiance, DATA_DIR, n_samples=3, random_state=None)
plt.savefig(fig_dir / "eda_low_irradiance_samples.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nObrázky s najvyšším žiarením:")
high_irradiance = df_valid.sort("Irradiance", descending=True).head(3)
fig = display_sample_images(high_irradiance, DATA_DIR, n_samples=3, random_state=None)
plt.savefig(fig_dir / "eda_high_irradiance_samples.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Analýza numerických features
#
# Analyzujeme vzťahy medzi numerickými premennými a cieľovou premennou.

# %%
feature_cols = get_feature_columns(df_valid)
print(f"Numerické features ({len(feature_cols)}):")
for col in feature_cols:
    print(f"  - {col}")

fig = plot_feature_importance_vs_target(df_valid, feature_cols, "Irradiance")
plt.savefig(fig_dir / "eda_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Korelačná matica
#
# Vizualizujeme korelácie medzi všetkými numerickými premennými.

# %%
fig = plot_correlation_matrix(df_valid, feature_cols, "Irradiance")
plt.savefig(fig_dir / "eda_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Detailná analýza vybraných features
#
# Pozrime sa podrobnejšie na features s najvyššou koreláciou s cieľovou premennou.

# %%
correlations = []
for feat in feature_cols:
    x = df_valid[feat].to_numpy()
    y = df_valid["Irradiance"].to_numpy()
    corr = np.corrcoef(x, y)[0, 1]
    correlations.append((feat, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("Top 10 korelácií s Irradiance:")
for feat, corr in correlations[:10]:
    print(f"  {feat:25s}: {corr:+.4f}")

top_features = [c[0] for c in correlations[:6]]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, feat in enumerate(top_features):
    ax = axes[idx]
    x = df_valid[feat].to_numpy()
    y = df_valid["Irradiance"].to_numpy()
    corr = np.corrcoef(x, y)[0, 1]

    ax.scatter(x, y, alpha=0.3, s=10)
    ax.set_xlabel(feat)
    ax.set_ylabel("Irradiance")
    ax.set_title(f"Irradiance vs {feat}\n(corr={corr:.3f})")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "eda_top_features_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10. Rozdelenie dát na train/val/test
#
# Rozdelíme dáta na trénovaciu, validačnú a testovaciu množinu (70/15/15).

# %%
# Rozdelenie dát
splits = split_data(
    df_valid,
    target_column="Irradiance",
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
)

train_df = splits["train"]
val_df = splits["val"]
test_df = splits["test"]

print("\nOverenie rozdelenia:")
print(f"Train: {len(train_df)} samples")
print(f"Val:   {len(val_df)} samples")
print(f"Test:  {len(test_df)} samples")
print(f"Total: {len(train_df) + len(val_df) + len(test_df)} samples")

# %% [markdown]
# ## 11. Kontrola distribúcie po rozdelení
#
# Overiť, či je distribúcia cieľovej premennej konzistentná naprieč split-mi.

# %%
from scipy import stats

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (split_name, split_df) in enumerate(
    [("Train", train_df), ("Val", val_df), ("Test", test_df)]
):
    values = split_df["Irradiance"].to_numpy()

    axes[idx].hist(values, bins=50, edgecolor="black", alpha=0.7)
    axes[idx].set_xlabel("Irradiance")
    axes[idx].set_ylabel("Frequency")
    axes[idx].set_title(
        f"{split_name} Set\n(n={len(split_df)}, mean={values.mean():.2f})"
    )
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "eda_split_distributions.png", dpi=150, bbox_inches="tight")
plt.show()


ks_train_val = stats.ks_2samp(
    train_df["Irradiance"].to_numpy(), val_df["Irradiance"].to_numpy()
)
ks_train_test = stats.ks_2samp(
    train_df["Irradiance"].to_numpy(), test_df["Irradiance"].to_numpy()
)
ks_val_test = stats.ks_2samp(
    val_df["Irradiance"].to_numpy(), test_df["Irradiance"].to_numpy()
)

print("\nKolmogorov-Smirnov testy (distribúcie by mali byť podobné):")
print(
    f"Train vs Val:  statistic={ks_train_val.statistic:.4f}, p-value={ks_train_val.pvalue:.4f}"
)
print(
    f"Train vs Test: statistic={ks_train_test.statistic:.4f}, p-value={ks_train_test.pvalue:.4f}"
)
print(
    f"Val vs Test:   statistic={ks_val_test.statistic:.4f}, p-value={ks_val_test.pvalue:.4f}"
)

# %% [markdown]
# ## 12. Uloženie rozdelených dát
#
# Uložíme rozdelené datasety pre ďalšie použitie.

# %%
processed_dir = project_root / "dataset" / "processed"
processed_dir.mkdir(exist_ok=True)

train_df.write_csv(processed_dir / "train.csv")
val_df.write_csv(processed_dir / "val.csv")
test_df.write_csv(processed_dir / "test.csv")

print(f"Dáta uložené do: {processed_dir}")
print("  - train.csv")
print("  - val.csv")
print("  - test.csv")

# %% [markdown]
# ## 13. Predzpracovanie obrázkov: Resizing
#
# Pre zrýchlenie trénovania naresizujeme všetky obrázky na 224×224 px vopred.

# %%
from data.image_preprocessing import resize_and_save_images

resized_dir = project_root / "dataset" / "processed" / "resized"

print("Resizing obrázkov...")
print(f"Výstupný priečinok: {resized_dir}")

print("\nResizing train images...")
train_df_resized = resize_and_save_images(
    train_df, DATA_DIR, resized_dir / "train", image_size=(128, 128)
)

print("\nResizing val images...")
val_df_resized = resize_and_save_images(
    val_df, DATA_DIR, resized_dir / "val", image_size=(128, 128)
)

print("\nResizing test images...")
test_df_resized = resize_and_save_images(
    test_df, DATA_DIR, resized_dir / "test", image_size=(128, 128)
)
train_df_resized.write_csv(processed_dir / "train.csv")
val_df_resized.write_csv(processed_dir / "val.csv")
test_df_resized.write_csv(processed_dir / "test.csv")

print("\nVšetky obrázky naresizované a uložené!")
print(f"   Train: {len(train_df_resized)} obrázkov")
print(f"   Val:   {len(val_df_resized)} obrázkov")
print(f"   Test:  {len(test_df_resized)} obrázkov")
# %% [markdown]
# ## 1. Import knižníc

# %% [markdown]
# # Trénovanie CNN pre predikciu slnečného žiarenia
#
# Tento notebook trénuje konvolučné neurónové siete na predikciu slnečného žiarenia z RGB obrázkov pomocou Intel Arc GPU akcelerácie.

# %%
import sys
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import torch

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))

from models.cnn import create_cnn_model
from models.dataset import create_data_loaders
from models.utils import get_device, count_parameters
from models.training import (
    train_model,
    evaluate_model,
    create_optimizer_and_criterion,
    load_checkpoint,
)
from utils.evaluation import (
    plot_training_history,
    plot_residuals,
    plot_predictions_comparison,
)

print(f"Verzia PyTorch: {torch.__version__}")

try:
    import intel_extension_for_pytorch as ipex

    print(f"Intel Extension for PyTorch: {ipex.__version__}")
except (ImportError, OSError) as e:
    print(f"Intel Extension for PyTorch nie je dostupné: {e}")

if hasattr(torch, "xpu") and torch.xpu.is_available():
    print(f"Intel XPU dostupné: {torch.xpu.device_count()} zariadení")
    for i in range(torch.xpu.device_count()):
        print(f"  [{i}]: {torch.xpu.get_device_name(i)}")
else:
    print("Používa sa CPU (GPU nebolo detekované)")


device = get_device()


# %% [markdown]
# ## 2. Načítanie spracovaných dát

# %%
data_dir = project_root / "dataset" / "Solar_data"
processed_dir = project_root / "dataset" / "processed"

train_df = pl.read_csv(processed_dir / "train.csv")
val_df = pl.read_csv(processed_dir / "val.csv")
test_df = pl.read_csv(processed_dir / "test.csv")

print(f"Trénovacie vzorky: {len(train_df)}")
print(f"Validačné vzorky: {len(val_df)}")
print(f"Testovacie vzorky: {len(test_df)}")

if "ResizedImagePath" in train_df.columns:
    print("\nPoužívame pred-resizované obrázky (rýchlejšie načítavanie)")
else:
    print(
        "\nResizované obrázky nenájdené - spustite najprv 01_exploratory_data_analysis.ipynb"
    )


# %% [markdown]
# ## 3. Definícia hyperparametrických konfigurácií
#

# %%
configs = [
    {
        "name": "Konfigurácia 1: Základná",
        "conv_channels": [16, 32, 64],
        "fc_hidden": [32, 16],
        "dropout": 0.65,
        "learning_rate": 0.001,
        "weight_decay": 0.03,
        "batch_size": 128,
        "epochs": 8,
    },
    {
        "name": "Konfigurácia 2: Hlbšia sieť",
        "conv_channels": [16, 32, 64, 128],
        "fc_hidden": [64, 32],
        "dropout": 0.65,
        "learning_rate": 0.001,
        "weight_decay": 0.03,
        "batch_size": 128,
        "epochs": 8,
    },
    {
        "name": "Konfigurácia 3: Vyšší dropout",
        "conv_channels": [16, 32, 64],
        "fc_hidden": [32, 16],
        "dropout": 0.7,
        "learning_rate": 0.001,
        "weight_decay": 0.03,
        "batch_size": 128,
        "epochs": 8,
    },
    {
        "name": "Konfigurácia 4: Väčší weight decay",
        "conv_channels": [16, 32, 64],
        "fc_hidden": [32, 16],
        "dropout": 0.65,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "batch_size": 128,
        "epochs": 8,
    },
    {
        "name": "Konfigurácia 5: Širšia sieť",
        "conv_channels": [24, 48, 96],
        "fc_hidden": [48, 24],
        "dropout": 0.65,
        "learning_rate": 0.0005,
        "weight_decay": 0.04,
        "batch_size": 128,
        "epochs": 8,
    },
]

print(f"Celkový počet konfigurácií: {len(configs)}")
print("\nPravidlo: Conv kanály > FC neuróny, Dropout len na FC")

# %% [markdown]
# ## 4. Trénovanie všetkých konfigurácií

# %%
results = []
checkpoint_dir = project_root / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

for idx, config in enumerate(configs, 1):
    print(f"\n{'=' * 80}")
    print(f"Trénovanie {config['name']} ({idx}/{len(configs)})")
    print(f"{'=' * 80}")
    print(f"Parametre: {config}")

    train_loader, val_loader, test_loader = create_data_loaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        data_dir=data_dir,
        batch_size=config["batch_size"],
        num_workers=0,
    )

    model = create_cnn_model(
        conv_channels=config["conv_channels"],
        fc_hidden=config["fc_hidden"],
        dropout_rate=config["dropout"],
        device=device,
    )

    print(f"\nPočet parametrov modelu: {count_parameters(model):,}")

    optimizer, criterion = create_optimizer_and_criterion(
        model,
        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
        optimizer_type="adamw",
    )

    config_checkpoint_dir = checkpoint_dir / f"config_{idx}"
    config_checkpoint_dir.mkdir(exist_ok=True)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config["epochs"],
        device=device,
        checkpoint_dir=config_checkpoint_dir,
        early_stopping_patience=3,
        verbose=True,
    )

    best_model_path = config_checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        try:
            load_checkpoint(model, optimizer, best_model_path, device)
            print(f"\nNačítaný najlepší model z {best_model_path}")
        except Exception as e:
            print(f"\nChyba pri načítaní checkpointu: {e}")
    else:
        print(f"\nCheckpoint neexistuje: {best_model_path}")

    print("\nVyhodnocovanie na trénovacej množine...")
    train_metrics = evaluate_model(model, train_loader, criterion, device)

    print("Vyhodnocovanie na testovacej množine...")
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    result = {
        "config_name": config["name"],
        "config_idx": idx,
        "config": config,
        "history": history,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "num_parameters": count_parameters(model),
    }
    results.append(result)

    print(f"\nFinálne výsledky pre {config['name']}:")
    print(
        f"  Trénovacia - MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, "
        f"RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}"
    )
    print(
        f"  Testovacia  - MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}, "
        f"RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}"
    )

print(f"\n{'=' * 80}")
print("Všetky konfigurácie úspešne natrénované!")
print(f"{'=' * 80}")


# %% [markdown]
# ## 5. Porovnávacia tabuľka výsledkov

# %%
table_data = []
for result in results:
    table_data.append(
        {
            "Konfigurácia": result["config_name"],
            "Parametre": f"{result['num_parameters']:,}",
            "Train MSE": f"{result['train_metrics']['mse']:.4f}",
            "Train MAE": f"{result['train_metrics']['mae']:.4f}",
            "Train RMSE": f"{result['train_metrics']['rmse']:.4f}",
            "Train R²": f"{result['train_metrics']['r2']:.4f}",
            "Test MSE": f"{result['test_metrics']['mse']:.4f}",
            "Test MAE": f"{result['test_metrics']['mae']:.4f}",
            "Test RMSE": f"{result['test_metrics']['rmse']:.4f}",
            "Test R²": f"{result['test_metrics']['r2']:.4f}",
        }
    )

results_df = pl.DataFrame(table_data)

print("\nPorovnanie výsledkov:")

print(results_df)

results_df.write_csv(project_root / "results_table.csv")
print(f"\nVýsledky uložené do {project_root / 'results_table.csv'}")


# %% [markdown]
# ## 6. Výber najlepšej konfigurácie

# %%
best_idx = np.argmax([r["test_metrics"]["r2"] for r in results])
best_result = results[best_idx]

print(f"Najlepšia konfigurácia: {best_result['config_name']}")
print(f"Test R²: {best_result['test_metrics']['r2']:.4f}")
print(f"Test RMSE: {best_result['test_metrics']['rmse']:.4f}")
print(f"Test MAE: {best_result['test_metrics']['mae']:.4f}")
print(f"Test MSE: {best_result['test_metrics']['mse']:.4f}")

# %%
fig_dir = project_root / "figures"
fig_dir.mkdir(exist_ok=True)

fig = plot_training_history(
    best_result["history"],
    save_path=fig_dir / f"training_history_config_{best_result['config_idx']}.png",
)
plt.show()
print(f"História trénovania uložená do {fig_dir}")

# %% [markdown]
# ## 8. Analýza reziduí najlepšieho modelu

# %%
fig = plot_residuals(
    targets=best_result["test_metrics"]["targets"],
    predictions=best_result["test_metrics"]["predictions"],
    split_name="Test",
    save_path=fig_dir / f"residuals_config_{best_result['config_idx']}.png",
)
plt.show()
print(f"Grafy reziduí uložené do {fig_dir}")

# %% [markdown]
# ## 9. Porovnanie predikcií najlepšieho modelu

# %%
fig = plot_predictions_comparison(
    targets=best_result["test_metrics"]["targets"],
    predictions=best_result["test_metrics"]["predictions"],
    split_name="Test",
    n_samples=100,
    save_path=fig_dir
    / f"predictions_comparison_config_{best_result['config_idx']}.png",
)
plt.show()
print(f"Porovnanie predikcií uložené do {fig_dir}")

# %% [markdown]
# ## 10. História trénovania všetkých konfigurácií

# %%
for result in results:
    print(f"\nVykresľovanie histórie trénovania pre {result['config_name']}")
    fig = plot_training_history(
        result["history"],
        save_path=fig_dir / f"training_history_config_{result['config_idx']}.png",
    )
    plt.show()
    plt.close()

print(f"\nVšetky histórie trénovania uložené do {fig_dir}")

# %% [markdown]
# ## 11. Súhrnné štatistiky

# %%
print("\n" + "=" * 80)
print("FINÁLNE ZHRNUTIE")
print("=" * 80)

print(f"\nNajlepšia konfigurácia: {best_result['config_name']}")
print(f"Počet parametrov: {best_result['num_parameters']:,}")
print("\nDetail konfigurácie:")
for key, value in best_result["config"].items():
    print(f"  {key}: {value}")

print("\nVýkon najlepšieho modelu:")
print("  Trénovacia množina:")
print(f"    MSE:  {best_result['train_metrics']['mse']:.4f}")
print(f"    MAE:  {best_result['train_metrics']['mae']:.4f}")
print(f"    RMSE: {best_result['train_metrics']['rmse']:.4f}")
print(f"    R²:   {best_result['train_metrics']['r2']:.4f}")
print("\n  Testovacia množina:")
print(f"    MSE:  {best_result['test_metrics']['mse']:.4f}")
print(f"    MAE:  {best_result['test_metrics']['mae']:.4f}")
print(f"    RMSE: {best_result['test_metrics']['rmse']:.4f}")
print(f"    R²:   {best_result['test_metrics']['r2']:.4f}")

print("\nVšetky výsledky a obrázky uložené do:")
print(f"  - Tabuľka výsledkov: {project_root / 'results_table.csv'}")
print(f"  - Obrázky: {fig_dir}")
print(f"  - Kontrolné body: {checkpoint_dir}")
print("=" * 80)


# %% [markdown]
# ## 12. Bonus: Vizualizácia konvolučných filtrov

# %%
from utils.bonus import visualize_conv_filters

best_model = create_cnn_model(
    conv_channels=best_result["config"]["conv_channels"],
    fc_hidden=best_result["config"]["fc_hidden"],
    dropout_rate=best_result["config"]["dropout"],
    device=device,
)

best_model_path = (
    checkpoint_dir / f"config_{best_result['config_idx']}" / "best_model.pth"
)
optimizer_dummy, _ = create_optimizer_and_criterion(best_model, learning_rate=0.001)
load_checkpoint(best_model, optimizer_dummy, best_model_path, device)
best_model.eval()

print(f"Načítaný najlepší model: {best_result['config_name']}")
print(f"Počet konvolučných blokov: {len(best_model.conv_layers)}")

print("\n" + "=" * 60)
print("Vizualizácia filtrov prvej konvolučnej vrstvy")
print("=" * 60)
save_path = fig_dir / f"conv_filters_layer_0_config_{best_result['config_idx']}.png"
visualize_conv_filters(best_model, layer_idx=0, max_filters=32, save_path=save_path)
plt.show()

if len(best_model.conv_layers) > 1:
    print("\n" + "=" * 60)
    print("Vizualizácia filtrov druhej konvolučnej vrstvy")
    print("=" * 60)
    save_path = fig_dir / f"conv_filters_layer_1_config_{best_result['config_idx']}.png"
    visualize_conv_filters(best_model, layer_idx=1, max_filters=32, save_path=save_path)
    plt.show()


# %% [markdown]
# ## 13. Bonus: Testovanie na vlastných obrázkoch oblohy

# %%
from utils.bonus import predict_custom_images

my_images_dir = project_root / "dataset" / "Solar_data" / "my_images"
my_images_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TESTOVANIE NA VLASTNÝCH OBRÁZKOCH OBLOHY")
print("=" * 80)
print(f"\nPriečinok s obrázkami: {my_images_dir}")

custom_results = predict_custom_images(best_model, my_images_dir, device)

if len(custom_results) >= 5:
    print(f"\nÚspešne spracovaných {len(custom_results)} vlastných obrázkov")
elif len(custom_results) > 0:
    print(f"\nNájdených len {len(custom_results)} obrázkov (minimum je 5)")
else:
    print(
        "\nŽiadne obrázky neboli nájdené. Prosím pridajte vlastné fotografie oblohy do:"
    )
    print(f"  {my_images_dir}")


# %% [markdown]
# ### 14.1 Vizualizácia predikcií vlastných obrázkov

# %%
from utils.bonus import visualize_custom_predictions

if len(custom_results) > 0:
    save_path = (
        fig_dir / f"custom_images_predictions_config_{best_result['config_idx']}.png"
    )
    visualize_custom_predictions(custom_results, save_path=save_path)
    plt.show()
else:
    print(
        "Žiadne vlastné obrázky na vizualizáciu. Pridajte obrázky do priečinka my_images a spustite túto bunku znovu."
    )


# %% [markdown]
# # BONUS: Predikcia žiarenia o 15 minút

# %%

import pandas as pd


def prepare_prediction_data_from_processed(df_polars, data_dir):
    """
    Pripraví dáta pre predikciu o 15 minút z Polars DataFrame.
    Páruje aktuálny obrázok s hodnotou žiarenia o 15 minút neskôr.
    """
    df = df_polars.to_pandas(use_pyarrow_extension_array=False)

    df["DateTime_parsed"] = pd.to_datetime(
        df["DateTime_clean"], format="%d/%m/%Y %H:%M:%S.%f", errors="coerce"
    )

    df = df.sort_values(["Month", "DateTime_parsed"]).reset_index(drop=True)

    prediction_data = []

    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if current_row["Month"] != next_row["Month"]:
            continue

        if pd.isna(current_row["DateTime_parsed"]) or pd.isna(
            next_row["DateTime_parsed"]
        ):
            continue

        time_diff = (
            next_row["DateTime_parsed"] - current_row["DateTime_parsed"]
        ).total_seconds() / 60

        if 10 <= time_diff <= 20:
            prediction_data.append(
                {
                    "PictureName": current_row["PictureName"],
                    "Month": current_row["Month"],
                    "CurrentIrradiance": current_row["Irradiance"],
                    "FutureIrradiance": next_row["Irradiance"],
                    "TimeDiff": time_diff,
                }
            )

    prediction_df = pd.DataFrame(prediction_data)
    return prediction_df


print("Príprava dát pre predikciu o 15 minút...")
pred_df = prepare_prediction_data_from_processed(train_df, data_dir)
print(f"Vytvorených párov z trénovacích dát: {len(pred_df)}")

pred_val = prepare_prediction_data_from_processed(val_df, data_dir)
pred_test = prepare_prediction_data_from_processed(test_df, data_dir)
print(f"Vytvorených párov z validačných dát: {len(pred_val)}")
print(f"Vytvorených párov z testovacích dát: {len(pred_test)}")

if len(pred_df) > 0:
    print(f"\nPriemerný časový rozdiel: {pred_df['TimeDiff'].mean():.1f} min")

# %%
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Transformácie - rovnaké ako pri trénovaní
pred_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PredictionDataset(Dataset):
    """Dataset pre predikciu žiarenia o 15 minút."""

    def __init__(self, prediction_df, data_dir, transform=None):
        self.prediction_df = prediction_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.prediction_df)

    def __getitem__(self, idx):
        row = self.prediction_df.iloc[idx]

        month_str = str(int(row["Month"])).zfill(2)
        image_path = self.data_dir / month_str / "original" / row["PictureName"]

        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Chyba pri načítaní {image_path}: {e}")
            image = torch.zeros(3, 128, 128)

        target = torch.tensor(row["FutureIrradiance"], dtype=torch.float32)

        return image, target


if len(pred_df) > 0:
    pred_train_dataset = PredictionDataset(pred_df, data_dir, transform=pred_transform)
    pred_val_dataset = PredictionDataset(pred_val, data_dir, transform=pred_transform)
    pred_test_dataset = PredictionDataset(pred_test, data_dir, transform=pred_transform)

    batch_size = best_result["config"]["batch_size"]
    pred_train_loader = DataLoader(
        pred_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    pred_val_loader = DataLoader(
        pred_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    pred_test_loader = DataLoader(
        pred_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Trénovacia množina: {len(pred_train_dataset)}")
    print(f"Validačná množina: {len(pred_val_dataset)}")
    print(f"Testovacia množina: {len(pred_test_dataset)}")
else:
    print("Nedostatok dát pre predikciu o 15 minút!")

# %%


if len(pred_df) > 0:
    print("Trénovanie CNN pre predikciu žiarenia o 15 minút...")
    print("=" * 60)
    best_config = best_result["config"]
    print(f"Konfigurácia: {best_config['name']}")

    pred_model = create_cnn_model(
        conv_channels=best_config["conv_channels"],
        fc_hidden=best_config["fc_hidden"],
        dropout_rate=best_config["dropout"],
        device=device,
    )

    pred_optimizer, pred_criterion = create_optimizer_and_criterion(
        pred_model,
        learning_rate=best_config["learning_rate"],
        weight_decay=best_config["weight_decay"],
        optimizer_type="adamw",
    )

    pred_history = train_model(
        model=pred_model,
        train_loader=pred_train_loader,
        val_loader=pred_val_loader,
        criterion=pred_criterion,
        optimizer=pred_optimizer,
        epochs=best_config["epochs"],
        device=device,
        checkpoint_dir=checkpoint_dir / "prediction_15min",
        early_stopping_patience=3,
        verbose=True,
    )

    print("\nTrénovanie dokončené!")
else:
    print("Preskakujem trénovanie - nedostatok dát")

# %%
if len(pred_df) > 0:
    print("Evaluácia modelu pre predikciu o 15 minút")
    print("=" * 60)

    pred_metrics = evaluate_model(pred_model, pred_test_loader, pred_criterion, device)

    print("\nVýsledky predikcie o 15 minút:")
    print(f"  MSE:  {pred_metrics['mse']:.4f}")
    print(f"  MAE:  {pred_metrics['mae']:.4f}")
    print(f"  RMSE: {pred_metrics['rmse']:.4f}")
    print(f"  R²:   {pred_metrics['r2']:.4f}")

    print("\nPorovnanie:")
    print(
        f"  Odhad (aktuálne žiarenie):    R²={best_result['test_metrics']['r2']:.4f}, RMSE={best_result['test_metrics']['rmse']:.4f}"
    )
    print(
        f"  Predikcia (o 15 min):         R²={pred_metrics['r2']:.4f}, RMSE={pred_metrics['rmse']:.4f}"
    )
else:
    print("Preskakujem evaluáciu - model nebol natrénovaný")

# %%
if len(pred_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    all_preds = pred_metrics["predictions"]
    all_targets = pred_metrics["targets"]

    # 1. Predikcie vs skutočné hodnoty
    ax1 = axes[0]
    ax1.scatter(all_targets, all_preds, alpha=0.3, s=10)
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    ax1.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideálna predikcia"
    )
    ax1.set_xlabel("Skutočné žiarenie o 15 min [W/m²]")
    ax1.set_ylabel("Predikované žiarenie [W/m²]")
    ax1.set_title(f"Predikcia o 15 minút\nR²={pred_metrics['r2']:.4f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Distribúcia chýb
    ax2 = axes[1]
    errors = all_preds - all_targets
    ax2.hist(errors, bins=50, edgecolor="black", alpha=0.7)
    ax2.axvline(0, color="r", linestyle="--", label="Nulová chyba")
    ax2.set_xlabel("Chyba predikcie [W/m²]")
    ax2.set_ylabel("Počet vzoriek")
    ax2.set_title(f"Distribúcia chýb\nMAE={pred_metrics['mae']:.2f} W/m²")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Residuals vs skutočné
    ax3 = axes[2]
    ax3.scatter(all_targets, errors, alpha=0.3, s=10)
    ax3.axhline(0, color="r", linestyle="--")
    ax3.set_xlabel("Skutočné žiarenie o 15 min [W/m²]")
    ax3.set_ylabel("Chyba predikcie [W/m²]")
    ax3.set_title("Reziduálny graf")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "bonus_prediction_15min.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nGraf uložený: {fig_dir / 'bonus_prediction_15min.png'}")
else:
    print("Preskakujem vizualizáciu - model nebol natrénovaný")
# %% [markdown]
# ## 1. Import knižníc

# %%

import sys
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))

from models.feature_extraction import FeatureExtractor
from models.utils import get_device
from utils.clustering import (
    reduce_dimensionality,
    cluster_features,
    visualize_clusters_2d,
    display_cluster_samples,
    compute_average_images,
    analyze_clusters,
)
from utils.evaluation import (
    plot_training_history,
    plot_residuals,
    plot_predictions_comparison,
)

print(f"Verzia PyTorch: {torch.__version__}")

try:
    import intel_extension_for_pytorch as ipex

    print(f"Intel Extension for PyTorch: {ipex.__version__}")
except (ImportError, OSError) as e:
    print(f"Intel Extension for PyTorch nie je dostupné: {e}")

if hasattr(torch, "xpu") and torch.xpu.is_available():
    print(f"Intel XPU dostupné: {torch.xpu.device_count()} zariadení")
    for i in range(torch.xpu.device_count()):
        print(f"  [{i}]: {torch.xpu.get_device_name(i)}")
else:
    print("Používa sa CPU (GPU nebolo detekované)")

device = get_device()

# %% [markdown]
# ## 2. Načítanie spracovaných dát

# %%
data_dir = project_root / "dataset" / "Solar_data"
processed_dir = project_root / "dataset" / "processed"

train_df = pl.read_csv(processed_dir / "train.csv")
val_df = pl.read_csv(processed_dir / "val.csv")
test_df = pl.read_csv(processed_dir / "test.csv")

print(f"Trénovacie vzorky: {len(train_df)}")
print(f"Validačné vzorky: {len(val_df)}")
print(f"Testovacie vzorky: {len(test_df)}")

# Zobrazenie štatistík cieľovej premennej
print(f"\nŠtatistiky Irradiance:")
print(
    f"  Trénovacie - priemer: {train_df['Irradiance'].mean():.2f}, std: {train_df['Irradiance'].std():.2f}"
)
print(
    f"  Validačné  - priemer: {val_df['Irradiance'].mean():.2f}, std: {val_df['Irradiance'].std():.2f}"
)
print(
    f"  Testovacie - priemer: {test_df['Irradiance'].mean():.2f}, std: {test_df['Irradiance'].std():.2f}"
)

# %% [markdown]
# ## 3. Výber predtrénovaného modelu pre extrakciu príznakov
#
# Vyberieme MobileNetV2 (menší počet parametrov = rýchlejšie výpočty)

# %%
model_name = "mobilenet_v2"

print(f"Vybraný model: {model_name}")
print("\nInicializácia Feature Extractor...")

feature_extractor = FeatureExtractor(model_name=model_name, device=device)

print(f"Model úspešne načítaný na zariadenie: {device}")
print(f"Veľkosť výstupného feature vektora: {feature_extractor.feature_dim}")

# %% [markdown]
# ## 4. Extrakcia príznakov z obrázkov
#
# Použijeme predtrénovaný model na extrakciu príznakov zo všetkých obrázkov.

# %%
print("=" * 80)
print("EXTRAKCIA PRÍZNAKOV Z TRÉNOVACEJ MNOŽINY")
print("=" * 80)
train_features_df = feature_extractor.extract_features_from_dataframe(
    train_df, data_dir, batch_size=64, use_resized=True
)

print("\n" + "=" * 80)
print("EXTRAKCIA PRÍZNAKOV Z VALIDAČNEJ MNOŽINY")
print("=" * 80)
val_features_df = feature_extractor.extract_features_from_dataframe(
    val_df, data_dir, batch_size=64, use_resized=True
)

print("\n" + "=" * 80)
print("EXTRAKCIA PRÍZNAKOV Z TESTOVACEJ MNOŽINY")
print("=" * 80)
test_features_df = feature_extractor.extract_features_from_dataframe(
    test_df, data_dir, batch_size=64, use_resized=True
)

print("\n" + "=" * 80)
print("EXTRAKCIA PRÍZNAKOV DOKONČENÁ")
print("=" * 80)
print(f"Trénovacie príznaky: {train_features_df.shape}")
print(f"Validačné príznaky: {val_features_df.shape}")
print(f"Testovacie príznaky: {test_features_df.shape}")

# %% [markdown]
# ## 5. Uloženie extrahovaných príznakov do DataFrame

# %%
features_dir = processed_dir / "features"
features_dir.mkdir(exist_ok=True)

train_features_df.write_csv(features_dir / f"train_features_{model_name}.csv")
val_features_df.write_csv(features_dir / f"val_features_{model_name}.csv")
test_features_df.write_csv(features_dir / f"test_features_{model_name}.csv")

print(f"Príznaky uložené do: {features_dir}")
print(f"  - train_features_{model_name}.csv")
print(f"  - val_features_{model_name}.csv")
print(f"  - test_features_{model_name}.csv")

print("\nPríklad štruktúry DataFrame:")
print(train_features_df.head(3))

# %% [markdown]
# ## 6. Redukcia dimenzie príznakov (PCA)
#
# Pre vizualizáciu a zhlukovanie znížime dimenziu príznakov pomocou PCA.

# %%
feature_cols = [col for col in train_features_df.columns if col.startswith("feature_")]

X_train = train_features_df.select(feature_cols).to_numpy()
X_val = val_features_df.select(feature_cols).to_numpy()
X_test = test_features_df.select(feature_cols).to_numpy()

print(f"Pôvodná dimenzia príznakov: {X_train.shape[1]}")

n_components = 50
X_train_reduced, pca_model = reduce_dimensionality(
    X_train, method="pca", n_components=n_components
)
X_val_reduced = pca_model.transform(X_val)
X_test_reduced = pca_model.transform(X_test)

print(f"Redukovaná dimenzia: {X_train_reduced.shape[1]}")
print(
    f"Vysvetlený rozptyl (prvých 10 komponentov): {pca_model.explained_variance_ratio_[:10]}"
)
print(f"Celkový vysvetlený rozptyl: {pca_model.explained_variance_ratio_.sum():.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(
    range(1, len(pca_model.explained_variance_ratio_) + 1),
    pca_model.explained_variance_ratio_,
    "b-",
)
plt.xlabel("Počet komponentov")
plt.ylabel("Vysvetlený rozptyl")
plt.title("Vysvetlený rozptyl jednotlivými komponentmi")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(
    range(1, len(pca_model.explained_variance_ratio_) + 1),
    np.cumsum(pca_model.explained_variance_ratio_),
    "r-",
)
plt.xlabel("Počet komponentov")
plt.ylabel("Kumulatívny vysvetlený rozptyl")
plt.title("Kumulatívny vysvetlený rozptyl")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Zhlukovanie príznakov pomocou K-Means
#
# Použijeme K-Means na vytvorenie zhlukov z extrahovaných príznakov.

# %%
from sklearn.metrics import silhouette_score

inertias = []
silhouette_scores = []
K_range = range(2, 11)

print("Hľadanie optimálneho počtu zhlukov...")
for k in K_range:
    cluster_labels, kmeans_model = cluster_features(
        X_train_reduced, method="kmeans", n_clusters=k
    )
    inertias.append(kmeans_model.inertia_)
    sil_score = silhouette_score(X_train_reduced, cluster_labels)
    silhouette_scores.append(sil_score)
    print(f"k={k}: Inertia={kmeans_model.inertia_:.2f}, Silhouette={sil_score:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(K_range, inertias, "bo-")
axes[0].set_xlabel("Počet zhlukov (k)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, "ro-")
axes[1].set_xlabel("Počet zhlukov (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Analysis")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

optimal_k = 7
print(f"\n{'=' * 80}")
print(f"Vybraný počet zhlukov: {optimal_k}")
print(f"{'=' * 80}")

# %% [markdown]
# ## 8. Finálne zhlukovanie a vizualizácia

# %%
cluster_labels, kmeans_final = cluster_features(
    X_train_reduced, method="kmeans", n_clusters=optimal_k
)

train_features_df = train_features_df.with_columns(pl.Series("Cluster", cluster_labels))

print("Rozdelenie vzoriek do zhlukov:")
cluster_counts = (
    train_features_df.group_by("Cluster").agg(pl.count().alias("count")).sort("Cluster")
)
print(cluster_counts)

print("\nRedukcia na 2D pomocou t-SNE pre vizualizáciu...")
X_train_2d, tsne_model = reduce_dimensionality(
    X_train_reduced, method="tsne", n_components=2
)

fig_dir = project_root / "figures"
fig_dir.mkdir(exist_ok=True)

targets = train_features_df["Irradiance"].to_numpy()

fig = visualize_clusters_2d(
    X_train_2d, cluster_labels, targets=targets, save_path=fig_dir / "clusters_2d.png"
)
plt.show()

# %% [markdown]
# ## 9. Zobrazenie ukážkových obrázkov pre každý zhluk

# %%
print("=" * 80)
print("UKÁŽKOVÉ OBRÁZKY PRE KAŽDÝ ZHLUK")
print("=" * 80)

for cluster_id in range(optimal_k):
    print(f"\n{'=' * 60}")
    print(f"Zhluk {cluster_id}")
    print(f"{'=' * 60}")

    fig = display_cluster_samples(
        train_features_df,
        data_dir=data_dir,
        cluster_id=cluster_id,
        n_samples=8,
        save_path=fig_dir / f"cluster_{cluster_id}_samples.png",
    )
    plt.show()

# %% [markdown]
# ## 10. Zobrazenie priemerných obrázkov pre každý zhluk

# %%
print("Výpočet priemerných obrázkov pre každý zhluk...")
average_images_dict = compute_average_images(train_features_df, data_dir)

n_clusters = len(average_images_dict)
cols = min(4, n_clusters)
rows = (n_clusters + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
if n_clusters == 1:
    axes = np.array([axes])
axes = axes.flatten()

for idx, (cluster_id, avg_img) in enumerate(sorted(average_images_dict.items())):
    axes[idx].imshow(avg_img)
    axes[idx].set_title(f"Zhluk {cluster_id} (priemer)")
    axes[idx].axis("off")

for idx in range(n_clusters, len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
save_path = fig_dir / "average_cluster_images.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Priemerné obrázky uložené do {save_path}")
plt.show()

# %% [markdown]
# ## 11. Analýza výsledkov zhlukovania

# %%
cluster_stats = analyze_clusters(train_features_df)

print("=" * 80)
print("ANALÝZA ZHLUKOV")
print("=" * 80)
print(cluster_stats)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

cluster_data = []
for cluster_id in range(optimal_k):
    cluster_df = train_features_df.filter(pl.col("Cluster") == cluster_id)
    cluster_data.append(cluster_df["Irradiance"].to_numpy())

axes[0].boxplot(cluster_data, labels=[f"Zhluk {i}" for i in range(optimal_k)])
axes[0].set_xlabel("Zhluk")
axes[0].set_ylabel("Irradiance")
axes[0].set_title("Distribúcia Irradiance podľa zhlukov (Boxplot)")
axes[0].grid(True, alpha=0.3)

for cluster_id in range(optimal_k):
    cluster_df = train_features_df.filter(pl.col("Cluster") == cluster_id)
    axes[1].hist(
        cluster_df["Irradiance"].to_numpy(),
        alpha=0.5,
        label=f"Zhluk {cluster_id}",
        bins=30,
    )

axes[1].set_xlabel("Irradiance")
axes[1].set_ylabel("Frekvencia")
axes[1].set_title("Distribúcia Irradiance podľa zhlukov (Histogram)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    fig_dir / "cluster_irradiance_distribution.png", dpi=150, bbox_inches="tight"
)
plt.show()

print("\n" + "=" * 80)
print("INTERPRETÁCIA ZHLUKOV")
print("=" * 80)
for row in cluster_stats.iter_rows(named=True):
    cluster_id = row["Cluster"]
    count = row["count"]
    mean_irr = row["mean_irradiance"]
    std_irr = row["std_irradiance"]

    print(f"\nZhluk {cluster_id}: {count} vzoriek")
    print(f"  Priemerná Irradiance: {mean_irr:.2f} ± {std_irr:.2f}")

    if mean_irr < 100:
        print(
            "  Interpretácia: Veľmi nízke žiarenie - pravdepodobne nočné alebo husté mraky"
        )
    elif mean_irr < 300:
        print("  Interpretácia: Nízke žiarenie - zamračené, podvečer/skoré ráno")
    elif mean_irr < 600:
        print("  Interpretácia: Stredné žiarenie - polojasno, čiastočne oblačno")
    elif mean_irr < 900:
        print("  Interpretácia: Vysoké žiarenie - jasno, málo oblakov")
    else:
        print(
            "  Interpretácia: Veľmi vysoké žiarenie - jasná obloha, priame slnečné svetlo"
        )

# %% [markdown]
# ## 12. Trénovanie regresora na extrahovaných príznakoch
#
# Použijeme Gradient Boosting Regressor na extrahovaných príznakoch.

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge

y_train = train_features_df["Irradiance"].to_numpy()
y_val = val_features_df["Irradiance"].to_numpy()
y_test = test_features_df["Irradiance"].to_numpy()

print("=" * 80)
print("TRANSFER LEARNING")
print("=" * 80)

# 1. Viac PCA komponentov (300 namiesto 50)
n_components_improved = 300
print(f"\n1. PCA: {n_components} → {n_components_improved} komponentov")

pca_improved = PCA(n_components=n_components_improved, random_state=42)
X_train_improved = pca_improved.fit_transform(X_train)
X_val_improved = pca_improved.transform(X_val)
X_test_improved = pca_improved.transform(X_test)

variance_improved = pca_improved.explained_variance_ratio_.sum()
print(
    f"   Vysvetlený rozptyl: {variance_improved:.4f} ({variance_improved * 100:.2f}%)"
)

# 2. Kombinácia s tabuľkovými príznakmi
TABULAR_FEATURES = [
    "Temperature",
    "Pressure",
    "Humidity",
    "WindDirection",
    "Speed",
    "SunDurationSensor",
    "Sunshine_L",
    "SunElevation",
    "SunAzimuth",
]

tabular_available = [f for f in TABULAR_FEATURES if f in train_df.columns]
print(f"\n2. Tabuľkové príznaky: {len(tabular_available)}")

if tabular_available:
    tabular_train = train_df.select(tabular_available).to_pandas().values
    tabular_val = val_df.select(tabular_available).to_pandas().values
    tabular_test = test_df.select(tabular_available).to_pandas().values

    tab_scaler = StandardScaler()
    tabular_train_scaled = tab_scaler.fit_transform(tabular_train)
    tabular_val_scaled = tab_scaler.transform(tabular_val)
    tabular_test_scaled = tab_scaler.transform(tabular_test)

    X_train_combined = np.hstack([X_train_improved, tabular_train_scaled])
    X_val_combined = np.hstack([X_val_improved, tabular_val_scaled])
    X_test_combined = np.hstack([X_test_improved, tabular_test_scaled])
else:
    X_train_combined = X_train_improved
    X_val_combined = X_val_improved
    X_test_combined = X_test_improved

print(f"   Celkový počet príznakov: {X_train_combined.shape[1]}")

# 3. Stacking Ensemble (GBR + RF + Ridge)
print("\n3. Stacking Ensemble Regressor")

base_estimators = [
    (
        "gbr",
        GradientBoostingRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
        ),
    ),
    (
        "rf",
        RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        ),
    ),
]

stacking_model = StackingRegressor(
    estimators=base_estimators, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1
)

print("   Trénovanie...")
stacking_model.fit(X_train_combined, y_train)

y_train_pred = stacking_model.predict(X_train_combined)
y_val_pred = stacking_model.predict(X_val_combined)
y_test_pred = stacking_model.predict(X_test_combined)

train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n" + "=" * 80)
print("VÝSLEDKY VYLEPŠENÉHO TRANSFER LEARNING")
print("=" * 80)
print(f"\nTrain R²:  {train_r2:.4f}")
print(f"Val R²:    {val_r2:.4f}")
print(f"Test R²:   {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE:  {test_mae:.4f}")

# %% [markdown]
# ## 13. Porovnanie s výsledkami CNN z predošlého notebooku

# %%
results_table_path = project_root / "results_table.csv"

if results_table_path.exists():
    cnn_results = pl.read_csv(results_table_path)

    r2_col = None
    rmse_col = None
    for col in cnn_results.columns:
        if "R²" in col or "R2" in col or "r2" in col.lower():
            r2_col = col
        if "RMSE" in col or "rmse" in col.lower():
            rmse_col = col

    if r2_col and rmse_col:
        best_cnn_r2 = float(cnn_results.select(pl.col(r2_col).max())[0, 0])
        best_cnn_rmse = float(
            cnn_results.filter(
                pl.col(r2_col) == cnn_results.select(pl.col(r2_col).max())[0, 0]
            ).select(pl.col(rmse_col))[0, 0]
        )

        print("=" * 80)
        print("POROVNANIE S CNN")
        print("=" * 80)

        print("\nNajlepší CNN model:")
        print(f"  Test R²:   {best_cnn_r2:.4f}")
        print(f"  Test RMSE: {best_cnn_rmse:.4f}")

        print("\nTransfer Learning (XGBoost + 300 PCA + Tabuľky):")
        print(f"  Test R²:   {test_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")

        r2_diff = test_r2 - best_cnn_r2
        rmse_diff = best_cnn_rmse - test_rmse

        print("\nZmena:")
        print(f"  ΔR²:   {r2_diff:+.4f}")
        print(f"  ΔRMSE: {rmse_diff:+.4f}")

        if test_r2 > best_cnn_r2:
            print("\nTransfer Learning PREKONÁVA CNN!")
        else:
            print("\nTransfer Learning ešte neprekonáva CNN")
else:
    print("results_table.csv nebol nájdený")

# %% [markdown]
# ## 14. Vizualizácia reziduálov

# %%
fig = plot_residuals(
    targets=y_test,
    predictions=y_test_pred,
    split_name="Test (Transfer Learning)",
    save_path=fig_dir / "transfer_learning_residuals.png",
)
plt.show()

# %% [markdown]
# ## 15. Porovnanie predikcií vs. skutočné hodnoty

# %%
fig = plot_predictions_comparison(
    targets=y_test,
    predictions=y_test_pred,
    split_name="Test (Transfer Learning)",
    n_samples=100,
    save_path=fig_dir / "transfer_learning_predictions.png",
)
plt.show()

# %% [markdown]
# ## 16. Analýza chýb: Kde sa model mýli?

# %%
errors = np.abs(y_test - y_test_pred)

worst_indices = np.argsort(errors)[-10:][::-1]

print("=" * 80)
print("TOP 10 NAJHORŠÍCH PREDIKCIÍ")
print("=" * 80)

for rank, idx in enumerate(worst_indices, 1):
    true_val = y_test[idx]
    pred_val = y_test_pred[idx]
    error = errors[idx]
    print(
        f"{rank}. Index: {idx}, Skutočná: {true_val:.2f}, Predikcia: {pred_val:.2f}, Chyba: {error:.2f}"
    )

from PIL import Image  # noqa: E402

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

has_picture_name = "PictureName" in test_df.columns
has_month = "Month" in test_df.columns

print(f"\nPictureName dostupné v test_df: {has_picture_name}")
print(f"Month dostupné v test_df: {has_month}")

for i, idx in enumerate(worst_indices):
    img_loaded = False
    try:
        if has_picture_name and has_month and idx < len(test_df):
            row = test_df.row(int(idx), named=True)
            picture_name = row["PictureName"]
            month_raw = row["Month"]

            if isinstance(month_raw, int):
                month = str(month_raw).zfill(2)
            else:
                month = str(month_raw).zfill(2)

            possible_paths = [
                data_dir / month / "original" / picture_name,
                data_dir / month / "ResizedImages_128" / picture_name,
            ]

            for full_path in possible_paths:
                if full_path.exists():
                    img = Image.open(full_path)
                    axes[i].imshow(img)
                    img_loaded = True
                    break

            if not img_loaded:
                print(
                    f"  Obrázok nenájdený pre index {idx}: {picture_name} (mesiac {month})"
                )
                axes[i].text(
                    0.5,
                    0.5,
                    f"Obrázok\nnenájdený\n{picture_name[:15]}...",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                    fontsize=8,
                )
    except Exception as e:
        print(f"Chyba pri načítaní obrázka {idx}: {e}")
        axes[i].text(
            0.5,
            0.5,
            f"Chyba:\n{str(e)[:30]}",
            ha="center",
            va="center",
            transform=axes[i].transAxes,
            fontsize=8,
        )

    true_val = y_test[idx]
    pred_val = y_test_pred[idx]
    error = errors[idx]

    axes[i].set_title(
        f"Skutočná: {true_val:.1f}\nPredikcia: {pred_val:.1f}\nChyba: {error:.1f}",
        fontsize=10,
    )
    axes[i].axis("off")

plt.suptitle("TOP 10 najhorších predikcií modelu", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(fig_dir / "worst_predictions.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 80)
print("ANALÝZA CHÝB PODĽA ROZSAHU IRRADIANCE")
print("=" * 80)

ranges = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1200)]
for low, high in ranges:
    mask = (y_test >= low) & (y_test < high)
    if mask.sum() > 0:
        range_errors = errors[mask]
        print(f"\nRozsah {low}-{high} W/m²:")
        print(f"  Počet vzoriek: {mask.sum()}")
        print(f"  Priemerná chyba: {range_errors.mean():.2f}")
        print(f"  Medián chyby: {np.median(range_errors):.2f}")
        print(f"  Max chyba: {range_errors.max():.2f}")
        print(f"  RMSE: {np.sqrt((range_errors**2).mean()):.2f}")

print("\n" + "=" * 80)
print("ZÁVER ANALÝZY CHÝB")
print("=" * 80)
print("Model má tendenciu robiť najväčšie chyby pri:")
print("1. Extrémnych hodnotách žiarenia (veľmi nízke alebo veľmi vysoké)")
print("2. Prechodových podmienkach (čiastočne oblačno)")
print("3. Neobvyklých poveternostných situáciách")
print("\nMožné zlepšenia:")
print("- Augmentácia dát pre zriedkavé prípady")
print("- Použitie ensemble metód")
print("- Fine-tuning predtrénovaného modelu")
print("- Pridanie časových/meteorologických features")

# %% [markdown]
# ## 17. Finálne zhrnutie

# %%
print("=" * 80)
print("FINÁLNE ZHRNUTIE - TRANSFER LEARNING A ZHLUKOVANIE")
print("=" * 80)

print("\n1. EXTRAKCIA PRÍZNAKOV")
print(f"   Model: {model_name}")
print(f"   Pôvodná dimenzia príznakov: {len(feature_cols)}")
print(f"   Redukovaná dimenzia (PCA): {n_components}")
print(f"   Vysvetlený rozptyl: {pca_model.explained_variance_ratio_.sum():.2%}")

print("\n2. ZHLUKOVANIE")
print("   Algoritmus: K-Means")
print(f"   Počet zhlukov: {optimal_k}")
print("   Vizualizácia: t-SNE (2D)")

print("\n3. TRÉNOVANIE REGRESORA")
print("   Model: Stacking Ensemble (GBR + RF + Ridge)")
print(f"   Trénovacie vzorky: {len(y_train)}")
print(f"   Validačné vzorky: {len(y_val)}")
print(f"   Testovacie vzorky: {len(y_test)}")

print("\n4. VÝSLEDKY NA TESTOVACEJ MNOŽINE")
print(f"   MSE:  {test_mse:.4f}")
print(f"   MAE:  {test_mae:.4f}")
print(f"   RMSE: {test_rmse:.4f}")
print(f"   R²:   {test_r2:.4f}")

if results_table_path.exists():
    print("\n5. POROVNANIE S CNN")
    print(
        f"   CNN (najlepší):           R²={best_cnn_r2:.4f}, RMSE={best_cnn_rmse:.4f}"
    )
    print(f"   Transfer Learning:        R²={test_r2:.4f}, RMSE={test_rmse:.4f}")
    print(f"   Zlepšenie R²:             {test_r2 - best_cnn_r2:+.4f}")
    print(f"   Zlepšenie RMSE:           {best_cnn_rmse - test_rmse:+.4f}")

print("\n6. ULOŽENÉ SÚBORY")
print(f"   - Príznaky: {features_dir}")
print(f"   - Grafy: {fig_dir}")
print("     • clusters_2d.png")
print("     • cluster_<id>_samples.png (pre každý zhluk)")
print("     • average_cluster_images.png")
print("     • cluster_irradiance_distribution.png")
print("     • transfer_learning_residuals.png")
print("     • transfer_learning_predictions.png")
print("     • worst_predictions.png")

print("\n" + "=" * 80)
print("NOTEBOOK ÚSPEŠNE DOKONČENÝ!")
print("=" * 80)

# %% [markdown]
# # BONUS: Fúzia modelov - kombinácia obrazových a tabuľkových dát
#
# Vytvoríme fúzny model, ktorý kombinuje:
# 1. **Obrazové príznaky** - extrahované z ResNet (transfer learning)
# 2. **Tabuľkové dáta** - senzorové údaje (teplota, vlhkosť, tlak, atď.)

# %%
TABULAR_FEATURES = [
    "BodyTemperature",
    "RelativeHumidity",
    "HumidityTemp",
    "Pressure",
    "PressureAvg",
    "PressureTemp",
    "TiltAngle",
    "SunAzimuth",
    "SunZenith",
]

print("Kontrola tabuľkových príznakov:")
tabular_available = []
for feature in TABULAR_FEATURES:
    if feature in train_df.columns:
        tabular_available.append(feature)
        print(f"{feature}")
    else:
        print(f"{feature} - NENÁJDENÉ")

print(
    f"\nDostupných tabuľkových príznakov: {len(tabular_available)}/{len(TABULAR_FEATURES)}"
)

if len(tabular_available) == 0:
    print("\nUpozornenie: Žiadne tabuľkové príznaky nie sú dostupné!")
    print("Bonus sekcia bude preskočená.")

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler

if len(tabular_available) > 0:
    print("Príprava kombinovaných príznakov...")

    tabular_scaler = StandardScaler()

    tabular_train_data = train_df[tabular_available].to_pandas()
    tabular_train_scaled = tabular_scaler.fit_transform(tabular_train_data)

    tabular_val_data = val_df[tabular_available].to_pandas()
    tabular_val_scaled = tabular_scaler.transform(tabular_val_data)

    tabular_test_data = test_df[tabular_available].to_pandas()
    tabular_test_scaled = tabular_scaler.transform(tabular_test_data)

    X_fusion_train = np.hstack([X_train_reduced, tabular_train_scaled])
    X_fusion_val = np.hstack([X_val_reduced, tabular_val_scaled])
    X_fusion_test = np.hstack([X_test_reduced, tabular_test_scaled])

    y_fusion_train = y_train
    y_fusion_val = y_val
    y_fusion_test = y_test

    print("\nFúzný dataset vytvorený:")
    print(f"  Trénovacia množina: {X_fusion_train.shape}")
    print(f"  Validačná množina: {X_fusion_val.shape}")
    print(f"  Testovacia množina: {X_fusion_test.shape}")
    print(f"  Obrazové príznaky (PCA): {X_train_reduced.shape[1]}")
    print(f"  Tabuľkové príznaky: {len(tabular_available)}")
    print(f"  Celkovo: {X_fusion_train.shape[1]} príznakov")
else:
    print("Fúzny model nemožno vytvoriť - žiadne tabuľkové príznaky nie sú dostupné")

# %%
if len(tabular_available) > 0:
    X_img_train = X_train_reduced
    X_img_val = X_val_reduced
    X_img_test = X_test_reduced
    y_img_train = y_train
    y_img_val = y_val
    y_img_test = y_test

    X_tab_train = tabular_train_scaled
    X_tab_val = tabular_val_scaled
    X_tab_test = tabular_test_scaled
    y_tab_train = y_train
    y_tab_val = y_val
    y_tab_test = y_test

    print("Dáta pre porovnanie modelov pripravené:")
    print(f"  Obrazové príznaky:  {X_img_train.shape[1]} príznakov")
    print(f"  Tabuľkové príznaky: {X_tab_train.shape[1]} príznakov")
    print(f"  Fúzia:              {X_fusion_train.shape[1]} príznakov")
else:
    print("Bonus sekcia sa preskakuje - žiadne tabuľkové príznaky")

# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if len(tabular_available) > 0:

    def train_and_evaluate_gbr(X_train, y_train, X_test, y_test, name):
        """Trénuje GBR a vracia metriky."""
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        return {
            "name": name,
            "model": model,
            "y_pred": y_pred,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    print("Trénovanie modelov pre porovnanie...")
    print("=" * 60)

    # 1. Model len s obrazovými príznakmi
    print("\n1. Trénovanie modelu s OBRAZOVÝMI príznakmi...")
    result_image = train_and_evaluate_gbr(
        X_img_train, y_img_train, X_img_test, y_img_test, "Obrazové"
    )
    print(f"   R²: {result_image['r2']:.4f}, RMSE: {result_image['rmse']:.4f}")

    # 2. Model len s tabuľkovými príznakmi
    print("\n2. Trénovanie modelu s TABUĽKOVÝMI príznakmi...")
    result_tabular = train_and_evaluate_gbr(
        X_tab_train, y_tab_train, X_tab_test, y_tab_test, "Tabuľkové"
    )
    print(f"   R²: {result_tabular['r2']:.4f}, RMSE: {result_tabular['rmse']:.4f}")

    # 3. Fúzny model (obrazové + tabuľkové)
    print("\n3. Trénovanie FÚZNEHO modelu (obrazové + tabuľkové)...")
    result_fusion = train_and_evaluate_gbr(
        X_fusion_train, y_fusion_train, X_fusion_test, y_fusion_test, "Fúzia"
    )
    print(f"   R²: {result_fusion['r2']:.4f}, RMSE: {result_fusion['rmse']:.4f}")

    print("\n" + "=" * 60)
    print("Trénovanie dokončené!")
else:
    print("Bonus sekcia sa preskakuje - žiadne tabuľkové príznaky")

# %%
if len(tabular_available) > 0:
    print("=" * 70)
    print("POROVNANIE MODELOV - FÚZIA VS. JEDNOTLIVÉ PRÍZNAKY")
    print("=" * 70)

    comparison_data = []
    for result in [result_image, result_tabular, result_fusion]:
        comparison_data.append(
            {
                "Model": result["name"],
                "MSE": result["mse"],
                "MAE": result["mae"],
                "RMSE": result["rmse"],
                "R²": result["r2"],
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    print("\nZlepšenie fúzie oproti obrazovým príznikom:")
    print(f"  R² zlepšenie:   {result_fusion['r2'] - result_image['r2']:+.4f}")
    print(f"  RMSE zlepšenie: {result_image['rmse'] - result_fusion['rmse']:+.4f}")

    print("\nZlepšenie fúzie oproti tabuľkovým príznikom:")
    print(f"  R² zlepšenie:   {result_fusion['r2'] - result_tabular['r2']:+.4f}")
    print(f"  RMSE zlepšenie: {result_tabular['rmse'] - result_fusion['rmse']:+.4f}")
else:
    print("Bonus sekcia sa preskakuje - žiadne tabuľkové príznaky")

# %%
if len(tabular_available) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    results = [result_image, result_tabular, result_fusion]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    titles = ["Obrazové príznaky", "Tabuľkové príznaky", "Fúzia (obrazové + tabuľkové)"]

    for i, (result, color, title) in enumerate(zip(results, colors, titles)):
        ax = axes[0, i]
        ax.scatter(y_fusion_test, result["y_pred"], alpha=0.3, s=10, c=color)
        min_val = min(y_fusion_test.min(), result["y_pred"].min())
        max_val = max(y_fusion_test.max(), result["y_pred"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
        ax.set_xlabel("Skutočné žiarenie [W/m²]")
        ax.set_ylabel("Predikované žiarenie [W/m²]")
        ax.set_title(f"{title}\nR²={result['r2']:.4f}")
        ax.grid(True, alpha=0.3)

    for i, (result, color, title) in enumerate(zip(results, colors, titles)):
        ax = axes[1, i]
        errors = result["y_pred"] - y_fusion_test
        ax.hist(errors, bins=50, edgecolor="black", alpha=0.7, color=color)
        ax.axvline(0, color="black", linestyle="--", lw=2)
        ax.set_xlabel("Chyba predikcie [W/m²]")
        ax.set_ylabel("Počet vzoriek")
        ax.set_title(f"MAE={result['mae']:.2f}, RMSE={result['rmse']:.2f}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "bonus_fusion_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Graf uložený: {fig_dir / 'bonus_fusion_comparison.png'}")
else:
    print("Bonus sekcia sa preskakuje - žiadne tabuľkové príznaky")

# %%
if len(tabular_available) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    model_names = ["Obrazové", "Tabuľkové", "Fúzia"]
    r2_values = [result_image["r2"], result_tabular["r2"], result_fusion["r2"]]
    rmse_values = [result_image["rmse"], result_tabular["rmse"], result_fusion["rmse"]]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    ax1 = axes[0]
    bars1 = ax1.bar(
        model_names, r2_values, color=colors, edgecolor="black", linewidth=1.5
    )
    ax1.set_ylabel("R² skóre")
    ax1.set_title("Porovnanie R² skóre")
    ax1.set_ylim(0, 1)
    for bar, val in zip(bars1, r2_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    bars2 = ax2.bar(
        model_names, rmse_values, color=colors, edgecolor="black", linewidth=1.5
    )
    ax2.set_ylabel("RMSE [W/m²]")
    ax2.set_title("Porovnanie RMSE")
    for bar, val in zip(bars2, rmse_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(fig_dir / "bonus_fusion_metrics.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Graf uložený: {fig_dir / 'bonus_fusion_metrics.png'}")
else:
    print("Bonus sekcia sa preskakuje - žiadne tabuľkové príznaky")

# %%
if len(tabular_available) > 0:
    image_feature_cols = [f"Image_PC{i + 1}" for i in range(X_img_train.shape[1])]
    tabular_feature_cols = [f"tab_{c}" for c in tabular_available]
    feature_names = image_feature_cols + tabular_feature_cols

    feature_importance = result_fusion["model"].feature_importances_

    sorted_idx = np.argsort(feature_importance)[::-1][:20]  # Top 20

    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = [feature_names[i] for i in sorted_idx]
    top_importance = feature_importance[sorted_idx]

    bar_colors = ["#3498db" if "Image" in f else "#e74c3c" for f in top_features]

    bars = ax.barh(
        range(len(top_features)), top_importance, color=bar_colors, edgecolor="black"
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel("Dôležitosť príznaku")
    ax.set_title(
        "Top 20 najdôležitejších príznakov vo fúznom modeli\n(modrá = obrazové, červená = tabuľkové)"
    )
    ax.grid(True, alpha=0.3, axis="x")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3498db", label="Obrazové príznaky"),
        Patch(facecolor="#e74c3c", label="Tabuľkové príznaky"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(
        fig_dir / "bonus_fusion_feature_importance.png", dpi=150, bbox_inches="tight"
    )
    plt.show()

    image_importance = sum(feature_importance[: len(image_feature_cols)])
    tabular_importance = sum(feature_importance[len(image_feature_cols) :])
    print("\nSúhrnná dôležitosť príznakov:")
    print(f"  Obrazové príznaky: {image_importance:.2%}")
    print(f"  Tabuľkové príznaky: {tabular_importance:.2%}")
else:
    print("Bonus sekcia sa preskakuje - žiadne tabuľkové príznaky")
from pathlib import Path
from PIL import Image
import polars as pl
from tqdm import tqdm


def resize_and_save_images(
    df: pl.DataFrame,
    data_dir: Path,
    output_dir: Path,
    image_size: tuple = (128, 128),
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


"""Data loading utilities using Polars."""

from pathlib import Path
from typing import List

import polars as pl
from PIL import Image
import numpy as np


def load_solar_data(month_folder: Path) -> pl.DataFrame:
    """
    Load solar data from a specific month folder.

    Args:
        month_folder: Path to month folder (e.g., dataset/Solar_data/01)

    Returns:
        Polars DataFrame with solar data
    """
    csv_path = month_folder / "out_data.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pl.read_csv(csv_path)

    df = df.with_columns(
        [pl.col("DateTime").str.split("#").list.get(0).alias("DateTime_clean")]
    )

    month_name = month_folder.name
    df = df.with_columns([pl.lit(month_name).alias("Month")])

    return df


def load_all_months(
    data_dir: Path, months: List[str] = ["01", "04", "07", "10"]
) -> pl.DataFrame:
    """
    Load data from all specified months.

    Args:
        data_dir: Path to Solar_data directory
        months: List of month folders to load

    Returns:
        Combined Polars DataFrame
    """
    dfs = []

    for month in months:
        month_path = data_dir / month
        if month_path.exists():
            print(f"Loading data from month {month}...")
            df = load_solar_data(month_path)
            dfs.append(df)
        else:
            print(f"Warning: Month folder {month} not found")

    if not dfs:
        raise ValueError("No data loaded. Check data directory.")

    combined_df = pl.concat(dfs)

    print(f"\nTotal records loaded: {len(combined_df)}")
    print(f"Months: {combined_df['Month'].unique().sort()}")

    return combined_df


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image and return as numpy array (RGB).

    Args:
        image_path: Path to image file

    Returns:
        Numpy array of shape (H, W, 3)
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def get_image_path(picture_name: str, data_dir: Path, month: str) -> Path:
    """
    Get full path to image file.

    Args:
        picture_name: Name from PictureName column
        data_dir: Path to Solar_data directory
        month: Month folder (e.g., "01")

    Returns:
        Path to image file
    """
    image_path = data_dir / month / "original" / picture_name
    return image_path


def verify_image_existence(df: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    """
    Add a column indicating whether the image file exists.

    Args:
        df: DataFrame with PictureName and Month columns
        data_dir: Path to Solar_data directory

    Returns:
        DataFrame with ImageExists column
    """
    image_exists = []

    for row in df.iter_rows(named=True):
        img_path = get_image_path(row["PictureName"], data_dir, row["Month"])
        image_exists.append(img_path.exists())

    df = df.with_columns([pl.Series("ImageExists", image_exists)])

    return df


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
    from .loading import verify_image_existence

    df = verify_image_existence(df, data_dir)

    df_valid = df.filter(pl.col("ImageExists"))

    print(f"Valid image-metadata pairs: {len(df_valid)} / {len(df)}")

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


import torch
import torch.nn as nn


class IrradianceCNN(nn.Module):
    def __init__(
        self,
        input_channels=3,
        conv_channels=[32, 64, 128],
        fc_hidden=[512, 256],
        dropout_rate=0.5,
        use_batch_norm=True,
    ):
        super(IrradianceCNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        in_channels = input_channels

        for out_channels in conv_channels:
            layers = []
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            self.conv_layers.append(nn.Sequential(*layers))
            in_channels = out_channels

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        fc_input_size = conv_channels[-1] * 7 * 7

        self.fc_layers = nn.ModuleList()
        prev_size = fc_input_size

        for hidden_size in fc_hidden:
            self.fc_layers.append(nn.Linear(prev_size, hidden_size))
            self.fc_layers.append(nn.ReLU(inplace=True))
            self.fc_layers.append(nn.Dropout(p=dropout_rate))
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, 1)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        x = self.output_layer(x)

        return x.squeeze(1)


def create_cnn_model(
    conv_channels=[32, 64, 128],
    fc_hidden=[512, 256],
    dropout_rate=0.5,
    use_batch_norm=True,
    device=None,
):
    if device is None:
        from .utils import get_device

        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    model = IrradianceCNN(
        input_channels=3,
        conv_channels=conv_channels,
        fc_hidden=fc_hidden,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    )

    model = model.to(device)

    return model


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
        image_size: tuple = (128, 128),
        transform=None,
        normalize=True,
        use_resized: bool = True,
    ):
        self.df = df
        self.data_dir = data_dir
        self.target_column = target_column
        self.image_size = image_size
        self.transform = transform
        self.normalize = normalize
        self.use_resized = use_resized

        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)

        if (
            self.use_resized
            and "ResizedImagePath" in row
            and row["ResizedImagePath"] is not None
        ):
            image_path = Path(row["ResizedImagePath"])
        else:
            month_str = str(row["Month"]).zfill(2)
            image_path = self.data_dir / month_str / "original" / row["PictureName"]

        image = Image.open(image_path).convert("RGB")

        if not (
            self.use_resized
            and "ResizedImagePath" in row
            and row["ResizedImagePath"] is not None
        ):
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
    image_size: tuple = (128, 128),
    num_workers: int = 0,
    normalize: bool = True,
    use_resized: bool = True,
):
    train_dataset = SolarDataset(
        train_df,
        data_dir,
        image_size=image_size,
        normalize=normalize,
        use_resized=use_resized,
    )

    val_dataset = SolarDataset(
        val_df,
        data_dir,
        image_size=image_size,
        normalize=normalize,
        use_resized=use_resized,
    )

    test_dataset = SolarDataset(
        test_df,
        data_dir,
        image_size=image_size,
        normalize=normalize,
        use_resized=use_resized,
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
    optimizer_type="adamw",
):
    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "adam":
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

    # IPEX optimize() vypnuté - spôsobuje extrémne spomalenie trénovania
    # Model stále beží na XPU, len bez dodatočnej optimalizácie
    # if IPEX_AVAILABLE:
    #     try:
    #         device = next(model.parameters()).device
    #         if str(device).startswith("xpu"):
    #             model.train()
    #             model, optimizer = ipex.optimize(
    #                 model, optimizer=optimizer, dtype=torch.float32, level="O1"
    #             )
    #             print("Model optimized with IPEX")
    #     except Exception as e:
    #         print(f"Could not optimize with IPEX: {e}")

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
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(str(filepath), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]


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


import torch

try:
    import intel_extension_for_pytorch as ipex

    IPEX_AVAILABLE = True
except (ImportError, OSError):
    IPEX_AVAILABLE = False


def get_device():
    if IPEX_AVAILABLE and hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"Intel Arc GPU detected: {torch.xpu.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU detected)")
        return device


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def predict_custom_images(model, images_dir, device, image_size=(128, 128)):
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

    unique_files = {}
    for img_path in image_files:
        unique_files[img_path.name] = img_path
    image_files = list(unique_files.values())

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

    unique_results = {}
    for r in custom_results:
        unique_results[r["filename"]] = r
    custom_results = list(unique_results.values())

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
