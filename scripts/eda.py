from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prediction_service.paths import DATA_RAW_DIR, FIGURES_DIR  # noqa: E402


def main() -> None:
    files = ["train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt"]
    columns = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"s{i}" for i in range(1, 22)]

    print("Task 1: Load all 4 sub-datasets (FD001-FD004) into DataFrames")
    datasets = {}
    for file in files:
        file_path = DATA_RAW_DIR / file
        datasets[file] = pd.read_csv(file_path, sep=r"\s+", header=None, names=columns)
    print("Data loaded successfully.\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Task 2: Inspect shape, dtypes, missing values")
    for name, df in datasets.items():
        print(f"--- {name} ---")
        print(f"Shape: {df.shape}")
        print(f"Total Missing Values: {df.isnull().sum().sum()}")
        print("Data Types (first 5 columns):")
        print(df.dtypes.head())
        print("-" * 30)
    print()

    print("Task 3: Plot sensor trends over cycles per engine - spot degradation")
    plt.figure(figsize=(15, 10))
    unit1_fd001 = datasets["train_FD001.txt"][datasets["train_FD001.txt"]["unit"] == 1]
    for i, sensor in enumerate(["s2", "s3", "s4", "s7"], 1):
        plt.subplot(2, 2, i)
        plt.plot(unit1_fd001["cycle"], unit1_fd001[sensor])
        plt.title(f"Sensor {sensor} Trend (FD001, Unit 1)")
        plt.xlabel("Cycle")
        plt.ylabel("Sensor Value")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sensor_trends_fd001_unit1.png")
    plt.close()
    print("Saved trend plot to 'sensor_trends_fd001_unit1.png'.\n")

    print("Task 4: Identify flat/useless sensors: s1, s5, s10, s16, s18, s19 -> drop them")
    sensors_to_drop = ["s1", "s5", "s10", "s16", "s18", "s19"]
    for name in datasets:
        original_shape = datasets[name].shape
        datasets[name] = datasets[name].drop(columns=sensors_to_drop)
        new_shape = datasets[name].shape
        print(f"{name}: Dropped sensors. Shape changed from {original_shape} to {new_shape}")
    print()

    print("Task 5: Compute correlation matrix between sensors")
    sensor_cols = [col for col in datasets["train_FD001.txt"].columns if col.startswith("s")]
    corr_matrix = datasets["train_FD001.txt"][sensor_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 8})
    plt.title("Sensor Correlation Matrix (FD001)")
    plt.savefig(FIGURES_DIR / "correlation_matrix_fd001.png")
    plt.close()
    print("Saved correlation matrix heatmap to 'correlation_matrix_fd001.png'.\n")

    print("Task 6: Check operating condition clusters (FD002/FD004 have 6 conditions)")
    fd002 = datasets["train_FD002.txt"]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="op_setting_1", y="op_setting_2", hue="op_setting_3", data=fd002, palette="viridis", alpha=0.6)
    plt.title("Operating Conditions Clustering in FD002 (6 expected)")
    plt.savefig(FIGURES_DIR / "op_conditions_clusters_fd002.png")
    plt.close()
    print("Saved operating condition clusters plot to 'op_conditions_clusters_fd002.png'.\n")

    print("Step 1 Complete! EDA outputs generated in the figures directory.")


if __name__ == "__main__":
    main()
