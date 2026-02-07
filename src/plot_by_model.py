import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "results" / "results.csv"

df = pd.read_csv(csv_path)

for model in df["model"].unique():
    sub = df[df["model"] == model]

    plt.figure()
    for aug in ["A0", "A1", "A2", "A3"]:
        aug_data = sub[sub["augmentation"] == aug]
        plt.plot(aug_data["scenario"], aug_data["accuracy"], marker="o", label=aug)

    plt.title(f"Accuracy for {model}")
    plt.xlabel("Scenario")
    plt.ylabel("Accuracy")
    plt.legend(title="Augmentation")
    plt.ylim(0, 1)
    plt.grid(True)

    out_path = project_root / "results" / f"{model}_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("Saved:", out_path)
