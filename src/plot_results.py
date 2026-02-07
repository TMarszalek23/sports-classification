import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "results" / "results.csv"

df = pd.read_csv(csv_path)

plt.figure()
plt.bar(range(len(df)), df["accuracy"])
plt.xticks(range(len(df)), df["model"] + "_" + df["scenario"] + "_" + df["augmentation"], rotation=90)
plt.ylabel("Accuracy")
plt.title("Comparison of all experiments")
plt.tight_layout()

out_path = project_root / "results" / "accuracy_plot.png"
plt.savefig(out_path, dpi=150)

print("Plot saved to:", out_path)
