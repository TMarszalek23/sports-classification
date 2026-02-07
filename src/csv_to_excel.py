import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "results" / "results.csv"
excel_path = project_root / "results" / "results.xlsx"

df = pd.read_csv(csv_path)
df.to_excel(excel_path, index=False)

print("Excel saved to:", excel_path)
