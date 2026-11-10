import pandas as pd
import os

normal  = pd.read_csv("data/pump_normal_only.csv")
warning = pd.read_csv("data/pump_warning_only.csv")
fault   = pd.read_csv("data/pump_fault_only.csv")

combined = pd.concat([normal, warning, fault], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
combined.to_csv("data/pump_combined_dataset.csv", index=False)

print(f"Total rows    : {len(combined)}")
print(f"Columns       : {list(combined.columns)}")
print()
print("Label distribution:")
print(combined["label"].value_counts().to_string())
print()
print(combined.drop(columns="label").describe().round(3).to_string())
print()
print("Saved to: data/pump_combined_dataset.csv")
