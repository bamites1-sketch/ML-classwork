import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 500

normal_df = pd.DataFrame({
    "vibration_mm_s":  np.round(np.random.uniform(0.5,  3.5,  n), 3),
    "rpm":             np.round(np.random.uniform(2800, 3200, n), 1),
    "temperature_c":   np.round(np.random.uniform(40,   70,   n), 2),
    "pressure_bar":    np.round(np.random.uniform(4.0,  7.0,  n), 3),
    "operating_hours": np.round(np.random.uniform(0,    8000, n), 1),
    "label":           "NORMAL"
})

os.makedirs("data", exist_ok=True)
normal_df.to_csv("data/pump_normal_only.csv", index=False)

print(f"Rows: {len(normal_df)}")
print(f"Label: {normal_df['label'].unique()[0]}")
print(f"Saved to: data/pump_normal_only.csv")
print()
print(normal_df.drop(columns="label").describe().round(3).to_string())
print()
print(normal_df.head().to_string(index=False))
