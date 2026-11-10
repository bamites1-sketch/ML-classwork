import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 500

normal_df = pd.DataFrame({
    "vibration_mm_s":  np.round(np.clip(np.random.normal(2.0,  0.5,  n), 0.5,  3.5),  3),
    "rpm":             np.round(np.clip(np.random.normal(3000, 80,   n), 2800, 3200),  1),
    "temperature_c":   np.round(np.clip(np.random.normal(55,   6,    n), 40,   70),    2),
    "pressure_bar":    np.round(np.clip(np.random.normal(5.5,  0.6,  n), 4.0,  7.0),  3),
    "operating_hours": np.round(np.clip(np.random.normal(4000, 1500, n), 0,    8000),  1),
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
