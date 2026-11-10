import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 500
per_group = n // 5

pattern_1 = pd.DataFrame({
    "vibration_mm_s":  np.round(np.random.uniform(8.0,  14.0, per_group), 3),
    "rpm":             np.round(np.random.uniform(2900,  3200, per_group), 1),
    "temperature_c":   np.round(np.random.uniform(70,    85,   per_group), 2),
    "pressure_bar":    np.round(np.random.uniform(4.5,   6.5,  per_group), 3),
    "operating_hours": np.round(np.random.uniform(4000,  9000, per_group), 1),
    "label":           "FAULT",
})

pattern_2 = pd.DataFrame({
    "vibration_mm_s":  np.round(np.random.uniform(3.0,  6.0,  per_group), 3),
    "rpm":             np.round(np.random.uniform(2800,  3100, per_group), 1),
    "temperature_c":   np.round(np.random.uniform(90,    120,  per_group), 2),
    "pressure_bar":    np.round(np.random.uniform(4.0,   7.0,  per_group), 3),
    "operating_hours": np.round(np.random.uniform(2000,  8000, per_group), 1),
    "label":           "FAULT",
})

pattern_3 = pd.DataFrame({
    "vibration_mm_s":  np.round(np.random.uniform(5.0,  9.0,  per_group), 3),
    "rpm":             np.round(np.random.uniform(2900,  3200, per_group), 1),
    "temperature_c":   np.round(np.random.uniform(70,    90,   per_group), 2),
    "pressure_bar":    np.round(np.random.uniform(1.0,   3.0,  per_group), 3),
    "operating_hours": np.round(np.random.uniform(1000,  7000, per_group), 1),
    "label":           "FAULT",
})

pattern_4 = pd.DataFrame({
    "vibration_mm_s":  np.round(np.random.uniform(4.0,  7.0,  per_group), 3),
    "rpm":             np.round(np.random.uniform(2800,  3100, per_group), 1),
    "temperature_c":   np.round(np.random.uniform(75,    95,   per_group), 2),
    "pressure_bar":    np.round(np.random.uniform(9.0,   12.0, per_group), 3),
    "operating_hours": np.round(np.random.uniform(3000,  8500, per_group), 1),
    "label":           "FAULT",
})

pattern_5 = pd.DataFrame({
    "vibration_mm_s":  np.round(np.random.uniform(6.0,  10.0,  per_group), 3),
    "rpm":             np.round(np.random.uniform(2850,  3150,  per_group), 1),
    "temperature_c":   np.round(np.random.uniform(70,    88,    per_group), 2),
    "pressure_bar":    np.round(np.random.uniform(3.5,   6.5,   per_group), 3),
    "operating_hours": np.round(np.random.uniform(12000, 20000, per_group), 1),
    "label":           "FAULT",
})

fault_df = pd.concat([pattern_1, pattern_2, pattern_3, pattern_4, pattern_5], ignore_index=True)
fault_df = fault_df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
fault_df.to_csv("data/pump_fault_only.csv", index=False)

print("=" * 60)
print("FAULT pump dataset - validation report")
print("=" * 60)

print(f"Total number of rows:    {len(fault_df)}")
print(f"Total number of columns: {len(fault_df.columns)}")
print(f"Column names:            {list(fault_df.columns)}")
print(f"Unique label values:     {fault_df['label'].unique().tolist()}")
print()
print("Summary statistics for the five sensor features:")
print(fault_df.drop(columns="label").describe().round(3).to_string())
print()
print("First five observations:")
print(fault_df.head().to_string(index=False))
print()
print("=" * 60)
print("Checks")
print("=" * 60)

features = ["vibration_mm_s", "rpm", "temperature_c", "pressure_bar", "operating_hours"]

check_rows    = len(fault_df) == 500
check_label   = (fault_df["label"] == "FAULT").all()
check_missing = not fault_df.isnull().values.any()
check_numeric = all(pd.api.types.is_numeric_dtype(fault_df[f]) for f in features)

print(f"Exactly 500 observations:      {check_rows}    (found {len(fault_df)})")
print(f"Every label is FAULT:          {check_label}   (labels: {fault_df['label'].unique().tolist()})")
print(f"No missing values:             {check_missing}  (total NaN: {int(fault_df.isnull().sum().sum())})")
print(f"All five features are numeric: {check_numeric}")
print()
print("Saved to: data/pump_fault_only.csv")
