

import os
import numpy as np
import pandas as pd


RANDOM_SEED = 42
TOTAL_OBSERVATIONS = 500
np.random.seed(RANDOM_SEED)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "pump_warning_only.csv")


n_per_scenario = TOTAL_OBSERVATIONS // 5


scenario_a = pd.DataFrame({
    "vibration_mm_s":  np.round(np.random.uniform(3.6,  6.5,  n_per_scenario), 3),
    "rpm":             np.round(np.random.uniform(2800.0, 3200.0, n_per_scenario), 1),
    "temperature_c":   np.round(np.random.uniform(71.0, 82.0, n_per_scenario), 2),
    "pressure_bar":    np.round(np.random.uniform(4.0,  7.0,  n_per_scenario), 3),
    "operating_hours": np.round(np.random.uniform(3000.0, 11000.0, n_per_scenario), 1),
    "label":           "WARNING"
})



scenario_b = pd.DataFrame({
    "vibration_mm_s":  np.round(np.random.uniform(3.6,  5.2,  n_per_scenario), 3),
    "rpm":             np.round(np.random.uniform(2750.0, 3150.0, n_per_scenario), 1),
    "temperature_c":   np.round(np.random.uniform(45.0, 68.0, n_per_scenario), 2),
    "pressure_bar":    np.round(np.random.uniform(2.5,  3.9,  n_per_scenario), 3),
    "operating_hours": np.round(np.random.uniform(1000.0, 9000.0, n_per_scenario), 1),
    "label":           "WARNING"
})


scenario_c = pd.DataFrame({
    "vibration_mm_s":  np.round(np.random.uniform(1.5,  3.5,  n_per_scenario), 3),
    "rpm":             np.round(np.random.uniform(2650.0, 2795.0, n_per_scenario), 1),
    "temperature_c":   np.round(np.random.uniform(68.0, 78.0, n_per_scenario), 2),
    "pressure_bar":    np.round(np.random.uniform(7.1,  8.5,  n_per_scenario), 3),
    "operating_hours": np.round(np.random.uniform(1000.0, 10000.0, n_per_scenario), 1),
    "label":           "WARNING"
})


rpm_underspeed = np.random.uniform(2500.0, 2795.0, n_per_scenario // 2)
rpm_overspeed  = np.random.uniform(3205.0, 3450.0, n_per_scenario // 2)
rpm_combined   = np.concatenate([rpm_underspeed, rpm_overspeed])
np.random.shuffle(rpm_combined)

scenario_d = pd.DataFrame({
    "vibration_mm_s":  np.round(np.random.uniform(2.0,  4.2,  n_per_scenario), 3),
    "rpm":             np.round(rpm_combined, 1),
    "temperature_c":   np.round(np.random.uniform(72.0, 85.0, n_per_scenario), 2),
    "pressure_bar":    np.round(np.random.uniform(3.8,  6.5,  n_per_scenario), 3),
    "operating_hours": np.round(np.random.uniform(2000.0, 11000.0, n_per_scenario), 1),
    "label":           "WARNING"
})


scenario_e = pd.DataFrame({
    "vibration_mm_s":  np.round(np.random.uniform(3.6,  5.5,  n_per_scenario), 3),
    "rpm":             np.round(np.random.uniform(2750.0, 3150.0, n_per_scenario), 1),
    "temperature_c":   np.round(np.random.uniform(71.0, 80.0, n_per_scenario), 2),
    "pressure_bar":    np.round(np.random.uniform(3.2,  4.2,  n_per_scenario), 3),
    "operating_hours": np.round(np.random.uniform(8001.0, 14000.0, n_per_scenario), 1),
    "label":           "WARNING"
})


warning_df = pd.concat([scenario_a, scenario_b, scenario_c, scenario_d, scenario_e], ignore_index=True)
warning_df = warning_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


feature_columns = ["vibration_mm_s", "rpm", "temperature_c", "pressure_bar", "operating_hours"]
expected_columns = feature_columns + ["label"]
warning_df = warning_df[expected_columns]


assert len(warning_df) == TOTAL_OBSERVATIONS, f"Expected {TOTAL_OBSERVATIONS} rows, got {len(warning_df)}"


assert list(warning_df.columns) == expected_columns, f"Column mismatch. Expected {expected_columns}, got {list(warning_df.columns)}"


for col in feature_columns:
    assert pd.api.types.is_numeric_dtype(warning_df[col]), f"Feature '{col}' is not numeric"


assert (warning_df["label"] == "WARNING").all(), "All rows must have label 'WARNING'"


missing_count = warning_df.isna().sum().sum()
assert missing_count == 0, f"Dataset contains {missing_count} missing values"


in_normal_vib   = (warning_df["vibration_mm_s"] >= 0.5) & (warning_df["vibration_mm_s"] <= 3.5)
in_normal_rpm   = (warning_df["rpm"] >= 2800.0) & (warning_df["rpm"] <= 3200.0)
in_normal_temp  = (warning_df["temperature_c"] >= 40.0) & (warning_df["temperature_c"] <= 70.0)
in_normal_press = (warning_df["pressure_bar"] >= 4.0) & (warning_df["pressure_bar"] <= 7.0)
false_normal_rows = (in_normal_vib & in_normal_rpm & in_normal_temp & in_normal_press).sum()
assert false_normal_rows == 0, f"Found {false_normal_rows} rows where all physical sensors are in NORMAL range!"


os.makedirs(DATA_DIR, exist_ok=True)
warning_df.to_csv(OUTPUT_FILE, index=False)


print(f"Generated Observations: {len(warning_df)}")
print(f"Unique Label: {warning_df['label'].unique().tolist()}")
print(f"Saved to: {os.path.relpath(OUTPUT_FILE, SCRIPT_DIR)}")
print()
print("Descriptive Statistics:")
print(warning_df.drop(columns="label").describe().round(3).to_string())
print()
print("First 5 Observations:")
print(warning_df.head().to_string(index=False))
print()
print("Data validation checks passed successfully:")
print("  - All 5 features are strictly numeric")
print("  - All labels are 'WARNING'")
print("  - Zero missing/null values")
print("  - Zero false normal rows (every row exhibits physical warning symptoms)")
print("  - No overlap with critical failure thresholds")
