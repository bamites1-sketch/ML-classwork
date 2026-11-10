# Pump Condition Classification — Classwork 3.12

## Group

**Group name:** Beamlak, Biniyam and Mekdelawit

**Members:**

- Beamlak Tesfahun (0357/16)
- Biniyam Gebrie (0449/16)
- Mekdelawit Habtamu (1266/16)

Practical Engineering Case Study: classify each pump observation as **NORMAL**, **WARNING**, or **FAULT** using labelled historical sensor data.

## Features

| Feature | Unit | Description |
|---|---|---|
| vibration_mm_s | mm/s | Vibration level |
| rpm | — | Rotational speed |
| temperature_c | °C | Operating temperature |
| pressure_bar | bar | Pump pressure |
| operating_hours | h | Total hours in service |

## Dataset

| File | Label | Rows |
|---|---|---|
| data/pump_normal_only.csv | NORMAL | 500 |
| data/pump_warning_only.csv | WARNING | 500 |
| data/pump_fault_only.csv | FAULT | 500 |
| data/pump_combined_dataset.csv | All | 1500 |

## How to build the combined dataset

```bash
python combine_dataset.py
```

## Project structure

```
├── student1_fault_data.py       # FAULT data preparation
├── student2_warning_data.py     # WARNING data preparation
├── student3_normal_data.py      # NORMAL data preparation
├── combine_dataset.py           # Merges all three into one dataset
├── data/
│   ├── pump_fault_only.csv
│   ├── pump_warning_only.csv
│   ├── pump_normal_only.csv
│   └── pump_combined_dataset.csv
└── README.md
```

## Potential algorithms

Decision Tree, Random Forest, SVM, Naïve Bayes, AdaBoost

## Evaluation metrics

Recall for Fault, Precision for Fault, Macro F1, Confusion Matrix

> Predictions should support — not replace — engineering inspection and safety procedures.
