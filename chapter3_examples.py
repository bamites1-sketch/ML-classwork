"""
Practical Engineering Case Study: Supervised Machine Learning Classification

Algorithms:
1. Support Vector Machines (SVM)
2. Naïve Bayes (Gaussian Naïve Bayes)
3. Decision Tree
4. Random Forest
5. AdaBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ============================================================
# PART 1. CREATE DATASET (30 RECORDS)
# ============================================================

data = [
    # ID, Study_Hours, Attendance, Assignment_Score, Previous_Exam, Academic_Risk

    [ 1, 18, 94, 81, 82, 0],
    [ 2, 23, 92, 86, 89, 0],
    [ 3, 22, 92, 84, 79, 0],
    [ 4, 14, 92, 90, 75, 0],
    [ 5, 21, 95, 84, 87, 0],
    [ 6, 19, 93, 94, 75, 0],
    [ 7, 16, 93, 87, 89, 0],
    [ 8, 20, 89, 92, 84, 0],
    [ 9, 21, 88, 81, 83, 0],
    [10, 17, 85, 79, 75, 0],
    [11, 14, 91, 90, 82, 0],
    [12, 21, 95, 87, 78, 0],
    [13, 14, 98, 85, 75, 0],
    [14, 19, 89, 81, 76, 0],
    [15, 17, 96, 79, 79, 0],

    [16,  4, 61, 48, 24, 1],
    [17,  6, 45, 49, 17, 1],
    [18,  4, 46, 39, 18, 1],
    [19,  4, 44, 30, 44, 1],
    [20,  4, 37, 30, 18, 1],
    [21,  7, 42, 38, 30, 1],
    [22,  4, 32, 26, 41, 1],
    [23,  3, 51, 31, 18, 1],
    [24,  6, 40, 30, 30, 1],
    [25,  6, 36, 24, 36, 1],
    [26,  5, 55, 25, 32, 1],
    [27,  3, 33, 47, 41, 1],
    [28,  1, 50, 30, 21, 1],
    [29,  1, 57, 35, 20, 1],
    [30,  2, 40, 46, 24, 1],
]

columns = [
    "Student_ID",
    "Study_Hours",
    "Attendance_Percentage",
    "Assignment_Score",
    "Previous_Exam_Score",
    "Academic_Risk"
]

df = pd.DataFrame(data, columns=columns)

print("\nDATASET: 30 STUDENT RECORDS")
print(df.to_string(index=False))

print("\nClass distribution:")
print(
    df["Academic_Risk"]
    .value_counts()
    .rename(index={0: "Low risk", 1: "High risk"})
)

# Separate predictors (X) and target (y).

feature_cols = [
    "Study_Hours",
    "Attendance_Percentage",
    "Assignment_Score",
    "Previous_Exam_Score"
]

X = df[feature_cols]
y = df["Academic_Risk"]

# Initial data visualization.

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Attendance_Percentage",
    hue="Academic_Risk",
    style="Academic_Risk",
    s=100
)

plt.title("Student study hours vs. attendance")
plt.xlabel("Study Hours per Week")
plt.ylabel("Attendance (%)")
plt.legend(title="0 = Low risk, 1 = High risk")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.heatmap(
    df[feature_cols + ["Academic_Risk"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation matrix of student academic variables")
plt.tight_layout()
plt.show()


# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================

# Stratification preserves the class ratio in both subsets.
# With only 30 records, test metrics will be highly sensitive
# to individual records.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# ============================================================
# PART 3. DEFINE FIVE INDEPENDENT CLASSIFIERS
# ============================================================

# SVM: scaling is important because SVM uses distances/margins.

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

# Gaussian Naive Bayes.

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

# Decision Tree.

decision_tree_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# Random Forest.

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

# AdaBoost.

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}


# ============================================================
# PART 4. TRAIN, PREDICT, EVALUATE, AND VISUALIZE EACH MODEL
# ============================================================

results = []
predictions = {}

for model_name, model in models.items():

    print("\n" + "=" * 65)
    print(model_name.upper())
    print("=" * 65)

    # Learn from the training records.

    model.fit(X_train, y_train)

    # Predict labels for unseen test records.

    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    # Calculate evaluation metrics.

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")

    print("\nClassification report:")

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Low risk", "High risk"],
        zero_division=0
    ))

    # Confusion matrix visualization.

    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Low risk", "High risk"],
        cmap="Blues",
        colorbar=False,
        ax=ax
    )

    ax.set_title(f"{model_name}: Confusion Matrix")

    plt.tight_layout()
    plt.show()


# ============================================================
# PART 5. COMPARE THE FIVE ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")

print("\nMODEL COMPARISON")
print(results_df.round(3))

results_df.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1.05)
)

plt.title("Comparison of five supervised learning algorithms")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# ============================================================
# PART 6. INTERPRET TREE-BASED FEATURE IMPORTANCE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n{name} feature importance:")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))

    plt.barh(
        importance.index,
        importance.values
    )

    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Student academic feature")

    plt.tight_layout()
    plt.show()


# Visualize the learned Decision Tree.

plt.figure(figsize=(16, 8))

plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=["Low risk", "High risk"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree learned from the training records")

plt.tight_layout()
plt.show()


# ============================================================
# PART 7. TEST A NEW STUDENT OBSERVATION
# ============================================================

# Example values are illustrative.
# They are not a real academic assessment.

new_student = pd.DataFrame([{
    "Study_Hours": 4,
    "Attendance_Percentage": 45,
    "Assignment_Score": 38,
    "Previous_Exam_Score": 32
}])

print("\nNEW STUDENT OBSERVATION")
print(new_student)

for name, model in models.items():

    predicted_class = int(model.predict(new_student)[0])

    label = "High academic risk" if predicted_class == 1 else "Low academic risk"

    print(f"{name:15s} prediction: {label}")


print("\nEND OF PRACTICAL CASE STUDY")