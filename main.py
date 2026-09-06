# Student Placement Eligibility Prediction
# Logistic Regression Binary Classification

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# 1. LOAD DATASET

file_path = "dataset_07_student_placement_eligibility.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET PREVIEW ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 2. DATASET INFORMATION

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== SUMMARY STATISTICS ==========")
print(df.describe())

# 3. CHECK MISSING VALUES

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 4. CHECK DUPLICATES

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicates:", df.duplicated().sum())

# 5. CHECK TARGET CLASS BALANCE

print("\n========== TARGET CLASS BALANCE ==========")
print(df["target"].value_counts())

print("\nTarget proportions:")
print(df["target"].value_counts(normalize=True))

# 6. DEFINE FEATURES AND TARGET

features = [
    "cgpa",
    "attendance_pct",
    "coding_score",
    "projects_completed",
    "internship_months",
    "backlogs"
]

X = df[features]
y = df["target"]

# 7. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN / TEST SPLIT ==========")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 8. PREPROCESSING + LOGISTIC REGRESSION


model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# 9. TRAIN MODEL

model.fit(X_train, y_train)

print("\n========== MODEL TRAINED ==========")
print("Algorithm: Logistic Regression")
print("Scaling: StandardScaler")
print("Test size: 20%")
print("Random state: 42")

# 10. MAKE PREDICTIONS

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 11. EVALUATION METRICS

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# 12. CLASSIFICATION REPORT

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred, zero_division=0))

# 13. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Eligible", "Eligible"]
)

disp.plot()
plt.title("Student Placement Eligibility - Confusion Matrix")
plt.tight_layout()
plt.show()

# 14. COEFFICIENT INTERPRETATION

logistic_model = model.named_steps["logistic_regression"]

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": logistic_model.coef_[0]
})

coefficients["Absolute_Importance"] = coefficients["Coefficient"].abs()

coefficients = coefficients.sort_values(
    by="Absolute_Importance",
    ascending=False
)

print("\n========== FEATURE COEFFICIENTS ==========")
print(coefficients.to_string(index=False))

# 15. ODDS RATIOS

coefficients["Odds_Ratio"] = (
    coefficients["Coefficient"].apply(lambda x: __import__("math").exp(x))
)

print("\n========== ODDS RATIOS ==========")
print(
    coefficients[
        ["Feature", "Coefficient", "Odds_Ratio"]
    ].to_string(index=False)
)

# 16. CONCLUSION

# 15.5 ROC CURVE

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve - Logistic Regression")
plt.tight_layout()
plt.show()

# 15.6 FEATURE COEFFICIENT BAR CHART

plt.figure(figsize=(8, 5))

plt.barh(
    coefficients["Feature"],
    coefficients["Coefficient"]
)

plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Features")
plt.title("Feature Coefficients - Placement Eligibility")
plt.axvline(0, linewidth=1)
plt.tight_layout()
plt.show()

# 15.7 TARGET CLASS DISTRIBUTION

plt.figure(figsize=(6, 4))

df["target"].value_counts().sort_index().plot(
    kind="bar"
)

plt.xlabel("Placement Eligibility")
plt.ylabel("Number of Students")
plt.title("Target Class Distribution")
plt.xticks(
    [0, 1],
    ["Not Eligible (0)", "Eligible (1)"],
    rotation=0
)

plt.tight_layout()
plt.show()
