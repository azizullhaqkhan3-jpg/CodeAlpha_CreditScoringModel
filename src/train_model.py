
from pathlib import Path

import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

for folder in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# 1. Download the dataset from UCI
print("Downloading German Credit dataset...")

dataset = fetch_ucirepo(id=144)

X = dataset.data.features.copy()
y = dataset.data.targets.iloc[:, 0].copy()

# Convert target values: 1 = Good, 2 = Bad
y = y.astype(str).str.strip().map({
    "1": "Good",
    "2": "Bad",
})

if y.isna().any():
    raise ValueError("Unexpected target values in the dataset.")

# Save a local copy of the dataset
df = X.copy()
df["Credit_Risk"] = y
df.to_csv(DATA_DIR / "german_credit.csv", index=False)

print("Dataset shape:", df.shape)
print("Target distribution:")
print(y.value_counts())


# 2. Separate input features and target
X = df.drop(columns=["Credit_Risk"])
y = df["Credit_Risk"]

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# 3. Prepare the preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# 4. Create the classification model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )),
])


# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 6. Train the model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)


# 7. Evaluate the model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)

good_index = list(model.classes_).index("Good")
y_test_good = (y_test == "Good").astype(int)

roc_auc = roc_auc_score(
    y_test_good,
    y_proba[:, good_index],
)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0,
)

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=["Good", "Bad"],
)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")
print("\nClassification Report:")
print(report)
print("Confusion Matrix (Good, Bad):")
print(matrix)


# 8. Save the trained model and evaluation results
model_path = MODEL_DIR / "credit_scoring_model.joblib"
joblib.dump(model, model_path)

with open(
    OUTPUT_DIR / "evaluation_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
    file.write(report)
    file.write("\nConfusion Matrix (Good, Bad):\n")
    file.write(str(matrix))

print("\nModel saved to:", model_path)
print("Evaluation report saved to:", OUTPUT_DIR)
print("\nTraining completed successfully!")
