
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_scoring_model.joblib"
DATA_PATH = ROOT / "data" / "german_credit.csv"

# Load the trained model and dataset
model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

# Select a real record from the dataset as an example
sample = data.drop(columns=["Credit_Risk"]).iloc[[0]]

# Predict credit risk
prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print("\n===== CREDIT SCORING PREDICTION =====")
print("Predicted category:", prediction)

for category, probability in zip(model.classes_, probabilities):
    print(f"Probability of {category}: {probability:.2%}")

print("\nNote: This is a demonstration, not a real lending decision.")
