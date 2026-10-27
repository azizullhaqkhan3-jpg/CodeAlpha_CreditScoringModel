# CodeAlpha Credit Scoring Model

## 1. Project Overview

The Credit Scoring Model is a Machine Learning project developed as part of the CodeAlpha Machine Learning Internship.

The project uses historical financial data to classify individuals into two credit risk categories: **Good** and **Bad**.

The model is trained using the German Credit dataset from the UCI Machine Learning Repository.

## 2. Objectives

- Predict credit risk using historical data.
- Preprocess numerical and categorical features.
- Train a Random Forest classification model.
- Evaluate the model using standard classification metrics.
- Save the trained model for future predictions.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- UCI Machine Learning Repository

## 4. Machine Learning Algorithm

**Random Forest Classifier**

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions. It is used here to classify credit risk as Good or Bad.

## 5. Dataset

**Dataset:** German Credit Data

**Source:** UCI Machine Learning Repository

Dataset characteristics:

- Total records: 1,000
- Input features: 20
- Target classes: Good and Bad
- Good credit records: 700
- Bad credit records: 300

The dataset was downloaded programmatically and saved locally as `data/german_credit.csv`.

## 6. Project Structure

```text
CodeAlpha_CreditScoringModel/
├── data/
│   └── german_credit.csv
├── models/
│   └── credit_scoring_model.joblib
├── outputs/
│   └── evaluation_report.txt
├── src/
│   ├── train_model.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## 7. Data Preprocessing

The project performs the following preprocessing operations:

- Separates input features from the target variable.
- Identifies numerical and categorical features.
- Handles missing numerical values using median imputation.
- Handles missing categorical values using most-frequent imputation.
- Scales numerical features using StandardScaler.
- Encodes categorical features using OneHotEncoder.
- Splits the dataset into training and testing sets.

The dataset was divided as follows:

- Training records: 800 (80%)
- Testing records: 200 (20%)

The split uses `random_state=42` and stratification to preserve the class proportions.

## 8. Model Training

The Random Forest model was trained with these settings:

- Number of trees: 200
- Random state: 42
- Class weighting: Balanced

The trained model was saved to:

`models/credit_scoring_model.joblib`

## 9. Model Evaluation

The model was evaluated on 200 testing records.

| Metric | Result |
|---|---:|
| Accuracy | 69.00% |
| ROC-AUC | 0.7468 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Bad | 0.49 | 0.55 | 0.52 | 60 |
| Good | 0.80 | 0.75 | 0.77 | 140 |
| Macro Average | 0.64 | 0.65 | 0.64 | 200 |
| Weighted Average | 0.70 | 0.69 | 0.70 | 200 |

### Confusion Matrix

The terminal displayed the following confusion matrix in the order `Good, Bad`:

```text
[[105  35]
 [ 27  33]]
```

Interpretation:

- 105 Good records were correctly classified as Good.
- 35 Good records were incorrectly classified as Bad.
- 27 Bad records were incorrectly classified as Good.
- 33 Bad records were correctly classified as Bad.

**Note:** The model performs better at identifying Good credit records than Bad credit records. Further model evaluation and improvement may be needed.

## 10. Sample Prediction

The prediction script was executed successfully using the first record in the dataset.

Terminal output:

```text
===== CREDIT SCORING PREDICTION =====
Predicted category: Good
Probability of Bad: 9.00%
Probability of Good: 91.00%

Note: This is a demonstration, not a real lending decision.
```

This is an example prediction for one record, not an evaluation of the model's overall accuracy.

## 11. How to Install and Run

### Step 1: Install dependencies

Activate your virtual environment and run:

```bash
pip install -r requirements.txt
```

### Step 2: Train the model

```bash
python src/train_model.py
```

### Step 3: Run a sample prediction

```bash
python src/predict.py
```

The training script downloads the dataset when needed, trains the model, evaluates its performance, and saves the model and evaluation report.

## 12. Output Files

- `data/german_credit.csv` — downloaded dataset.
- `models/credit_scoring_model.joblib` — trained Machine Learning model.
- `outputs/evaluation_report.txt` — evaluation metrics and classification report.

## 13. Conclusion

The Credit Scoring Model successfully demonstrates a complete Machine Learning workflow, including data acquisition, preprocessing, model training, evaluation, and prediction.

The Random Forest model achieved 69% accuracy and a ROC-AUC score of 0.7468 on the test set. The results show that credit risk classification is possible, but the model requires further improvement before any practical application.

## 14. Disclaimer

This project is developed for educational purposes as part of the CodeAlpha Machine Learning Internship. Its predictions should not be used as the sole basis for real financial or lending decisions.
