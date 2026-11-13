from pathlib import Path
import joblib
import pandas as pd

model_path = Path("models/candidate/explainable_logistic_regression.joblib")
data_path = Path("data/processed_candidate/expression_matrix.csv")

model = joblib.load(model_path)
X = pd.read_csv(data_path, index_col=0)

expected = list(model.feature_names_in_)
actual = list(X.columns)

print("Training matrix shape:", X.shape)
print("Model expects genes:", len(expected))
print("Training matrix genes:", len(actual))
print("Exact gene-order match:", expected == actual)
print("Duplicate training gene names:", X.columns.duplicated().sum())
print("Missing expected genes:", len(set(expected) - set(actual)))
print("Unexpected genes:", len(set(actual) - set(expected)))

if expected != actual:
    print("\nFirst 10 expected genes:", expected[:10])
    print("First 10 actual genes:", actual[:10])
