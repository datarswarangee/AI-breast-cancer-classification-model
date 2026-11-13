
from pathlib import Path
import joblib

model_paths = [
    Path("models/candidate/baseline_logistic_regression.joblib"),
    Path("models/candidate/explainable_logistic_regression.joblib"),
]

for path in model_paths:
    print("\n" + "=" * 65)
    print("FILE:", path)

    if not path.exists():
        print("File not found.")
        continue

    model = joblib.load(path)
    print("Object type:", type(model).__name__)

    if hasattr(model, "named_steps"):
        print("Pipeline steps:", list(model.named_steps.keys()))
        for name, step in model.named_steps.items():
            print(f"\nStep: {name}")
            print("Type:", type(step).__name__)

            if hasattr(step, "n_features_in_"):
                print("Input features:", step.n_features_in_)

            if hasattr(step, "get_support"):
                print("Selected feature count:", len(step.get_support(indices=True)))

    if hasattr(model, "n_features_in_"):
        print("\nModel input features:", model.n_features_in_)

    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
        print("Feature names stored:", len(names))
        print("First 10 feature names:", names[:10])
