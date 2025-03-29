import joblib
import pandas as pd

# Load model from file
def load_model(filepath: str):
    return joblib.load(filepath)

# Make predictions
def predict(model, X_new: pd.DataFrame):
    return model.predict(X_new)
