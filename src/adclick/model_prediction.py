import joblib
import pandas as pd

MODEL_DIR = "models_registry/"

# Load model from file
def load_model(filename: str):
    filepath = MODEL_DIR + filename
    return joblib.load(filepath)

# Load scaler
def load_scaler():
    return joblib.load(MODEL_DIR + "scaler.pkl")

# Make predictions
def predict(model, X_new: pd.DataFrame):
    scaler = load_scaler()  # Load the saved scaler
    X_scaled = scaler.transform(X_new)  # Scale new data
    return model.predict(X_scaled)
