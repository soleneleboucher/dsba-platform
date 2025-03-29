import joblib
import os

MODEL_DIR = "models/"

# Save model
def save_model(model, filename: str):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filepath = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, filepath)

# Load model
def load_model(filename: str):
    filepath = os.path.join(MODEL_DIR, filename)
    return joblib.load(filepath)
