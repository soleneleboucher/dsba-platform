from fastapi import FastAPI, HTTPException
import joblib
import json
import pandas as pd
from adclick.model_prediction import load_model, predict
from adclick.model_registry import save_model
from adclick.model_training import ModelTrainer
from adclick.preprocessing import DataPreprocessor
from adclick.model_evaluation import evaluate_model


app = FastAPI()

# Load the models
MODEL_DIR = "models_registry/"
scaler_path = MODEL_DIR + "scaler.pkl"


@app.api_route("/train/", methods = ["GET", "POST"])
async def train_model(query: str, target_column: str = 'Clicks_Conversion', model_type: str = "tune_lgbm"):
    """
    Train the lgbm model.
    The query should be a json string representing a record.
    """
    try:
        record = json.loads(query)

        record_df = pd.DataFrame([record])

        # Preprocess data 
        preprocessor = DataPreprocessor()
        df = preprocessor.preprocess(record_df)

        # Use SMOTE
        trainer = ModelTrainer(scaler_path=scaler_path)
        df_resampled = trainer.apply_smote(df, target_col=target_column, numerical_vars=trainer.numerical_vars)

        X_train, X_test, y_train, y_test = trainer.train_test_split(df_resampled, target_col=target_column)

        # Train the right model
        if model_type == "lgbm":
            model = trainer.train_lgbm(X_train, y_train)
        if model_type == "tune_lgbm":
            model = trainer.tune_lgbm(X_train, y_train)
        elif model_type == "random_forest":
            model = trainer.train_random_forest(X_train, y_train) 
        elif model_type == "logistic_regression":
            model = trainer.train_logistic_regression(X_train, y_train) 
        else:
            raise HTTPException(status_code=400, detail="Unknown model.")

        # Save the model
        model_name = f"{model_type}_model.pkl"
        model_path = MODEL_DIR + model_name
        save_model(model, model_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/predict_lgbm/", methods = ["GET", "POST"])
async def predict(query: str, columns: str, model_name: str = "lgbm_tuned.pkl"):
    """
    Predict the target column of a record using a model.
    The query should be a json string representing a record.
    """
    try: 
        record = json.loads(query)

        columns_list = columns.split(",")
        record_df = pd.DataFrame([record], columns = columns_list)

        preprocessor = DataPreprocessor()
        preprocessed_data = preprocessor.preprocess(record_df)

        model_path = MODEL_DIR + model_name
        model = load_model(model_path)

        predictions = predict(model, preprocessed_data)

        return {"prediction": predictions}
        
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/evaluate/")
async def evaluate_model(query: str, target_column: str, model_name: str = "lgbm_tuned.pkl"):
    """
    Evaluate the model.
    """
    try: 
        record = json.loads(query)

        record_df = pd.DataFrame([record])

        preprocessor = DataPreprocessor()
        df = preprocessor.preprocess(record_df)

        trainer = ModelTrainer(scaler_path=scaler_path)
        df_resampled = trainer.apply_smote(df, target_col=target_column, numerical_vars=trainer.numerical_vars)

        X_train, X_test, y_train, y_test = trainer.train_test_split(df_resampled, target_col=target_column)


        model_path = MODEL_DIR + model_name
        model = load_model(model_path)

        metrics = evaluate_model(model, X_test, y_test)
    
        return {"metrics": metrics}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
