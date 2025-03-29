import argparse
import pandas as pd

from .data_loader import load_data
from .model_training import ModelTrainer
from .model_evaluation import evaluate_model
from .model_prediction import load_model, predict
from .model_registry import save_model
from .preprocessing import DataPreprocessor

MODELS = {
    "logistic_regression": "logistic_regression.pkl",
    "random_forest": "random_forest.pkl",
    "lgbm": "lgbm.pkl"
}

def main():
    parser = argparse.ArgumentParser(description="CLI tool for ML model training and evaluation")
    parser.add_argument("--data", type=str, help="Path to the dataset (CSV format)")
    parser.add_argument("--task", choices=["train", "evaluate", "predict"], help="Task to perform")
    parser.add_argument("--model", choices=MODELS.keys(), help="Model to use")
    parser.add_argument("--predict_data", type=str, help="Path to new data for prediction (CSV format)")
    
    args = parser.parse_args()
    
    # Mode selection (Train/Evaluate/Predict)
    if args.task == "train":
        df = load_data(args.data)
        preprocessor = DataPreprocessor()
        df = preprocessor.preprocess(df)
        trainer = ModelTrainer()
        df = trainer.apply_smote(df, preprocessor.target_col, trainer.numerical_vars)
        X_train, X_test, y_train, y_test = trainer.train_test_split(df, preprocessor.target_col)
        
        # Model selection (LR/RF/LGBM)
        if args.model == "logistic_regression":
            model = trainer.train_logistic_regression(X_train, y_train)
        elif args.model == "random_forest":
            model = trainer.train_random_forest(X_train, y_train)
        elif args.model == "lgbm":
            model = trainer.train_lgbm(X_train, y_train)
        
        save_model(model, MODELS[args.model])
        print(f"{args.model} trained and saved successfully.")
    
    elif args.task == "evaluate":
        df = load_data(args.data)
        preprocessor = DataPreprocessor()
        df = preprocessor.preprocess(df)
        trainer = ModelTrainer()
        _, X_test, _, y_test = trainer.train_test_split(df, preprocessor.target_col)
        model = load_model(MODELS[args.model])
        metrics = evaluate_model(model, X_test, y_test)
        print(metrics)
    
    elif args.task == "predict":
        model = load_model(MODELS[args.model])
        df_new = load_data(args.predict_data)
        preprocessor = DataPreprocessor()
        df_new = preprocessor.preprocess(df_new)
        predictions = predict(model, df_new)
        print("Predictions:", predictions)

if __name__ == "__main__":
    main()
