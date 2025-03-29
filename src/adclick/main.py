import argparse
from data_loader import load_data
from preprocessing import preprocess_data, feature_engineering, apply_smote, split_data, normalize_data
from model_training import train_model
from model_evaluation import evaluate_model
from model_registry import save_model, load_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

def main():
    parser = argparse.ArgumentParser(description="Ad Click Prediction CLI")
    parser.add_argument("mode", choices=["train", "evaluate", "predict"], help="Mode: train, evaluate, or predict")
    parser.add_argument("--model", choices=["logistic", "random_forest", "lgbm"], help="Choose a model", required=True)
    parser.add_argument("--data", type=str, help="Path to dataset", required=True)
    parser.add_argument("--output", type=str, help="Path to save model", default="model.pkl")
    parser.add_argument("--input", type=str, help="Path to input data for prediction")
    
    args = parser.parse_args()
    
    # Load & preprocess data
    df = load_data(args.data)
    df = preprocess_data(df)
    df = feature_engineering(df)
    df = apply_smote(df, 0.7)
    X_train, X_test, y_train, y_test = split_data(df, "Clicks_Conversion")
    X_train, X_test = normalize_data(X_train, X_test, ["Time_On_Previous_Website", "Number_of_Previous_Orders", "Daytime"])
    
    if args.mode == "train":
        if args.model == "logistic":
            model = LogisticRegression()
        elif args.model == "random_forest":
            model = RandomForestClassifier()
        elif args.model == "lgbm":
            model = LGBMClassifier()

        model = train_model(model, X_train, y_train)
        save_model(model, args.output)
        print(f"Model saved to {args.output}")

    elif args.mode == "evaluate":
        model = load_model(args.output)
        evaluate_model(model, X_test, y_test)

    elif args.mode == "predict":
        if args.input is None:
            print("Error: --input is required for prediction mode.")
            return
        
        model = load_model(args.output)
        X_new = load_data(args.input)  # Assuming new data is in a CSV
        predictions = model.predict(X_new)
        print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()
