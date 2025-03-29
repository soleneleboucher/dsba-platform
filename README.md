# Machine Learning CLI Tool

## Overview
This CLI tool allows users to train, evaluate, and predict using machine learning models on structured data. It supports logistic regression, random forests, and LightGBM models.

## Installation
Ensure you have the required dependencies installed:
```bash
pip install pandas numpy scikit-learn imbalanced-learn lightgbm optuna joblib
```

## Usage
Run the `main.py` script using the command line:

### 1. Train a Model
```bash
python main.py --data path/to/data.csv --task train --model logistic_regression
```

### 2. Evaluate a Model
```bash
python main.py --data path/to/data.csv --task evaluate --model logistic_regression
```

### 3. Make Predictions
```bash
python main.py --task predict --model logistic_regression --predict_data path/to/new_data.csv
```

## File Structure
- `dataloader.py` – Loads CSV data into Pandas DataFrame.
- `model_training.py` – Trains different ML models.
- `model_evaluation.py` – Evaluates trained models.
- `model_prediction.py` – Loads a model and makes predictions.
- `model_registry.py` – Saves and loads models.
- `main.py` – CLI entry point.

## Notes
- Ensure models are trained before running predictions.
- Models are saved in the `models/` directory.

