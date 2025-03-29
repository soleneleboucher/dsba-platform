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
python -m src.adclick.main --data path/to/data.csv --task train --model logistic_regression
```

### 2. Evaluate a Model
```bash
python -m src.adclick.main --data path/to/data.csv --task evaluate --model logistic_regression
```

### 3. Make Predictions
```bash
python -m src.adclick.main --task predict --model logistic_regression --predict_data path/to/new_data.csv
```

## File Structure
```bash
project-root/
│
├── src/
│   └── adclick/
│       ├── __init__.py
│       ├── main.py
│       ├── dataloader.py
│       ├── model_training.py
│       ├── model_evaluation.py
│       ├── model_prediction.py
│       ├── model_registry.py
│
├── models_registry/        # Where trained models are saved
├── requirements.txt
└── README.md
```

## Notes
- Ensure models are trained before running predictions.
- Models are saved in the `models_registry/` directory.

