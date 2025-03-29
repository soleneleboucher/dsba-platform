# Ad Click Prediction - CLI Usage Guide

This project trains machine learning models to predict ad click conversion based on user interactions. The CLI provides options for training, evaluating, and predicting using different models.

---

## Installation

Ensure you have all dependencies installed:
```sh
pip install -r requirements.txt
```

---

## CLI Commands

### Train a Model

Train a machine learning model on the dataset.
```sh
python main.py train --model <model_name> --data <data_file> --output <model_output>
```

#### Example (Train a Logistic Regression Model):
```sh
python main.py train --model logistic --data ClickTraining.csv --output logistic.pkl
```

**Arguments:**
- `--model`: Type of model (`logistic`, `random_forest`, `lgbm`).
- `--data`: Path to the training dataset.
- `--output`: File to save the trained model.

---

### Evaluate a Model

Evaluate a trained model on a dataset.
```sh
python main.py evaluate --model <model_file> --data <data_file>
```

#### Example:
```sh
python main.py evaluate --model logistic.pkl --data ClickTraining.csv
```

**Arguments:**
- `--model`: Path to the trained model file.
- `--data`: Path to the dataset for evaluation.

---

### Make Predictions

Use a trained model to predict ad clicks on new data.
```sh
python main.py predict --model <model_file> --input <input_data> --output <prediction_output>
```

#### Example:
```sh
python main.py predict --model logistic.pkl --input new_data.csv --output predictions.csv
```

**Arguments:**
- `--model`: Path to the trained model file.
- `--input`: Path to new data for predictions.
- `--output`: File to save predictions.

---

## Additional Notes

- The preprocessing step automatically applies **scaling and SMOTE** (if needed).
- Ensure that the **scaler.pkl** is available when making predictions.
- The CLI provides options for different models (`logistic`, `random_forest`, `lgbm`).

For more details, refer to the project documentation.

