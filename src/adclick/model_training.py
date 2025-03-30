import numpy as np
import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

import optuna

from .model_registry import save_model


class ModelTrainer:
    def __init__(self, test_size=0.2, smote_threshold=0.3, scaler_path="scaler.pkl", numerical_vars = None):
        """
        Parameters:
        - test_size: Train-test split ratio (default: 0.2)
        - smote_threshold: Apply SMOTE if minority class is below this fraction (default: 0.3)
        - scaler_path (str, default="scaler.pkl"): Path to save the trained scaler.
        """
        self.test_size = test_size
        self.smote_threshold = smote_threshold
        self.scaler_path = scaler_path
        self.scaler = MinMaxScaler()

        # Numerical variables after feature engineering
        self.numerical_vars = numerical_vars if numerical_vars else ['Time_On_Previous_Website',	'Number_of_Previous_Orders', 'Daytime', 'DaytimeXOrders', 'InvTimeonPrevSite'] 


    def apply_smote(self, df, target_col, numerical_vars):

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Get proportion of each class
        class_counts = y.value_counts(normalize=True)

        # Apply SMOTE if class imbalance exceeds the threshold
        if class_counts.min() < self.smote_threshold:
            smote = SMOTE(sampling_strategy=self.smote_threshold, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X_resampled.columns), 
                                pd.DataFrame(y_resampled, columns=[target_col])], axis=1)
            return df_resampled
        
        # If not imbalanced, returns the dataframe as is
        return df

    def train_test_split(self, df, target_col):

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42, stratify=y)

        # Scale numerical variables #TO DO: Only scale for logistic regression!
        X_train[self.numerical_vars] = self.scaler.fit_transform(X_train[self.numerical_vars])
        X_test[self.numerical_vars] = self.scaler.transform(X_test[self.numerical_vars])
        
        joblib.dump(self.scaler, self.scaler_path)  # Save the scaler

        return X_train, X_test, y_train, y_test
        
    # Model 1. Logistic Regression
    def train_logistic_regression(self, X_train, y_train):
        model = LogisticRegression(max_iter = 5000)
        model.fit(X_train, y_train)
        save_model(model, "logistic_regression.pkl")
        return model

    # Model 2. Random Forest
    def train_random_forest(self, X_train, y_train):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        save_model(model, "random_forest.pkl")
        return model

    def tune_random_forest(self, X_train, y_train):
        """
        Tune the hyperparameters of the RF model
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        save_model(grid_search.best_estimator_, "random_forest_tuned.pkl")
        return grid_search.best_estimator_

    # Model 3. Light Gradient Boosting
    def tune_lgbm(self, X_train, y_train):
        """
        Tune hyperparameters of the Gradient Boosting Model
        """
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
            }
            stratKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            f1_scores = []
            
            for train_idx, val_idx in stratKF.split(X_train, y_train):
                X_train_CV, y_train_CV = X_train.iloc[train_idx], y_train.iloc[train_idx]
                X_val_CV, y_val_CV = X_train.iloc[val_idx], y_train.iloc[val_idx]
                
                model = LGBMClassifier(**params, random_state=42)
                model.fit(X_train_CV, y_train_CV)
                
                y_prob = model.predict_proba(X_val_CV)[:, 1]
                y_pred = (y_prob > 0.5).astype(int)
                f1 = f1_score(y_val_CV, y_pred)
                
                f1_scores.append(f1)
            
            return np.mean(f1_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        
        best_model = LGBMClassifier(**study.best_params)
        best_model.fit(X_train, y_train)
        save_model(best_model, "lgbm_tuned.pkl")
        return best_model

    def train_lgbm(self, X_train, y_train):
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        save_model(model, "lgbm.pkl")
        
        return model
