"""
This module is just a convenience to train a simple classifier.
Its presence is a bit artificial for the exercice and not required to develop an MLOps platform.
The MLOps course is not about model training.
"""

from dataclasses import dataclass
import logging
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from dsba.model_registry import ClassifierMetadata
from .preprocessing import split_features_and_target, preprocess_dataframe


def train_simple_classifier(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start training a simple classifier")
    df = preprocess_dataframe(df)
    X, y = split_features_and_target(df, target_column)
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X, y)

    logging.info("Done training a simple classifier")
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="xgboost",
        target_column=target_column,
        hyperparameters={"random_state": 42},
        description="",
        performance_metrics={},
    )
    return model, metadata


def train_logistic_regression(
        df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start training a logistic regression")
    df = preprocess_dataframe(df)
    X, y = split_features_and_target(df, target_column)

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],  
        "max_iter": [1000, 2000, 5000]  
    }

    model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    logging.info(f"Best hyperparameters found: {best_params}")

    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="logistic regression",
        target_column=target_column,
        hyperparameters=best_params,
        description="Logistic Regression using GridSearchCV",
        performance_metrics={},
    )
    return best_model, metadata


"""
def train_logistic_regression(
        df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start training a logistic regression")
    df = preprocess_dataframe(df)
    X, y = split_features_and_target(df, target_column)

    param_grid = {
        "C": [0.01, 0.1, 1, 10],  
        "max_iter": [1000]  
    }

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    

    logging.info("Done training a simple classifier")

    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="logistic regression",
        target_column=target_column,
        hyperparameters={"random_state": 42},
        description="Logistic Regression using GridSearchCV",
        performance_metrics={},
    )
    return model, metadata
"""
