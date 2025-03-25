import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from dsba.model_training import train_simple_classifier, train_logistic_regression
from dsba.model_registry import ClassifierMetadata
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


@pytest.fixture
def sample_data():
    """Fixture to generate a sample dataset."""
    np.random.seed(42)
    df = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "target": np.random.choice([0, 1], size=100)
    })
    return df


def test_train_simple_classifier(sample_data):
    """Test training the XGBoost classifier."""
    model, metadata = train_simple_classifier(sample_data, "target", "test_model_xgb")

    assert isinstance(model, xgb.XGBClassifier), "The returned model should be an XGBClassifier"
    assert isinstance(metadata, ClassifierMetadata), "The returned metadata should be an instance of ClassifierMetadata"
    assert metadata.algorithm == "xgboost", "The algorithm mentioned in the metadata should be XGBoost"
    assert "random_state" in metadata.hyperparameters, "The hyperparameters field should contain 'random_state'"
    
    # Check if the model can make predictions
    X = sample_data.drop(columns=["target"])
    predictions = model.predict(X)
    assert len(predictions) == len(X), "The number of predictions should match the number of samples"


def test_train_logistic_regression(sample_data):
    """Test training the logistic regression model using GridSearchCV."""
    model, metadata = train_logistic_regression(sample_data, "target", "test_model_logistic")

    assert isinstance(model, LogisticRegression), "The returned model should be an instance of LogisticRegression"
    assert isinstance(metadata, ClassifierMetadata), "The returned metadata should be an instance of ClassifierMetadata"
    assert metadata.algorithm == "logistic regression", "The algorithm mentioned in the metadata should be 'logistic regression'"
    assert "C" in metadata.hyperparameters, "The hyperparameters should contain 'C'"
    assert "max_iter" in metadata.hyperparameters, "The hyperparameters should contain 'max_iter'"

    # Check if the model can make predictions
    X = sample_data.drop(columns=["target"])
    predictions = model.predict(X)
    assert len(predictions) == len(X), "The number of predictions should match the number of samples"


@patch("dsba.model_training.logging.info")
def test_logging(mock_logging, sample_data):
    """Test whether logging messages are correctly triggered."""
    train_simple_classifier(sample_data, "target", "test_model_xgb")
    train_logistic_regression(sample_data, "target", "test_model_logistic")

    # Verify that the training logs are triggered
    mock_logging.assert_any_call("Start training a simple classifier")
    mock_logging.assert_any_call("Start training a logistic regression")
