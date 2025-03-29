# [TO DO] Needs to be packaged into a class


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import optuna






# Train model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Hyperparameter tuning for RandomForest
def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Hyperparameter tuning for LightGBM using Optuna
def tune_lgbm(X_train, y_train):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
        }
        model = LGBMClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        return model

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial), n_trials=10)
    return LGBMClassifier(**study.best_params)
