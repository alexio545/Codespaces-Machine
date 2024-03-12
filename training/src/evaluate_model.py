import warnings

# Ignore warnings
warnings.filterwarnings(action="ignore")

import hydra
import joblib
import mlflow
import pandas as pd
from helper import BaseLogger
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import os 

logger = BaseLogger()   # Model logging in mlflow


def load_data(path: DictConfig):
    """
    Load test data from the specified paths.

    Args:
        path (DictConfig): Configuration object containing paths to test data files.

    Returns:
        pd.DataFrame: Test features.
        pd.DataFrame: Test target.
    """
    X_test = pd.read_csv(abspath(path.X_test.path))
    y_test = pd.read_csv(abspath(path.y_test.path))
    return X_test, y_test


def load_model(model_path: str):
    """
    Load a trained model from the specified path.

    Args:
        model_path (str): Path to the trained model file.

    Returns:
        XGBClassifier: Trained XGBoost model.
    """
    return joblib.load(model_path)


def predict(model: XGBClassifier, X_test: pd.DataFrame):
    """
    Generate predictions using the trained model.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        X_test (pd.DataFrame): Test features.

    Returns:
        numpy.ndarray: Predicted labels.
    """
    return model.predict(X_test)


def log_params(model: XGBClassifier, features: list):
    """
    Log model parameters and features.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        features (list): List of feature names.
    """
    logger.log_params({"model_class": type(model).__name__})
    model_params = model.get_params()

    for arg, value in model_params.items():
        logger.log_params({arg: value})

    logger.log_params({"features": features})


def log_metrics(**metrics: dict):
    """
    Log evaluation metrics.

    Args:
        **metrics (dict): Evaluation metrics and their values.
    """
    logger.log_metrics(metrics)


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def evaluate(config: DictConfig):
    """
    Evaluate the performance of the trained model.

    Args:
        config (DictConfig): Configuration object containing model parameters,
            paths to test data files, and MLflow tracking URI.
    """
    mlflow.set_tracking_uri(config.mlflow_tracking_ui)
    # mlflow.set_experiment('employee-churn')
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow_PASSWORD

    with mlflow.start_run():

        # Load test data and trained model
        X_test, y_test = load_data(config.processed)
        model = load_model(abspath(config.model.path))

        # Get predictions
        prediction = predict(model, X_test)

        # Calculate evaluation metrics
        f1 = f1_score(y_test, prediction)
        print(f"F1 Score of this model is {f1}.")

        accuracy = accuracy_score(y_test, prediction)
        print(f"Accuracy Score of this model is {accuracy}.")

        # Log model parameters and features
        log_params(model, config.process.features)
        # Log evaluation metrics
        log_metrics(f1_score=f1, accuracy_score=accuracy)


if __name__ == "__main__":
    evaluate()
