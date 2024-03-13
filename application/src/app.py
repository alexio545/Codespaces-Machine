import numpy as np
import pandas as pd
from fastapi import FastAPI
from hydra import compose, initialize
from patsy import dmatrix
from pydantic import BaseModel
import hydra
import joblib
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig



app = FastAPI()


class Employee(BaseModel):
    """
    Pydantic model representing employee data.

    Attributes:
        City (str): The city of the employee.
        PaymentTier (int): The payment tier of the employee.
        Age (int): The age of the employee.
        Gender (str): The gender of the employee.
        EverBenched (str): Whether the employee has been benched.
        ExperienceInCurrentDomain (int): The experience of the employee in the current domain.
    """
    City: str = "Pune"
    PaymentTier: int = 1
    Age: int = 25
    Gender: str = "Female"
    EverBenched: str = "No"
    ExperienceInCurrentDomain: int = 1


@hydra.main(config_path="../../config", config_name="main")
def load_model(model_path: str, config: DictConfig):
    """
    Load a machine learning model from a specified path.

    Args:
        model_path (str): Path to the serialized machine learning model file.
        config (DictConfig): Configuration object containing Hydra configuration.

    Returns:
        object: The loaded machine learning model.
    """
    joblib.load(model_path)
    model = load_model(abspath(MODEL_PATH))



def add_dummy_data(df: pd.DataFrame):
    """
    Add dummy rows to the DataFrame.

    This is necessary for patsy to create features similar to the training dataset.

    Args:
        df (pd.DataFrame): The original DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with dummy rows added.
    """
    rows = {
        "City": ["Bangalore", "New Delhi", "Pune"],
        "Gender": ["Male", "Female", "Female"],
        "EverBenched": ["Yes", "Yes", "No"],
        "PaymentTier": [0, 0, 0],
        "Age": [0, 0, 0],
        "ExperienceInCurrentDomain": [0, 0, 0],
    }
    dummy_df = pd.DataFrame(rows)
    return pd.concat([df, dummy_df])

def rename_columns(X: pd.DataFrame):
    """
    Rename columns of the DataFrame.

    Args:
        X (pd.DataFrame): The DataFrame with original column names.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """
    X.columns = X.columns.str.replace(r"\[", "_", regex=True).str.replace(r"\]", "", regex=True)
    return X

def transform_data(df: pd.DataFrame):
    """
    Transform the input DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be transformed.

    Returns:
        np.ndarray: Transformed data.
    """
    dummy_df = add_dummy_data(df)
    feature_str = " + ".join(FEATURES)
    dummy_X = dmatrix(f"{feature_str} - 1", dummy_df, return_type="dataframe")
    dummy_X = rename_columns(dummy_X)
    return dummy_X.iloc[0, :].values.reshape(1, -1)



@app.post("/predict")
async def predict(employee: Employee):
    """
    Transform the input data and make predictions.

    Args:
        employee (Employee): Employee data.

    Returns:
        np.ndarray: Model predictions.
    """
    df = pd.DataFrame(employee.dict(), index=[0])
    df = transform_data(df)
    result = model.predict(df)[0] # Assuming your model has a predict method
    return {"prediction": result}
