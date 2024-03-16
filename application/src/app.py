from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from hydra import compose, initialize
from patsy import dmatrix
import joblib
import uvicorn

from hydra.utils import to_absolute_path as abspath

app = FastAPI()

with initialize(config_path="../../config"):
    config = compose(config_name="main")
    FEATURES = config.process.features
    MODEL_NAME = config.model.name
model_path = f"../../{config.model.path}"
print(model_path)

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

class PredictionResult(BaseModel):
    """
    Pydantic model representing prediction result.

    Attributes:
        prediction (float): The prediction output.
    """
    prediction: float  # Adjust this field as per your model's prediction output type


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

model = joblib.load(model_path)

@app.get("/")
def read_root():
    return "Employee Churn  Prediction App"

@app.get('/info')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": xgboost,
        "version": 2
    }


@app.get('/health')
async def service_health():
    """Return service health"""
    return {
        "ok"
    }



@app.post("/predict", response_model=PredictionResult)
def predict(employee: Employee):
    """
    Predict employee retention based on provided data.

    Args:
        employee (Employee): Employee data.

    Returns:
        PredictionResult: Prediction result.
    """
    df = pd.DataFrame(employee.dict(), index=[0])
    df = transform_data(df)
    result = model.predict(df)[0]
    return PredictionResult(prediction=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)