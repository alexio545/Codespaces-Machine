import bentoml
import numpy as np
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from hydra import compose, initialize
from patsy import dmatrix
from pydantic import BaseModel

with initialize(config_path="../../config"):
    config = compose(config_name="main")
    FEATURES = config.process.features
    MODEL_NAME = config.model.name

#
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

# model = bentoml.picklable_model.load_runner(
#     f"{MODEL_NAME}:latest", method_name="predict"
# )

model = bentoml.xgboost.get(f"{MODEL_NAME}:latest").to_runner()
# Create service with the model
service = bentoml.Service("predict_employee", runners=[model])




@service.api(input=JSON(pydantic_model=Employee), output=NumpyNdarray())
def predict(employee: Employee) -> np.ndarray:
    """
    Transform the input data and make predictions.

    Args:
        employee (Employee): Employee data.

    Returns:
        np.ndarray: Model predictions.
    """
    df = pd.DataFrame(employee.dict(), index=[0])
    df = transform_data(df)
    result = model.run(df)[0]
    return np.array(result)


