import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from utils.logger import get_logger

import mlflow
from fastapi import FastAPI
from mlflow import MlflowClient

sys.path.insert(0, "ml_model")


mlflow.set_tracking_uri("http://mlflow_server:5000")

app = FastAPI()

client = MlflowClient()

logger = get_logger(Path(__file__).stem)


class DiabetesData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

    def to_frame(self):
        data_dict = {
            "age": self.age,
            "sex": self.sex,
            "bmi": self.bmi,
            "bp": self.bp,
            "s1": self.s1,
            "s2": self.s2,
            "s3": self.s3,
            "s4": self.s4,
            "s5": self.s5,
            "s6": self.s6,
        }
        df = pd.DataFrame(data_dict, index=[0])
        return df


@app.get("/")
def read_root():
    return {"Basic FastAPI API"}


@app.post("/{ml_model}/{model_version}/predict")
async def predict_diabetes(ml_model: str, model_version: str, diabetes_data: DiabetesData) -> dict:
    """Retrieves an ML model with a specific version, in order to the risk of diabetes.

    Args:
        ml_model (str): ML model to predict on. E.g. XGB
        model_version (str): Model version to use. E.g. latest.
        diabetes_data (DiabetesData): Diabetes data including the variables described in the DiabetesData class.

    Returns:
        dict: Dictionary containing the response from the model as well as some ML model information.
    """
    logger.info(f"Retrieving model: {ml_model}, version: {model_version}")
    # queries are not yet implemented, will add extra control and parameters for the model
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{ml_model}/{model_version}")
    prediction = model.predict(diabetes_data.to_frame())[0]
    return {"prediction": str(prediction), "ml_model": ml_model, "model_version": model_version}


@app.get("/health")
def check_health() -> dict:
    """Performs a health check on the server, to see if it's alive.

    Returns:
        dict: Dictionary containing the status.
    """
    return {"status": "ok"}
