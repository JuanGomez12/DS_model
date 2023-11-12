import sys
from enum import Enum
from pathlib import Path

import mlflow
from fastapi import FastAPI
from mlflow import MlflowClient
from utils.logger import get_logger
from utils.power_plant_data import PowerPlantData

sys.path.insert(0, "ml_model")


mlflow.set_tracking_uri("http://mlflow_server:5000")

app = FastAPI()

client = MlflowClient()

logger = get_logger(Path(__file__).stem)


class ModelName(str, Enum):
    xgb = "XGB"


@app.get("/")
def read_root():
    return {"Basic FastAPI API"}


@app.post("/{ml_model}/{model_version}/predict_electrical_output")
async def predict_electrical_output(
    power_plant_data: PowerPlantData,
    ml_model: ModelName,
    model_version: str,
) -> dict:
    """Retrieves an ML model with a specific version, in order to predict the electrical output of a power plant.

    Args:
        power_plant_data (PowerPlantData): Power plant data including the variables described in the PowerPlantData class.
        ml_model (ModelName): ML model to predict on. E.g. XGB
        model_version (str): Model version to use. E.g. latest.

    Returns:
        dict: Dictionary containing the response from the model as well as some ML model information.
    """

    if ml_model is ModelName.xgb:
        logger.info(f"Retrieving model: {ml_model}, version: {model_version}")
        # queries are not yet implemented, will add extra control and parameters for the model
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{ml_model.value}/{model_version}")
        prediction = model.predict(power_plant_data.to_frame())[0]
        return {
            "predicted_electrical_output": str(prediction),
            "ml_model": ml_model.value,
            "model_version": model_version,
        }


@app.get("/health")
def check_health() -> dict:
    """Performs a health check on the server, to see if it's alive.

    Returns:
        dict: Dictionary containing the status.
    """
    return {"status": "ok"}
