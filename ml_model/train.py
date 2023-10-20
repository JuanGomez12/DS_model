from pathlib import Path

from utils.load_config import load_config_file
from utils.logger import get_logger

import mlflow

mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("xgb")

log = get_logger(__name__)


# load config
config = load_config_file(Path("config") / "xgb.yml")
# Enable autologging
mlflow.sklearn.autolog(**config["mlflow"])
