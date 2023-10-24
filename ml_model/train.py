from pathlib import Path

from modeling.data_preprocessor import DataPreprocessor
from modeling.pipeline import ProcessingPipeline
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils.load_config import load_config_file
from utils.logger import get_logger
from xgboost import XGBRegressor

import mlflow  # Pre-commit keeps thinking it's an local import

mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("xgb")

logger = get_logger(Path(__file__).stem)


# load config
config = load_config_file(Path("config") / "xgb.yml", logger)
modeling_config = config["modeling"]

# Enable autologging
mlflow.sklearn.autolog(**config["mlflow"])

with mlflow.start_run():
    diabetes_data = load_diabetes(return_X_y=True, as_frame=True)
    target_feature = "target"

    data = diabetes_data[0]
    data[target_feature] = diabetes_data[1]

    train, test = train_test_split(
        data,
        test_size=modeling_config["test_size"],
        random_state=modeling_config["random_seed"],
    )

    X_train = train.drop(columns=[target_feature])
    y_train = train[target_feature]

    X_test = test.drop(columns=[target_feature])
    y_test = test[target_feature]

    data_preprocessor = DataPreprocessor()

    processing_pipeline = ProcessingPipeline(modeling_config)

    estimator = XGBRegressor(**modeling_config["estimator"])
    pipeline = Pipeline(
        [
            ("data_preprocessor", data_preprocessor),
            ("processing_pipeline", processing_pipeline),
            ("estimator", estimator),
        ]
    )

    # Training
    pipeline.fit(X_train, y_train)

    if config["results"]:
        # plot and results
        px.scatter()

    # Register model
