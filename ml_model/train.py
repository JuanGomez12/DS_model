import datetime
import time
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from joblib import dump
from mlflow.models.signature import infer_signature
from modeling.data_preprocessor import DataPreprocessor
from modeling.pipeline import ProcessingPipeline
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils.load_config import load_config_file
from utils.logger import get_logger
from utils.mlflow_utils import log_figure
from xgboost import XGBRegressor

import mlflow  # Pre-commit keeps thinking it's a local import

logger = get_logger(Path(__file__).stem)


# load config
config = load_config_file(Path("config") / "xgb.yml", logger)
path_config = config["paths"]
modeling_config = config["modeling"]
plot_config = config["plot_config"]

EXPERIMENT_NAME = config["mlflow"]["experiment_name"]
mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# Enable autologging
mlflow.sklearn.autolog(
    log_input_examples=config["mlflow"]["log_input_examples"],
    max_tuning_runs=config["mlflow"]["max_tuning_runs"],
    log_models=config["mlflow"]["log_models"],
)

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
            ("processing_pipeline", processing_pipeline.preprocessor),
            ("estimator", estimator),
        ]
    )

    # Training
    train_time_start = time.time()
    pipeline.fit(X_train, y_train)
    train_time_end = time.time()
    logger.info(f"Model ttaining time: {train_time_end-train_time_start} seconds")

    # Testing
    y_pred = pipeline.predict(X_test)

    logger.info("Creating visualizations")
    # plot and results
    fig = px.scatter(
        x=y_pred,
        y=y_test,
        color=X_test["age"],
        trendline="ols",
        trendline_scope="overall",
        trendline_color_override="black",
        labels={
            "x": "Predicted value",
            "y": "Actual value",
            "color": "Patient Age",
            "Overall Trendline": "Model Trendline",
        },
        title="Predicted vs actual values by age",
        width=plot_config["width"],
        height=plot_config["height"],
    )
    fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Prediction"), secondary_y=False)
    fig.update_layout(coloraxis=dict(colorbar=dict(orientation="h", y=-0.22)))

    # Save plot
    log_figure(mlflow, fig, "Predicted_vs_Actual_plot.png", width=plot_config["width"], height=plot_config["height"])

    # Save model to file
    model_save_path = config["paths"]["output_path"] / "xgb" / f"xgb_{datetime.datetime.now().date()}.pkl"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, str(model_save_path))

    # Register model
    # Here we could add logic to register model or not depending on previous iteration's results for example
    # This step can also be instead performed manually on the mlflow dashboard, looking at the metrics, parameters, etc
    if config["mlflow"]["auto_log_model"]:
        logger.info("Registering model")
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=EXPERIMENT_NAME,
            signature=signature,
            registered_model_name=f"XGB",
            input_example=X_test.sample(3),
        )
    logger.info("Done training model")
