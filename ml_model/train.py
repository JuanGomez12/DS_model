import datetime
import tempfile
import time
from pathlib import Path

from joblib import dump
from mlflow.models.signature import infer_signature
from modeling.data_preprocessor import DataPreprocessor
from modeling.pipeline import ProcessingPipeline
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils.data_visualizer import DataVisualizer
from utils.load_config import load_config_file
from utils.logger import get_logger
from utils.mlflow_utils import log_plotly_figure
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
            ("feature_selector", VarianceThreshold()),
            ("estimator", estimator),
        ]
    )

    # Training
    train_time_start = time.time()
    pipeline.fit(X_train, y_train)
    train_time_end = time.time()
    logger.info(f"Model training time: {train_time_end-train_time_start} seconds")

    # Testing
    y_pred = pipeline.predict(X_test)

    logger.info("Creating visualizations")
    # Feature information
    with tempfile.TemporaryDirectory() as temp_dir:
        fmap_save_path = Path(temp_dir) / "feature_map.txt"
        fmap = ProcessingPipeline.create_feature_selection_map(
            pipeline=pipeline, feature_selector_name="feature_selector", preprocessor_name="processing_pipeline"
        )
        fmap.to_csv(str(fmap_save_path), sep="\t", header=False)
        mlflow.log_artifact(str(fmap_save_path), "Results")

        image_save_path = Path(temp_dir) / "XGB_tree.png"
        image_save_path = DataVisualizer.save_xgb_tree_to_file(
            image_save_path=image_save_path,
            xgb_estimator=pipeline.named_steps["estimator"],
            num_trees=0,
            fmap_save_path=fmap_save_path,
        )
        mlflow.log_artifact(image_save_path, "Figures")

    # Predicted vs Actual values Scatterplot
    fig = DataVisualizer.create_predict_actual_scatter(
        y_pred=y_pred,
        y_test=y_test,
        color=X_test["age"],
        width=plot_config["width"],
        height=plot_config["height"],
        color_label="Patient Age",
    )

    # Save plot
    log_plotly_figure(
        mlflow_instance=mlflow,
        figure=fig,
        file_name="Predicted_vs_Actual_plot.png",
        width=plot_config["width"],
        height=plot_config["height"],
    )

    # XGB info

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
