from pathlib import Path

from modeling.data_preprocessor import DataPreprocessor
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from utils.load_config import load_config_file
from utils.logger import get_logger

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

    # Create the preprocessing pipeline
    if modeling_config["normalization_method"].lower() == "standardize":
        scaler = StandardScaler()
    elif modeling_config["normalization_method"].lower() == "normalize":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Normalization method not recognized")

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", scaler)])

    categoric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unavailable")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                selector(dtype_exclude=["category", "object"]),
            ),
            (
                "cat",
                categoric_transformer,
                selector(dtype_include=["category", "object"]),
            ),
        ],
        remainder="drop",
        sparse_threshold=0,  # XGB doesn't play well with sparse matrices
    )
