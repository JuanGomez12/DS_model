from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


class ProcessingPipeline:
    def __init__(self, modeling_config: dict) -> None:
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
                    "numeric",
                    numeric_transformer,
                    selector(dtype_exclude=["category", "object"]),
                ),
                (
                    "categorical",
                    categoric_transformer,
                    selector(dtype_include=["category", "object"]),
                ),
            ],
            remainder="drop",
            sparse_threshold=0,  # XGB doesn't play well with sparse matrices
        )
        self.preprocessor = preprocessor
