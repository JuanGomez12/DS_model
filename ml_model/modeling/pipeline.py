import pandas as pd
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

    @staticmethod
    def create_feature_selection_map(
        pipeline, feature_selector_name: str = "feature_selector", preprocessor_name: str = "preprocessor"
    ):
        feature_map = pd.DataFrame(
            pipeline.named_steps[feature_selector_name].get_feature_names_out(
                pipeline.named_steps[preprocessor_name].get_feature_names_out()
            ),
            columns=["feature"],
        )
        # (q=quantitative feature, i= binary feature)
        feature_map["feature_type"] = "i"
        for var in ["numerical__", "numeric__", "num__"]:
            feature_map.loc[feature_map.feature.str.contains(var), "feature_type"] = "q"
        feature_map["feature"] = feature_map.feature.str.replace(" ", "_")
        for var in ["numerical__", "numeric__", "num__", "categorical__", "cat__"]:
            feature_map["feature"] = feature_map.feature.str.replace(var, "")
        return feature_map
