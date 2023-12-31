from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import figure
from xgboost import XGBRegressor
from xgboost import plot_tree as xgb_plot_tree
from xgboost import to_graphviz


class DataVisualizer:
    @staticmethod
    def create_predict_actual_scatter(y_pred: np.array, y_test: np.array, color: Optional[np.array] = None, **kwargs):
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 768)
        color_label = kwargs.get("color_label", "")
        fig = px.scatter(
            x=y_pred,
            y=y_test,
            color=color,
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="black",
            labels={
                "x": "Predicted value",
                "y": "Actual value",
                "color": color_label,
                "Overall Trendline": "Model Trendline",
            },
            title="Predicted vs actual values by age",
            width=width,
            height=height,
        )
        fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Prediction"), secondary_y=False)
        fig.update_layout(coloraxis=dict(colorbar=dict(orientation="h", y=-0.22)))
        return fig

    def plot_xgb_tree(xgb_estimator: XGBRegressor, num_trees: int, fmap_save_path: Union[Path, str]) -> figure:
        """Plots and XGBoost tree using matplotlib's plotly.

        Args:
            xgb_estimator (XGBRegressor): XGBoost estimator.
            num_trees (int): Number of the tree to plot.
            fmap_save_path (Union[Path, str]): File path to the feature map.

        Returns:
            figure: XGB tree figure.
        """
        fig, ax = plt.subplots(figsize=(30, 30), dpi=300)
        xgb_plot_tree(xgb_estimator, num_trees=num_trees, ax=ax, fmap=str(fmap_save_path), rankdir="LR")
        return fig

    def save_xgb_tree_to_file(
        image_save_path: Union[Path, str],
        xgb_estimator: XGBRegressor,
        num_trees: int,
        fmap_save_path: Union[Path, str],
        dpi: int = 250,
    ):
        image = to_graphviz(xgb_estimator, num_trees=num_trees, fmap=str(fmap_save_path), rankdir="LR")
        image.graph_attr = {"dpi": str(dpi)}
        image_save_path = Path(image_save_path).with_suffix("")  # Rendering adds the png suffix
        image_save_path = image.render(str(image_save_path), format="png")
        return image_save_path
