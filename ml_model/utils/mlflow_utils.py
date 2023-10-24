import tempfile
from pathlib import Path


def log_figure(mlflow_instance, figure, file_name, width, height):
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / file_name
        figure.write_image(save_path, format="png", width=width, height=height)
        mlflow_instance.log_artifact(save_path, "Figures")
