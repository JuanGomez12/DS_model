# DS_model
Playground for messing around with different Data Science related tools

Currently:
- Pre-commit hooks already set up.
- Containerized using Docker.
- Incorporates MLflow using MinIO for replacement of AWS S3 for storing MLflow objects.

## Machine Learning Modeling
![XGB pipeline example](ml_model/images/XGB_pipeline_example.png)

The project uses Scikit-learn's pipeline approach and standardized API for data preprocessing and model training. It imputes all the input data, standardizes the numerical data and encodes the categorical data.

The current version includes the usage of an XGBoost model and a custom-made data cleaning class (`DataPreprocessor`) which is compatible with Scikit-learn's pipeline API.

## MLflow for machine learning lifecycle
![MLFlow example run](mlflow/images/Run_example.png)

Uses MLflow for maintaining the machine learning lifecycle, including:
- Logging
    - Data: features and samples used for training the ML models
    - Configuration: pipeline configuration, including any preprocessing done before training.
    - Parameters: model training parameters.
    - Models: trained models and pipelines.
    - Metrics: model training metrics and results.
    - Artifacts: additional files and images created while running the pipeline (e.g. result plots and graphs)
- Project/experiment based tracking
- Model registry
    - Store, organize and annotate existing models.
    - Deploy models in different environments (e.g. PySpark or through a REST API)

### MinIO instance for artifact storage
![MinIO run locally](mlflow/images/MinIO%20local%20run.png)

Creates and runs a MinIO server locally for storing S3 artifacts. It uses nginx as a reverse proxy.
