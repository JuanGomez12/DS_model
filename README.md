# DS_model
Playground for messing around with different Data Science related tools

Currently:
- Pre-commit hooks already set up.
- Containerized using Docker.
- Incorporates MLflow using MinIO for replacement of AWS S3 for storing MLflow objects.

## Machine Learning Modeling
![XGB pipeline example](ml_model/images/XGB_pipeline_example.png)

The project uses Scikit-learn's pipeline approach and standardized API for data preprocessing and model training. The current version includes the usage of an XGBoost model and a custom data cleaning class compatible with Scikit-learn's pipeline API.


## MLflow for machine learning lifecycle
![MLFlow example run](mlflow/images/Run_example.png)

Uses MLflow for maintaining the machine learning lifecycle, including:
- Logging
    - Data and configuration
    - Parameters
    - Metrics/results
    - Artifacts/models
- Project/experiment based tracking
- Model registry
    - Storage/organization/annotation of models
    - Deployment of models in different environments (e.g. PySpark or through a REST API)

### MinIO instance for artifact storage
![MinIO run locally](mlflow/images/MinIO%20local%20run.png)

Runs MinIO locally for storing S3 artifacts using nginx as a reverse proxy.


## Images



### MinIO run locally
