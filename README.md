# DS_model
[![Linting](https://github.com/JuanGomez12/DS_model/actions/workflows/pre-commit.yml/badge.svg?branch=main&event=push)](https://github.com/JuanGomez12/DS_model/actions/workflows/pre-commit.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)]( https://github.com/psf/black)

Playground for messing around with different Data Science related tools

Currently:
- Pre-commit hooks already set up.
- Containerized using Docker and orchestrated using Docker Compose.
- Incorporates MLflow for managing the machine learning lifecycle including model development, registry, and deployment, and experiment tracking.
- Uses MinIO to replace AWS S3 for storing MLflow objects.
- Deployed using the FastAPI web framework
- Option of training models on CPU or GPU

## Running the repository
In order to run the code correctly, you need to:
- Clone the repository
- Run: ```make deploy_local```

This will:
- Create the required containers (minio, mlflow server, etc)
- Create the mlflow bucket in MinIO in order to be able to store MLflow data.
- Run an instance of the implemented ML model.

What can then be accessed afterwards:
- MLflow can then be accessed by opening a web browser and pointing it to http://localhost:5000
- MinIO can be accessed by opening a web browser and accessing http://localhost:9001
  - The credentials for accessing the service can be found in .envs/local/local.env
- FastAPI API for predicting using the implemented ML model can be accessed through http://localhost:8000. The documentation for it can be accessed in http://localhost:8000/docs or http://localhost:8000/redoc
- FastAPI API for adding/checking the data being stored can be accessed through http://localhost:8080. The documentation for it can be accessed in http://localhost:8080/docs or http://localhost:8000/redoc


## Machine Learning Modeling
![XGB pipeline example](ml_model/images/XGB_pipeline_example.png)

The project uses Scikit-learn's pipeline approach and standardized API for data preprocessing and model training. It imputes all the input data, standardizes the numerical data and encodes the categorical data.

The current version includes the usage of an XGBoost model and a custom-made data cleaning class (`DataPreprocessor`) which is compatible with Scikit-learn's pipeline API.


## MLflow for machine learning lifecycle

Uses MLflow for maintaining the machine learning lifecycle, including:
### Logging
![MLFlow example run](mlflow/images/Run_example.png)
- Data: features and samples used for training the ML models
- Configuration: pipeline configuration, including any preprocessing done before training.
- Parameters: model training parameters.
- Models: trained models and pipelines.
- Metrics: model training metrics and results.
- Artifacts: additional files and images created while running the pipeline (e.g. result plots and graphs)
### Project/experiment based tracking
![MLflow Experiments View](<mlflow/images/MLflow Experiments View.png>)
### Model registry
![MLflow Model REgistry View](<mlflow/images/MLFlow Model Registry View.png>)
- Store, organize and annotate existing models.
- Deploy models in different environments (e.g. PySpark or through a REST API)


### MinIO instance for artifact storage
![MinIO run locally](mlflow/images/MinIO%20local%20run.png)

Creates and runs a MinIO server locally for storing S3 artifacts. It uses nginx as a reverse proxy.


## FastAPI
![Fast API Docs](<fastapi/images/Fast API Docs.png>)
Web framework used to deploy the machine learning models. RESTful API to handle model prediction requests.

Additional separate API used to push/read data to dedicated PostgreSQL service, using a RESTful API to handle creating new entries for the dataset as well as pulling data, feature names, etc.
![Data Management API Docs](<data_management/images/data_management_api.png>)


## Docker Compose
Defines the multi-container application to spin up or tear down all services with a single command. Services include:
- MLflow server
- MinIO server
- nginx for reverse proxying of MLflow AWS calls to MinIO.
- PostgreSQL server for managing MLflow data.
- pgadmin.
- training service, CPU based.
- development service, CPU-based
- FastAPI server for managing ML models.
- FastAPI server for managing the ML model training data.
- PostgreSQL server for storing/reading the ML model training data.
