#!make

SHELL := /bin/bash

BLUE="\033[00;94m"
GREEN="\033[00;92m"
RED="\033[00;31m"
RESTORE="\033[0m"
YELLOW="\033[00;93m"
CYAN="\e[0;96m"
GREY="\e[2:N"




create_mlflow_bucket:
# To be run inside MinIO to create a bucket, or ignore if it already exists
	mc alias set s3 http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD
	mc mb s3/$AWS_S3_BUCKET_NAME --ignore-existing --region $MINIO_REGION

mlflow_bucket:
	source .env
	docker compose run --rm create_buckets

test_debug:
	conda run --no--capture--output -n test coverage_run -m pytest
	conda run --no--capture--output -n test coverage html -d /app/test/coverage_report

coverage_report:
	echo ${GREEN}Running unit tests...{RESTORE}
	docker compose run --rm test bash -c "make test_debug"

build_project_local:
	echo ${GREEN} Building project locally {RESTORE}
	docker compose --env-file ./.envs/local/local.env build minio mlflow
	docker compose run --rm create_buckets
	echo ${GREEN}Done{RESTORE}

startup_project_local:
	echo ${GREEN} Building project locally {RESTORE}
	docker compose --env-file ./.envs/local/local.env up -d minio mlflow fastapi
	docker compose --env-file ./.envs/local/local.env run --rm create_buckets
	echo ${GREEN}Done{RESTORE}

run_ml_model_local:
	echo ${GREEN} Running ML model locally {RESTORE}
	docker compose --env-file ./.envs/local/local.env up ml_model_train_cpu
	echo ${GREEN}Done{RESTORE}

deploy_local:
	make build_project_local
	make startup_project_local
	echo ${GREEN} Wait 5 seconds for everything to be set up correctly {RESTORE}
	sleep 5
	make run_ml_model_local
