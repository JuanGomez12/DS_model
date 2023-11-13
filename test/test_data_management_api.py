import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, Path(__file__).parent)

from data_management_api.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Basic FastAPI API for data management"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_power_plant_row_retrieval():
    id = 1
    response = client.get(f"/power_plant_data/{id}")
    assert response.status_code == 200
    assert response.json().keys() == [id]
    assert len(response.json().values()) == 1
