from pathlib import Path

from fastapi import FastAPI
from utils.db_manager import PowerPlantDBManager
from utils.logger import get_logger
from utils.power_plant_data import PowerPlantData

app = FastAPI()


logger = get_logger(Path(__file__).stem)

TABLE_NAME = "powerplant"


@app.get("/")
def read_root():
    return {"message": "Basic FastAPI API for data management"}


@app.post("/power_plant_data/add")
async def add_power_plant_data(powerplant_data: PowerPlantData) -> dict:
    logger.info(f"Adding new row of data to Power Plant database")
    powerplant_db_manager = PowerPlantDBManager()
    powerplant_db_manager.insert_row(table_name=TABLE_NAME, args_dict=powerplant_data.to_dict())
    return {"data_added": powerplant_data.to_dict()}


@app.get("/power_plant_data/column_names")
async def get_power_plant_column_names(return_id: bool = False) -> dict:
    powerplant_db_manager = PowerPlantDBManager()
    column_names = powerplant_db_manager.retrieve_column_names(table_name=TABLE_NAME, ignore_id=not return_id)
    return {"column_names": column_names}


@app.get("/power_plant_data/column_types")
async def get_power_plant_column_types(return_id: bool = False) -> dict:
    powerplant_db_manager = PowerPlantDBManager()
    column_names = powerplant_db_manager.retrieve_column_types(table_name=TABLE_NAME, ignore_id=not return_id)
    return column_names


@app.get("/power_plant_data/total_rows")
async def get_power_plant_rows() -> dict:
    powerplant_db_manager = PowerPlantDBManager()
    row_count = powerplant_db_manager.count_rows(table_name=TABLE_NAME)
    return {"rows": row_count}


@app.get("/power_plant_data/retrieve_range")
async def get_plants(skip: int = 0, limit: int = 100) -> dict:
    powerplant_db_manager = PowerPlantDBManager()
    row_info = powerplant_db_manager.retrieve_rows(table_name=TABLE_NAME, limit=limit, offset=skip)
    return row_info


@app.get("/power_plant_data/{id}")
async def get_plant(id: int) -> dict:
    powerplant_db_manager = PowerPlantDBManager()
    row_info = powerplant_db_manager.retrieve_row(table_name=TABLE_NAME, row_id=id)
    return row_info


@app.get("/health")
def check_health() -> dict:
    """Performs a health check on the server, to see if it's alive.

    Returns:
        dict: Dictionary containing the status.
    """
    return {"status": "ok"}
