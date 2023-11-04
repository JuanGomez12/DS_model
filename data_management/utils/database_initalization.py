""""
Data based on the data used for:
Pınar Tüfekci, Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods, International Journal of Electrical Power & Energy Systems, Volume 60, September 2014, Pages 126-140, ISSN 0142-0615, http://dx.doi.org/10.1016/j.ijepes.2014.02.027.
(http://www.sciencedirect.com/science/article/pii/S0142061514000908)

Heysem Kaya, Pınar Tüfekci , Sadık Fikret Gürgen: Local and Global Learning Methods for Predicting Power of a Combined Gas & Steam Turbine, Proceedings of the International Conference on Emerging Trends in Computer and Electronics Engineering ICETCEE 2012, pp. 13-18 (Mar. 2012, Dubai)
"""
import logging
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Union

import pandas as pd
import psycopg2
import requests
from logger import get_logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.db_manager import PowerPlantDBManager

logger = get_logger(Path(__file__).stem)
TABLE_NAME = "powerplant"
TABLE_VARIABLES = {"AT": "NUMERIC", "V": "NUMERIC", "AP": "NUMERIC", "RH": "NUMERIC", "PE": "NUMERIC"}


def download_file(url: str, local_file_path: Path) -> Path:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return local_file_path


class PowerPlantDBInitializer(PowerPlantDBManager):
    def __init__(self, logger: logging.Logger = get_logger("PowerPlantInitializer")):
        super().__init__(logger)

    def create_table(self, table_name: str, columns: dict):
        if not self.table_exists(table_name):
            self.logger.info(f"Creating table: {table_name}")
            create_command = f"""CREATE TABLE {table_name} (id SERIAL PRIMARY KEY, """
            dtypes_command = ", ".join([f"{col} {columns[col]} NOT NULL" for col in columns])
            full_command = create_command + dtypes_command + ")"
            self.generate_connection()
            cur = self.conn.cursor()
            cur.execute(full_command)
            cur.close()
            self.commit_connection()
            self.logger.info(f"Table: {table_name} created successfully")

    def delete_table(self, table_name: str):
        self.logger.info(f"Deleting table: {table_name}")
        delete_command = f"""DROP TABLE {table_name}"""
        self.generate_connection()
        cur = self.conn.cursor()
        cur.execute(delete_command)
        cur.close()
        self.commit_connection()
        self.logger.info(f"Table {table_name} deleted from database")


def create_tables(pplant_data_path: Union[Path, str]):
    df = pd.read_excel(str(pplant_data_path / "CCPP" / "Folds5x2_pp.xlsx"))

    try:
        powerplant_db_initializer = PowerPlantDBInitializer()
        powerplant_db_initializer.create_table(TABLE_NAME, TABLE_VARIABLES)

        if powerplant_db_initializer.count_rows(TABLE_NAME) < 1:
            logger.info(f"Pushing power plant data to PostgreSQL")
            for row in df.iterrows():
                powerplant_db_initializer.insert_row(TABLE_NAME, row[1].to_dict())
        else:
            logger.warning(
                f"Database already contains {powerplant_db_initializer.count_rows()} rows of data, skipping initialization"
            )

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if powerplant_db_initializer.conn is not None:
            powerplant_db_initializer.close_connection()


def main():
    url = "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download data
        logger.info(f"Downloading data from {url}")
        temp_file = Path(tmp_dir) / "powerplant.zip"
        download_file(url, temp_file)

        # Extract/uncompress it
        logger.info("Uncompressing data")
        pplant_data_path = Path(tmp_dir) / "pplant"
        with zipfile.ZipFile(temp_file, "r") as zip_ref:
            zip_ref.extractall(pplant_data_path)

        # Push to data server
        logger.info("Pushing data to PostgreSQL")
        create_tables(pplant_data_path)
        logger.info("Done")


if __name__ == "__main__":
    main()
