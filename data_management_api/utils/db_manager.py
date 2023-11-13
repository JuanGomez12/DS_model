import logging
import os

import pandas as pd
import psycopg2
from utils.logger import get_logger


class PowerPlantDBManager:
    def __init__(self, logger: logging.Logger = get_logger("PowerPlantManager")):
        self.logger = logger
        self.host = os.environ.get("POSTGRES_HOST")
        self.database = os.environ.get("POSTGRES_DB")
        self.user = os.environ.get("POSTGRES_USER")
        self.password = os.environ.get("POSTGRES_PASSWORD")
        self.port = "5432"

        self.conn = None
        self.generate_connection()

    def rollback(self):
        """Rolls back changes done in the database."""
        self.generate_connection()
        cur = self.conn.cursor()
        cur.execute("ROLLBACK")
        cur.close()
        self.commit_connection()

    def generate_connection(self) -> psycopg2.connection:
        """Generates a connection with the database if it doesn't exist already.

        Returns:
            psycopg2.connection: Psycopg2 database connection.
        """
        if self.conn is None:
            self.logger.info("Creating connection to PostgreSQL")
            self.conn = psycopg2.connect(
                host=self.host, database=self.database, user=self.user, password=self.password, port=self.port
            )
        return self.conn

    def retrieve_column_names(self, table_name: str, ignore_id: bool = True) -> list:
        """Retrieves a list of column names from a given database table.

        Args:
            table_name (str): Table from which to retrieve the column names.
            ignore_id (bool, optional): If True, excludes the ID column, if the table has one, from the column list.
                Defaults to True.

        Returns:
            list: List of column names.
        """
        return list(self.retrieve_column_types(table_name, ignore_id).keys())

    def retrieve_column_types(self, table_name: str, ignore_id: bool = True) -> dict:
        """Retrieves a dictionary containing the column names as the keys and their
        respective data type as the values.

        Args:
            table_name (str): Table from which to retrieve the column names.
            ignore_id (bool, optional): If True, excludes the ID column, if the table has one, from the dictionary.
                Defaults to True

        Returns:
            dict: Dictionary of column names as keys and their respective data type as values.
        """
        if self.table_exists(table_name):

            self.generate_connection()
            query = (
                f"""SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';"""
            )
            cur = self.conn.cursor()
            cur.execute(query)
            response = cur.fetchall()
            cur.close()
            # response = [resp[0] for resp in response]
            if ignore_id:
                response = [resp for resp in response if resp[0] != "id"]
        else:
            response = []
        return dict(response)

    def commit_connection(self):
        """Commits the connection to the database."""
        if self.conn is not None:
            self.conn.commit()

    def close_connection(self):
        """Closes the connection to the database."""
        if self.conn is not None:
            self.logger.info("Closing connection to Postgresql")
            self.conn.close()
            self.conn = None

    def insert_row(self, table_name: str, args_dict: dict):
        """Inserts a row into a table in the database.

        Args:
            table_name (str): Name of the table on which to insert the row.
            args_dict (dict): Dictionary contianing the column names as keys and
            the values to add to them as values.
        """
        args_dict_lowercase = {str(k).lower(): v for k, v in args_dict.items()}
        if self.table_exists(table_name):
            columns = self.retrieve_column_names(table_name)
            self.generate_connection()
            query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(['%s' for _ in columns])})"
            cur = self.conn.cursor()
            args_list_ordered = [args_dict_lowercase[arg] for arg in columns]
            cur.execute(query, args_list_ordered)
            cur.close()
            self.commit_connection()

    def __retrieve_data(self, sql_query) -> pd.DataFrame:
        df = pd.read_sql_query(sql_query, self.generate_connection())
        df = df.set_index("id")
        return df

    def retrieve_data(self, table_name: str) -> dict:
        """Retrieves the data of a table in the database.

        Args:
            table_name (str): Name of the table.

        Returns:
            dict: Dictionary in the form {row_1: {column_name_1: value, column_name_2: value, ...}, ...}...
        """
        if self.table_exists(table_name):
            query = f"SELECT * from {table_name}"
            df = self.__retrieve_data(query)
        else:
            df = pd.DataFrame()
        return df.to_dict(orient="index")

    def retrieve_rows(self, table_name: str, limit: int = 100, offset: int = 0) -> dict:
        """Retrieves a set of rows from a specified table with a defined limit and
        offset.

        Args:
            table_name (str): Name of the table for which to retrieve the data
            limit (int, optional): Number of rows to retrieve. Defaults to 100.
            offset (int, optional): Offset from row 0 from which to retrieve the data. Defaults to 0.

        Returns:
            dict: Dictionary in the form {row_1: {column_name_1: value, column_name_2: value, ...}, ...}...
        """
        if self.table_exists(table_name):
            query = f"SELECT * from {table_name} LIMIT {limit} OFFSET {offset}"
            df = self.__retrieve_data(query)
        else:
            df = pd.DataFrame()
        return df.to_dict(orient="index")

    def retrieve_row(self, table_name: str, row_id: int) -> dict:
        """Retireves one row of data from a specified table.

        Args:
            table_name (str): Name of the table for which to retrieve the data
            row_id (int): Number of row to retrieve.

        Returns:
            dict: Dictionary in the form {row_1: {column_name_1: value, column_name_2: value, ...}, ...}...
        """
        if self.table_exists(table_name):
            query = f"SELECT * from {table_name} WHERE id={row_id}"
            df = self.__retrieve_data(query)
        else:
            df = pd.DataFrame()
        return df.to_dict(orient="index")

    def table_exists(self, table_name: str) -> bool:
        """Checks if a table exists in the database.

        Args:
            table_name (str): Name of the table to check for.

        Returns:
            bool: True if a table with that name exists on the database.
        """
        table_existence_command = f"""
        SELECT EXISTS(SELECT 1 FROM information_schema.tables
                        WHERE table_catalog='{self.database}' AND
                        table_schema='public' AND
                        table_name='{table_name}');
                        """
        self.generate_connection()
        cur = self.conn.cursor()
        cur.execute(table_existence_command)
        response = cur.fetchall()
        cur.close()
        return response[0][0]

    def retrieve_all(self, table_name: str) -> list:
        """Retrieves all of the data for a given table.

        Args:
            table_name (str): Name of the table for which to retrieve all of the data.

        Returns:
            list: List containing all of the data of the table specified.
        """
        if self.table_exists(table_name):
            self.generate_connection()
            cur = self.conn.cursor()
            cur.execute(f"SELECT * FROM {table_name}")
            response = cur.fetchall()
            cur.close()
        else:
            response = []
        return response

    def count_rows(self, table_name: str) -> int:
        """Counts the number of rows of data available in a table.

        Args:
            table_name (str): Name of the table to count the rows of.

        Returns:
            int: Number of rows of data.
        """
        if self.table_exists(table_name):
            self.generate_connection()
            cur = self.conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            response = cur.fetchall()[0][0]
            cur.close()
        else:
            response = 0
        return response
