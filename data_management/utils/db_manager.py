import logging
import os

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
        self.generate_connection()
        cur = self.conn.cursor()
        cur.execute("ROLLBACK")
        cur.close()
        self.commit_connection()

    def generate_connection(self):
        if self.conn is None:
            self.logger.info("Creating connection to PostgreSQL")
            self.conn = psycopg2.connect(
                host=self.host, database=self.database, user=self.user, password=self.password, port=self.port
            )

    def retrieve_column_names(self, table_name: str, ignore_id: bool = True) -> list:
        if self.table_exists(table_name):
            self.generate_connection()
            query = f"Select * FROM {table_name} LIMIT 0"
            query = f"""SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND
            table_name = '{table_name}';"""
            cur = self.conn.cursor()
            cur.execute(query)
            response = cur.fetchall()
            cur.close()
            response = [resp[0] for resp in response]
            if ignore_id:
                response = [resp for resp in response if "id" not in resp]
        else:
            response = []
        return response

    def commit_connection(self):
        if self.conn is not None:
            self.conn.commit()

    def close_connection(self):
        if self.conn is not None:
            self.logger.info("Closing connection to Postgresql")
            self.conn.close()

    def insert_row(self, table_name: str, args_dict: dict):
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

    def table_exists(self, table_name: str) -> bool:
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
        if self.table_exists(table_name):
            self.generate_connection()
            cur = self.conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            response = cur.fetchall()[0][0]
            cur.close()
        else:
            response = 0
        return response
