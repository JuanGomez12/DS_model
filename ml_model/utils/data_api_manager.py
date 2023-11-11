import requests


class DataAPIManager:
    def __init__(self, data_api_url: str) -> None:
        """Adds the base_url to which the Data API Manager will make calls to.

        Args:
            data_api_url (str): URL to which the API will connect to, e.g. "localhost".
        """
        self.base_url = f"{data_api_url}/power_plant_data/"

    def build_url(self, extra_path: str) -> str:
        """Builds an URL using the base_url as the prefix.

        Args:
            extra_path (str): Path to append to the end of the base URL

        Returns:
            str: Full URL.
        """
        return self.base_url + extra_path

    def get_response(self, url: str) -> dict:
        """Makes a request to the passed URL, converting the response into JSON/dict.

        Args:
            url (str): URL for which to make a request.

        Returns:
            dict: Dictionary containing the JSON response from the URL/server.
        """
        response = requests.get(url)
        return response.json()

    def get_total_rows(self) -> int:
        """Queries the API for the number of rows of data available.

        Returns:
            int: Number of rows of data available.
        """
        url = self.build_url("total_rows")
        return self.get_response(url)["rows"]

    def get_range(self, skip: int, limit: int) -> dict:
        """Queries the API for a group of rows of data, skipping the first skip amount
        of rows and limiting the response to the limit amount of rows.

        Args:
            skip (int): Number of rows to skip from the beginning of the databse.
            limit (int): Maximum number of rows to return

        Returns:
            dict: Dictionary containing the row number as the key, and the data as its value.
        """
        url = self.build_url(f"retrieve_range?limit={limit}&skip={skip}")
        return self.get_response(url)

    def get_feature_types(self, return_id: bool = False) -> dict:
        """Queries the API for a dictionary of the columns of data available, as well as
        its type.

        Args:
            return_id (bool, optional): If True, returns the ID of the element of the database as one of the columns.
                Defaults to False.

        Returns:
            dict: Dictionary contianing the names of the columns as the keys, and their type as its value.
        """
        url = self.build_url(f"column_types?return_id={return_id}")
        return self.get_response(url)

    def get_column_names(self, return_id: bool = False) -> list:
        """Queries the API for a list of the column names available in the database.

        Args:
            return_id (bool, optional):  If True, returns the ID of the element of the database as one of the columns.
                Defaults to False.

        Returns:
            list: List containing the names of the columns of the database.
        """
        url = self.build_url(f"column_names?return_id={return_id}")
        return self.get_response(url)["column_names"]
