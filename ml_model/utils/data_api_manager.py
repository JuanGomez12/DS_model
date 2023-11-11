import requests


class DataAPIManager:
    def __init__(self, data_api) -> None:
        self.base_url = f"{data_api}/power_plant_data/"

    def build_url(self, extra: str) -> str:
        return self.base_url + extra

    def get_response(self, url) -> dict:
        response = requests.get(url)
        return response.json()

    def get_total_rows(self) -> int:
        url = self.build_url("total_rows")
        return self.get_response(url)["rows"]

    def get_range(self, skip: int, limit: int) -> dict:
        url = self.build_url(f"retrieve_range?limit={limit}&skip={skip}")
        return self.get_response(url)

    def get_feature_types(self, return_id: bool = False) -> dict:
        url = self.build_url(f"column_types?return_id={return_id}")
        return self.get_response(url)

    def get_column_names(self, return_id: bool = False) -> dict:
        url = self.build_url(f"column_names?return_id={return_id}")
        return self.get_response(url)
