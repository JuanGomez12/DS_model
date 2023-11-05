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

    def get_feature_types(self) -> dict:
        # TODO
        pass

    def get_column_names(self) -> list:
        # TODO
        pass
