import pandas as pd
from pydantic import BaseModel


class PowerPlantData(BaseModel):
    temperature: float
    exhaust_vacuum: float
    atmospheric_pressure: float
    relative_humidity: float
    # electrical_output: float

    def to_frame(self) -> pd.DataFrame:
        data_dict = self.to_dict()
        df = pd.DataFrame(data_dict, index=[0])
        return df

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "exhaust_vacuum": self.exhaust_vacuum,
            "atmospheric_pressure": self.atmospheric_pressure,
            "relative_humidity": self.relative_humidity,
            # "electrical_output": self.electrical_output,
        }
