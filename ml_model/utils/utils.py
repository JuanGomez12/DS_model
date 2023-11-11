import datetime

import pandas as pd

POSTGRESQL_DATA_TYPES = {
    "bigint": int,
    "bit": bool,
    "boolean": bool,
    "bytea": bytes,
    "character": str,
    "character varying": str,
    "date": datetime,
    "double precision": float,
    "integer": int,
    "money": float,
    "numeric": float,
    "real": float,
    "serial": int,
    "smallint": int,
    "smallserial": int,
    "text": str,
    "time": datetime,
    "timestamp": pd.Timestamp,
}
