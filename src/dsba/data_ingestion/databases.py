from dataclasses import dataclass
from typing import Any
import pandas as pd
import sqlalchemy


@dataclass
class PostgresConfig:
    host: str
    port: int = 5432
    database: str = ""
    user: str = ""
    password: str = ""
    schema: str | None = None


def query_postgres(
    config: PostgresConfig, query: str, **pandas_kwargs: Any
) -> pd.DataFrame:
    connection_string = (
        f"postgresql://{config.user}:{config.password}@"
        f"{config.host}:{config.port}/{config.database}"
    )

    engine = sqlalchemy.create_engine(connection_string)

    with engine.connect() as connection:
        return pd.read_sql(query, connection, **pandas_kwargs)
