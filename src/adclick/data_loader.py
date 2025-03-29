import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Loads dataset from a CSV file."""
    return pd.read_csv(filepath, encoding="utf-8")