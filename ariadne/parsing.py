import pandas as pd

def parse_df(file_path: str,
             **read_csv_kw_params) -> pd.DataFrame:
    return pd.read_csv(file_path, **read_csv_kw_params)
