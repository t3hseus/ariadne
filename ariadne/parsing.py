import pandas as pd

from utils.jit_cacher import cache_result_df


@cache_result_df(csv_path_key='file_path')
def parse_df(file_path: str = '',
             **read_csv_kw_params) -> pd.DataFrame:
    return pd.read_csv(file_path, **read_csv_kw_params)
