import gin
import pandas as pd

@gin.configurable
class EventProcessor(object):

    def __init__(self,
                 processor_name: str,
                 data_df: pd.DataFrame = pd.DataFrame()
                 ):
        self._data_df = data_df
        self.processor_name = processor_name
