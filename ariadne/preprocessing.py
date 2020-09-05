import gin
import pandas as pd
from typing import List
from abc import ABCMeta, abstractmethod


class DataChunk(metaclass=ABCMeta):

    def __init__(self,
                 df_chunk_data: pd.DataFrame):
        self.df_chunk_data = df_chunk_data


class ProcessedDataChunk(metaclass=ABCMeta):

    def __init__(self,
                 processed_object: object):
        self.processed_object = processed_object
        pass


class ProcessedData(metaclass=ABCMeta):

    def __init__(self,
                 processed_data: List[ProcessedDataChunk]):
        self.processed_data = processed_data
        pass


class DataProcessor(metaclass=ABCMeta):

    def __init__(self,
                 processor_name: str,
                 data_df: pd.DataFrame
                 ):
        self._data_df = data_df
        self._processor_name = processor_name

    @abstractmethod
    def preprocess_chunk(self,
                         chunk: DataChunk,
                         idx: int) -> List[ProcessedDataChunk]:
        pass

    @abstractmethod
    def postprocess_chunk(self,
                          chunks: List[ProcessedDataChunk]) -> ProcessedData:
        pass

    @abstractmethod
    def save_on_disk(self,
                     processed_data: ProcessedData):
        pass
