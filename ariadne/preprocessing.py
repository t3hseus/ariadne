import gin
import pandas as pd
from typing import List, Iterable
from abc import ABCMeta, abstractmethod


class DataChunk(metaclass=ABCMeta):

    def __init__(self,
                 df_chunk_data: pd.DataFrame):
        self.df_chunk_data = df_chunk_data


class ProcessedDataChunk(metaclass=ABCMeta):

    def __init__(self,
                 processed_object: object):
        self.processed_object = processed_object


class ProcessedData(metaclass=ABCMeta):

    def __init__(self,
                 processed_data: List[ProcessedDataChunk]):
        self.processed_data = processed_data
        pass


class DataProcessor(metaclass=ABCMeta):

    def __init__(self,
                 processor_name: str,
                 output_dir: str,
                 data_df: pd.DataFrame
                 ):
        self.data_df = data_df
        self.processor_name = processor_name
        self.output_dir = output_dir

    @abstractmethod
    def generate_chunks_iterable(self) -> Iterable[DataChunk]:
        pass

    @abstractmethod
    def construct_chunk(self,
                        chunk_df: pd.DataFrame)->DataChunk:
        pass

    @abstractmethod
    def preprocess_chunk(self,
                         chunk: DataChunk,
                         idx: str) -> ProcessedDataChunk:
        pass

    @abstractmethod
    def postprocess_chunks(self,
                           chunks: List[ProcessedDataChunk]) -> ProcessedData:
        pass

    @abstractmethod
    def save_on_disk(self,
                     processed_data: ProcessedData):
        pass
