import gin
import numpy as np
import pandas as pd
from typing import List, Iterable, Set
from abc import ABCMeta, abstractmethod

from ariadne_v2.transformations import Compose, BaseTransformer


class DataChunk(metaclass=ABCMeta):
    def __init__(self,
                 np_index: np.ndarray,
                 np_chunk_data: np.ndarray,
                 columns: List[str]):
        self.index = np_index
        self.np_chunk_data = np_chunk_data
        self.columns = columns

    @staticmethod
    def from_df(df: pd.DataFrame):
        return DataChunk(np_chunk_data=df.values, np_index=df.index.values, columns=list(df.columns))

    def as_df(self):
        return pd.DataFrame(self.np_chunk_data, index=self.index, columns=self.columns)

class ProcessedDataChunk(metaclass=ABCMeta):
    def __init__(self,
                 processed_object: object):
        self.processed_object = processed_object


class ProcessedData(metaclass=ABCMeta):
    def __init__(self,
                 processed_data: List[ProcessedDataChunk]):
        self.processed_data = processed_data


class DataProcessor(metaclass=ABCMeta):
    def __init__(self,
                 processor_name: str,
                 output_dir: str,
                 transforms: List[BaseTransformer] = None):
        self.processor_name = processor_name
        self.output_dir = output_dir
        self.transformer = Compose(transforms)

    @abstractmethod
    def  construct_chunk(self, chunk: DataChunk)->DataChunk:
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
