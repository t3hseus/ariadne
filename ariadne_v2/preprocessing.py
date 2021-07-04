import gin
import numpy as np
import pandas as pd
from typing import List, Iterable, Set
from abc import ABCMeta, abstractmethod

from ariadne_v2.data_chunk import DataChunk
from ariadne_v2.transformations import Compose, BaseTransformer

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
    def construct_chunk(self, chunk: DataChunk) -> DataChunk:
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
