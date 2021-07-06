import pandas as pd

from typing import List, Iterable, Callable, Union, Tuple, Any
from abc import ABCMeta, abstractmethod

from ariadne_v2.dataset import AriadneDataset
from ariadne_v2.preprocessing import DFDataChunk
from ariadne_v2.transformations import BaseTransformer, Compose


class IInferrer(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ITransformer(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, chunk: DFDataChunk) -> DFDataChunk:
        pass


class IPreprocessor(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, chunk: pd.DataFrame) -> Union[tuple, None]:
        pass


class IModelLoader(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class IEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class IPostprocessor(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, data:Any, ds: AriadneDataset, idx:str):
        pass


class EventInferrer(IInferrer):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


# class GlobalEventInferrer(object):
#     def __init__(self,
#               load_model: IModelLoader,
#               postprocess_results: IPostprocessor,
#               inferrer: EventInferrer):
#         self.load_model = load_model
#         self.postprocess = postprocess_results
#         self.inferrer = inferrer
#         self.loaded_model = None
#
#     def __call__(self, data):
#         pass


class Transformer(ITransformer):

    def __init__(self, transforms: List[BaseTransformer]):
        self.transforms = Compose(transforms)

    def __call__(self, df, return_hash=False) -> DFDataChunk:
        return self.transforms(df, return_hash=return_hash)
