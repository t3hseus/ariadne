from typing import List

import gin

from ariadne_v2.transformations import BaseTransformer
from experiments.graph.inferrer import GraphPreprocessor
from experiments.graph.inferrer import SaveGraphs

from ariadne_v2.inference import Transformer


@gin.configurable
def graph_preprocessor(suffixes_df, **kwargs):
    return GraphPreprocessor(_suffixes_df=suffixes_df, kwargs=kwargs)


@gin.configurable
def save_graphs():
    return SaveGraphs()


@gin.configurable
def transformer(transforms: List[BaseTransformer]):
    return Transformer(transforms)
