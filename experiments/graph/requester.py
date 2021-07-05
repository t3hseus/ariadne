from typing import List

import gin

from ariadne_v2.transformations import BaseTransformer
from experiments.graph.inferrer import GraphPreprocessor, SaveGraphsHDF5, GraphDataset, SaveGraphsToDataset
from experiments.graph.inferrer import SaveGraphs

from ariadne_v2.inference import Transformer


@gin.configurable
def graph_preprocessor(suffixes_df, **kwargs):
    return GraphPreprocessor(_suffixes_df=suffixes_df, kwargs=kwargs)


@gin.configurable
def save_graphs():
    return SaveGraphs()

@gin.configurable
def save_graphs_hdf5():
    return SaveGraphsHDF5()

@gin.configurable
def save_graphs_to_dataset():
    return SaveGraphsToDataset()

@gin.configurable
def graphs_dataset(dataset_name):
    return GraphDataset(dataset_name)

@gin.configurable
def transformer(transforms: List[BaseTransformer]):
    return Transformer(transforms)
