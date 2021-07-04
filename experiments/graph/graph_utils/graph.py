"""
This module contains code for interacting with hit graphs.
A Graph is a namedtuple of matrices X, Ri, Ro, y.
"""

from collections import namedtuple
from typing import List

import h5py
import numpy as np

# A Graph is a namedtuple of matrices (X, Ri, Ro, y)
from ariadne_v2 import jit_cacher

Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y'])


def graph_to_sparse(graph):
    Ri_rows, Ri_cols = graph.Ri.nonzero()
    Ro_rows, Ro_cols = graph.Ro.nonzero()
    return dict(X=graph.X, y=graph.y,
                Ri_rows=Ri_rows, Ri_cols=Ri_cols,
                Ro_rows=Ro_rows, Ro_cols=Ro_cols)


def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, dtype=np.uint8):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[Ri_rows, Ri_cols] = 1
    Ro[Ro_rows, Ro_cols] = 1
    return Graph(X, Ri, Ro, y)


def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph_to_sparse(graph))


def save_graph_hdf5_custom(db: h5py.File, path, graph: Graph):
    if f"{path}" in db:
        del db[f"{path}"]
    db.create_dataset(f"{path}/X", data=graph.X, shape=graph.X.shape, compression="gzip")
    db.create_dataset(f"{path}/y", data=graph.y, shape=graph.y.shape, compression="gzip")
    db.create_dataset(f"{path}/Ri", data=graph.Ri, shape=graph.Ri.shape, compression="gzip")
    db.create_dataset(f"{path}/Ro", data=graph.Ro, shape=graph.Ro.shape, compression="gzip")

def read_graph_hdf5_custom(db: h5py.File, path, graph: Graph):
    return Graph(X=db[f"{path}/X"][()],
                 y=db[f"{path}/y"][()],
                 Ri=db[f"{path}/Ri"][()],
                 Ro=db[f"{path}/Ro"][()])

def save_graph_hdf5(db, graph, filename):
    with jit_cacher.instance() as cacher:
        cacher.store_custom(db, filename, graph, save_graph_hdf5_custom)

def read_graph_hdf5(db, filename):
    with jit_cacher.instance() as cacher:
        return cacher.read_custom(db, filename, read_graph_hdf5_custom)

def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)


def save_graphs_new(graphs):
    for graph_data_chunk in graphs:
        if graph_data_chunk.processed_object is None:
            continue
        processed_graph: Graph = graph_data_chunk.processed_object
        save_graph(processed_graph,
                   graph_data_chunk.output_name)


def load_graph(filename):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))
