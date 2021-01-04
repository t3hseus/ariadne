import numpy as np
from collections import namedtuple

Points = namedtuple('Points', ['X', 'track'])

def points_to_sparse(points):
    return dict(X=points.X, track=points.track)

def sparse_to_points(X, track):
    return Points(X, track)

def save_points(points, filename):
    np.savez(filename, **points_to_sparse(points))

def save_points_new(graphs):
    for points_data_chunk in graphs:
        if points_data_chunk.processed_object is None:
            continue
        processed_graph: Points = points_data_chunk.processed_object
        save_points(processed_graph,
                   points_data_chunk.output_name)

def load_points(filename):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return sparse_to_points(**dict(f.items()))