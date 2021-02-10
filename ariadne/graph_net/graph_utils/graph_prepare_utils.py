from typing import List

import gin
import pandas as pd
import numpy as np

from ariadne.graph_net.graph_utils.graph import Graph


def calc_dphi(phi1, phi2):
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2 * np.pi
    dphi[dphi < -np.pi] += 2 * np.pi
    return dphi


def get_edges_from_supernodes(sn_from, sn_to):
    prev_edges = sn_from.rename(columns={'to_ind': 'cur_ind'})
    next_edges = sn_to.rename(columns={'from_ind': 'cur_ind'})
    prev_edges['edge_index'] = prev_edges.index
    next_edges['edge_index'] = next_edges.index

    # merge segments by their common hit in the middle
    line_graph_edges = pd.merge(prev_edges, next_edges, on='cur_ind', suffixes=('_p', '_c'))

    line_graph_edges = line_graph_edges.assign(true_superedge=-1)

    # track the 'true' superedge. true_superedge is a combination of 2 consecutive true edges of 1 track
    index_true_superedge = line_graph_edges[(line_graph_edges.track_p == line_graph_edges.track_c) &
                                            (line_graph_edges.track_p != -1)]
    line_graph_edges.loc[index_true_superedge.index, 'true_superedge'] = index_true_superedge.track_p.values

    ba = line_graph_edges[['dx_p', 'dy_p', 'dz_p']].values
    cb = line_graph_edges[['dx_c', 'dy_c', 'dz_c']].values
    # this was an old weight function:
    # norm_ba = np.linalg.norm(ba, axis=1)
    # norm_cb = np.linalg.norm(cb, axis=1)
    # dot = np.einsum('ij,ij->i', ba, cb)
    # ^may be useful later but now weight function is just a norm:
    w = np.linalg.norm(ba - cb, axis=1)
    line_graph_edges['weight'] = w
    indexation = ['weight', 'true_superedge', 'edge_index_p', 'edge_index_c']
    return line_graph_edges[indexation]


@gin.configurable(denylist=['one_station_segments', 'station'])
def get_supernodes_df(one_station_segments,
                      axes,
                      suffix_p, suffix_c,
                      station=-1,
                      STATION_COUNT=0,
                      pi_fix=False, ):
    ret = pd.DataFrame()

    # a bit of overgeneralization, all values are from yaml config
    # usually suffixes are "_p" and "_c" (word "_previousЭ and "_current" shorten)
    # axes are the r, phi and z if there were transformation to the cylindrical coordinates

    x0_y0_z0 = one_station_segments[[axes[0] + suffix_p, axes[1] + suffix_p, axes[2] + suffix_p]].values
    x1_y1_z1 = one_station_segments[[axes[0] + suffix_c, axes[1] + suffix_c, axes[2] + suffix_c]].values
    # compute the difference of the hits coordinates information
    dx_dy_dz = x1_y1_z1 - x0_y0_z0

    # if we are in the cylindrical coordinates, we need to fix the radian angle values overflows
    if pi_fix:
        dy_fixed = calc_dphi(one_station_segments[axes[1] + suffix_p].values,
                             one_station_segments[axes[1] + suffix_c].values)
        ret = ret.assign(dx=dx_dy_dz[:, 0], dy=dy_fixed,

                         z_p=one_station_segments[axes[2] + suffix_p].values,
                         z_c=one_station_segments[axes[2] + suffix_c].values,
                         y_p=one_station_segments[axes[1] + suffix_p].values,
                         y_c=one_station_segments[axes[1] + suffix_c].values,

                         dz=dx_dy_dz[:, 2], z=(station + 1) / STATION_COUNT)
    else:
        ret = ret.assign(dx=dx_dy_dz[:, 0], dy=dx_dy_dz[:, 1],

                         z_p=one_station_segments[axes[2] + suffix_p].values,
                         z_c=one_station_segments[axes[2] + suffix_c].values,
                         y_p=one_station_segments[axes[1] + suffix_p].values,
                         y_c=one_station_segments[axes[1] + suffix_c].values,

                         dz=dx_dy_dz[:, 2], z=(station + 1) / STATION_COUNT)

    # saving this information in the dataframe for later use and for visualization and evalutation purposes
    ret = ret.assign(from_ind=one_station_segments[['index_old' + suffix_p]].values.astype(np.uint32))
    ret = ret.assign(to_ind=one_station_segments[['index_old' + suffix_c]].values.astype(np.uint32))
    ret = ret.assign(track=-1)
    ret = ret.set_index(one_station_segments.index)

    # saving this information in the dataframe for later use
    index_true_superedge = one_station_segments[
        (one_station_segments['track' + suffix_p] == one_station_segments['track' + suffix_c]) & \
        (one_station_segments['track' + suffix_p] != -1)]
    ret.loc[index_true_superedge.index, 'track'] = index_true_superedge['track' + suffix_p].values
    ret = ret.assign(station=station)
    return ret


def apply_nodes_restrictions(nodes,
                             restrictions_0, restrictions_1):
    return nodes[
        (nodes.dy > restrictions_0[0]) & (nodes.dy < restrictions_0[1]) &
        (nodes.dz > restrictions_1[0]) & (nodes.dz < restrictions_1[1])
        ]


@gin.configurable(denylist=['segments', 'restrictions_func'])
def get_pd_line_graph(segments,
                      restrictions_func,
                      restrictions_0, restrictions_1,
                      suffix_p, suffix_c):
    nodes = pd.DataFrame()
    edges = pd.DataFrame()

    by_stations = [df for (ind, df) in segments.groupby('station' + suffix_p)]
    # iterate each pair of stations
    # for each 'i' we will have a 2 dataframes of segments which go
    # from i-1'th to i'th stations and from i'th to i+1'th stations
    for i in range(1, len(by_stations)):
        # take segments (which will be nodes as a result) which go
        # from station i-1 to i AND from i to i+1
        supernodes_from = get_supernodes_df(by_stations[i - 1], station=i - 1, STATION_COUNT=3, pi_fix=True)
        supernodes_to = get_supernodes_df(by_stations[i], station=i, STATION_COUNT=3, pi_fix=True)

        if restrictions_func:
            # if restriction function is defined, apply it to edges (supernodes)
            supernodes_from = restrictions_func(supernodes_from, restrictions_0, restrictions_1)
            supernodes_to = restrictions_func(supernodes_to, restrictions_0, restrictions_1)

        nodes = nodes.append(supernodes_from, sort=False)
        nodes = nodes.append(supernodes_to, sort=False)

        # construct superedges (superedge is an edge which in fact
        # represents 2 consecutive edges (and these edges have the one common hit being the ending of the 1-st edge and
        # the beginning of the 2-nd edge), so it also captures information of the 3 consecutive hits)
        superedges = get_edges_from_supernodes(supernodes_from, supernodes_to)
        edges = edges.append(superedges, ignore_index=True, sort=False)

    nodes = nodes.loc[~nodes.index.duplicated(keep='first')]
    return nodes, edges


def to_pandas_graph_from_df(
        single_event_df: pd.DataFrame,
        suffixes=None,
        compute_is_true_track: bool = True
) -> pd.DataFrame:
    if suffixes is None:
        suffixes = ['_prev', '_current']
    assert single_event_df.event.nunique() == 1
    my_df = single_event_df.copy()
    my_df['index_old'] = my_df.index
    by_stations = [df for (ind, df) in my_df.groupby('station')]
    cartesian_product = pd.DataFrame()
    for i in range(1, len(by_stations)):
        cartesian_product = cartesian_product.append(
            pd.merge(by_stations[i - 1], by_stations[i], on='event', suffixes=suffixes),
            ignore_index=True, sort=False)
    if compute_is_true_track:
        pid1 = cartesian_product["track" + suffixes[0]].values
        pid2 = cartesian_product["track" + suffixes[1]].values
        cartesian_product['track'] = ((pid1 == pid2) & (pid1 != -1))
    return cartesian_product


@gin.configurable(denylist=['pd_edges_df'])
def apply_edge_restriction(pd_edges_df: pd.DataFrame,
                           edge_restriction: float):
    assert 'weight' in pd_edges_df
    return pd_edges_df[pd_edges_df.weight < edge_restriction]


@gin.configurable(allowlist=['index_label_prev', 'index_label_current'])
def construct_output_graph(hits,
                           edges,
                           feature_names,
                           feature_scale,
                           index_label_prev='edge_index_p',
                           index_label_current='edge_index_c'):
    # Prepare the graph matrices
    n_hits = hits.shape[0]
    n_edges = edges.shape[0]
    X = (hits[feature_names].values / feature_scale).astype(np.float32)
    Ri = np.zeros((n_hits, n_edges), dtype=np.float32)
    Ro = np.zeros((n_hits, n_edges), dtype=np.float32)
    y = np.zeros(n_edges, dtype=np.float32)
    # We have the segments' hits given by dataframe label,
    # so we need to translate into positional indices.
    # Use a series to map hit label-index onto positional-index.
    hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
    seg_start = hit_idx.loc[edges[index_label_prev]].values
    seg_end = hit_idx.loc[edges[index_label_current]].values
    # Now we can fill the association matrices.
    # Note that Ri maps hits onto their incoming edges,
    # which are actually segment endings.
    Ri[seg_end, np.arange(n_edges)] = 1
    Ro[seg_start, np.arange(n_edges)] = 1
    # Fill the segment labels
    pid1 = hits.track.loc[edges[index_label_prev]].values
    pid2 = hits.track.loc[edges[index_label_current]].values
    y[:] = ((pid1 == pid2) & (pid1 != -1))
    return Graph(X, Ri, Ro, y)
