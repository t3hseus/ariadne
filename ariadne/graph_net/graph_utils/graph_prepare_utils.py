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


def get_edges_from_supernodes(a, b,indices):

    indices_prev = indices.copy()
    indices_next = indices.copy()

    indices_prev['cur_ind'] = indices_prev.pop('to_ind')
    indices_next['cur_ind'] = indices_next.pop('from_ind')

    indices_prev['edge_index'] = indices_prev.pop('index')
    indices_next['edge_index'] = indices_next.pop('index')

    cur_ind_a =  indices_prev['cur_ind']
    cur_ind_b =  indices_next['cur_ind']

    to_merge = np.empty(shape=(b.shape[0] * a.shape[0], 2), dtype=np.int32)
    to_merge[:, 0] = np.tile(b[:, cur_ind_b ], (1, a.shape[0]))
    
    for i in range(a.shape[0]):
        to_merge[i * b.shape[0]:(i + 1) * b.shape[0], 1] = a[ i,cur_ind_a ]

    merge_ind = [(ind // b.shape[0], ind % b.shape[0]) for ind in np.where(to_merge[:, 0] == to_merge[:, 1])[0]]

    line_graph_edges = np.empty(shape=(len(merge_ind),  a.shape[1]+b.shape[1] + 2  ))
    for k, (i, j) in enumerate(merge_ind):
        line_graph_edges[k, :-2] = np.hstack((a[i, :], b[j, :]))
    # merge segments by their common hit in the middle

    indices_merged = {**{k +'_p': v for k, v in indices_prev.items()}, ** {k +'_c': v + len( indices_prev ) for k, v in indices_next.items()} }

    indices_merged['from_ind'] = indices_merged.pop('from_ind_p')
    indices_merged['to_ind'] = indices_merged.pop('to_ind_c')
    indices_merged['cur_ind'] = indices_merged.pop('cur_ind_c')

    line_graph_edges[:,-2] = -1
    indices_merged['true_superedge'] = line_graph_edges.shape[1] - 2
    indices_merged['weight'] = line_graph_edges.shape[1] - 1

    # track the 'true' superedge. true_superedge is a combination of 2 consecutive true edges of 1 track
    index_true_superedge = (line_graph_edges[:, indices_merged['track_p'] ]== line_graph_edges[:, indices_merged['track_c'] ]) & \
                                            (line_graph_edges[:, indices_merged['track_p'] ] != -1)
    
    line_graph_edges[ index_true_superedge, indices_merged['true_superedge'] ] = line_graph_edges[ index_true_superedge, indices_merged['track_p'] ]

    ba = line_graph_edges[:, [ indices_merged['dx_p'], indices_merged['dy_p'], indices_merged['dz_p']]]
    cb = line_graph_edges[:, [ indices_merged['dx_c'], indices_merged['dy_c'], indices_merged['dz_c']]]
   
    w = np.linalg.norm( (ba - cb).astype(np.float64), axis=1)

    line_graph_edges[:,indices_merged['weight']] = w
    indexation = ['weight', 'true_superedge', 'edge_index_p', 'edge_index_c', 'from_ind', 'cur_ind', 'to_ind']
    res = line_graph_edges[:,[indices_merged[ind] for ind in indexation]]
    return line_graph_edges[:,[indices_merged[ind] for ind in indexation]],indexation

def get_supernodes_df(one_station_segments, column_index,
                      axes,
                      suffix_p, suffix_c,
                      station=-1,
                      STATION_COUNT=0,
                      pi_fix=False, ):
    
    x0_y0_z0_ = one_station_segments[:,[ column_index[axes[0] + suffix_p], column_index[axes[1] + suffix_p], column_index[axes[2] + suffix_p]]]
    x1_y1_z1_ = one_station_segments[:,[ column_index[axes[0] + suffix_c], column_index[axes[1] + suffix_c], column_index[axes[2] + suffix_c]]]

    dx_dy_dz = x1_y1_z1_ - x0_y0_z0_

    # if we are in the cylindrical coordinates, we need to fix the radian angle values overflows

    if pi_fix:
        dy_fixed = calc_dphi(one_station_segments[:,column_index[axes[1] + suffix_p] ],
                             one_station_segments[:,column_index[axes[1] + suffix_c] ])
  
    ret_ = np.column_stack([dx_dy_dz[:, 0], dy_fixed,one_station_segments[:,column_index[axes[2] + suffix_p]] \
                           ,one_station_segments[:,column_index[axes[2] + suffix_c]],\
                           one_station_segments[:,column_index[axes[1] + suffix_p]], \
                           one_station_segments[:,column_index[axes[1] + suffix_c]], \
                           dx_dy_dz[:, 2],np.full( (one_station_segments.shape[0] , 1), (station + 1) / STATION_COUNT) ,\
                           one_station_segments[:,column_index['index_old' + suffix_p] ],\
                           one_station_segments[:,column_index['index_old' + suffix_c] ], \
                           np.full( (one_station_segments.shape[0] , 1),-1), \
                           np.full( (one_station_segments.shape[0] , 1),station), \
                           one_station_segments[:, column_index['index']] ])
    ret_columns = ['dx', 'dy', 'z_p', 'z_c', 'y_p', 'y_c', 'dz', 'z', 'from_ind', 'to_ind', 'track', 'station','index']

    ret_columns = dict(zip(ret_columns, list(range(0, len(ret_columns)))))

    # saving this information in the dataframe for later use
    index_true_superedge = one_station_segments[
        np.logical_and( one_station_segments[:, column_index['track' + suffix_p] ] == one_station_segments[:,column_index['track' + suffix_c]] , \
        one_station_segments[:,column_index['track' + suffix_p] ]!= -1) ]
   
    ret_[ np.isin(  ret_[:,ret_columns['index'] ] , index_true_superedge[:,column_index['index'] ] ), ret_columns['track'] ] = index_true_superedge[:,
                                                                       column_index['track' + suffix_p]]

    return ret_,ret_columns


def apply_nodes_restrictions(nodes,
                             restrictions_0, restrictions_1,indices):
                             
    mask1 = nodes[:, indices['dy']] > restrictions_0[0]
    mask2 = nodes[:, indices['dy']] < restrictions_0[1]
    mask3 = nodes[:, indices['dz']] > restrictions_1[0]
    mask4 = nodes[:, indices['dz']] < restrictions_1[1]
    
    return nodes[ mask1 & mask2 & mask3 & mask4 ]
   
@gin.configurable(denylist=['segments', 'restrictions_func'])
def get_pd_line_graph(segments,
                      restrictions_func,
                      restrictions_0, restrictions_1,
                      suffix_p, suffix_c, spec_kwargs):
    nodes = pd.DataFrame()
    edges = pd.DataFrame()

    by_stations = [df for (ind, df) in segments.groupby('station' + suffix_p)]
    # iterate each pair of stations
    # for each 'i' we will have a 2 dataframes of segments which go
    # from i-1'th to i'th stations and from i'th to i+1'th stations
    for i in range(1, len(by_stations)):
        # take segments (which will be nodes as a result) which go
        # from station i-1 to i AND from i to i+1
        supernodes_from = get_supernodes_df(by_stations[i - 1], **spec_kwargs, station=i - 1, STATION_COUNT=3, pi_fix=True)
        supernodes_to = get_supernodes_df(by_stations[i], **spec_kwargs, station=i, STATION_COUNT=3, pi_fix=True)

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
        compute_is_true_track: bool = True,
        save_index: bool = True
) -> pd.DataFrame:
    if suffixes is None:
        suffixes = ['_prev', '_current']
    assert single_event_df.event.nunique() == 1
    my_df = single_event_df.copy()
    if save_index:
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

@gin.configurable()
def get_plain_graph(
        single_event_df: pd.DataFrame,
        restrictions_0 = None,
        restrictions_1 = None,
        suffixes=None,
        compute_is_true_track: bool = True,
        save_index: bool = True
) -> pd.DataFrame:
    
    if suffixes is None:
        suffixes = ['_prev', '_current']

   
    assert single_event_df.event.nunique() == 1
    
    my_df =  np.c_[single_event_df.to_numpy(), single_event_df.index]
    column_index = dict(zip(single_event_df.columns, list(range(0, len(single_event_df.columns)))))
    column_index['index_old'] = my_df.shape[1] - 1
    
    stations = np.unique(my_df[:, column_index['station' ]])
   
    by_stations = []
    
    for station in stations:
        station_hits = my_df[my_df[:, column_index['station']] == station]
        
        by_stations.append( station_hits )
    
    indices_merged = {**{k + suffixes[0]: v for k, v in column_index.items()},
                      **{k + suffixes[1]: v + len(column_index) for k, v in column_index.items()}}


    indices_merged['dy'] = len( indices_merged )
    indices_merged['dz'] = len( indices_merged )

    full_size = 10000
    full = np.empty(shape=(full_size, len(indices_merged)))

    last_row_nodes = 0
    ev_ind = column_index['event']

    for i in range(1, len(by_stations)):

        a,b = by_stations[i - 1], by_stations[i]

        to_merge = np.empty(shape=(b.shape[0] * a.shape[0], 2), dtype=np.int32)
        to_merge[:, 0] = np.tile(b[:, ev_ind], (1, a.shape[0]))
        for i in range(a.shape[0]):
            to_merge[i * b.shape[0]:(i + 1) * b.shape[0], 1] = a[i, ev_ind]

        merge_ind = [(ind // b.shape[0], ind % b.shape[0]) for ind in np.where(to_merge[:, 0] == to_merge[:, 1])[0]]

        line_graph_edges = np.empty(shape=(len(merge_ind), a.shape[1] + b.shape[1] + 2))
        for k, (i, j) in enumerate(merge_ind):
            line_graph_edges[k, :-2] = np.hstack((a[i, :], b[j, :]))


        line_graph_edges[:,indices_merged['dz'] ] = line_graph_edges[:,indices_merged['z' + suffixes[1]]] - line_graph_edges[:,indices_merged['z' + suffixes[0]]]
        line_graph_edges[:,indices_merged['dy'] ] = calc_dphi(line_graph_edges[:,indices_merged['phi' + suffixes[0]]],
                                            line_graph_edges[:,indices_merged['phi' + suffixes[1]]])

        full[last_row_nodes: last_row_nodes + line_graph_edges.shape[0], :] = line_graph_edges

        last_row_nodes = last_row_nodes + line_graph_edges.shape[0]
     
    full = apply_nodes_restrictions(full[ 0:last_row_nodes,:], restrictions_0, restrictions_1,indices_merged)
    
    return single_event_df, pd.DataFrame( full,columns = indices_merged )

def apply_edge_restriction(pd_edges_df: pd.DataFrame,
                           edge_restriction: float):
    assert 'weight' in pd_edges_df
    return pd_edges_df[pd_edges_df.weight < edge_restriction]



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
