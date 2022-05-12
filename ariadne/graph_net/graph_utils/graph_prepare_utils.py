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
def apply_weight_restriction(edges,indices):
    return edges[ edges[:,indices['weight'] ]<  0.01]
def get_edges_from_supernodes_new( edges,indices,device='cpu'):

    node_from_edges = edges[:,[ indices['index_old_p'],indices['index_old_c'] ] ]

    a = edges[:, indices['index_old_c']]#.to(torch.int16)
    b = edges[:, indices['index_old_p']]

    res = a[:, None] == b

    inds = torch.nonzero(res)

    #indices_merged = {**{k + suffixes[0]: v for k, v in column_index.items()},
    #                 **{k + suffixes[1]: v + len(column_index) for k, v in column_index.items()}}

    #indices_merged['dy'] = len(indices_merged)
    #indices_merged['dz'] = len(indices_merged)
    columns = ['weight', 'is_true', 'index_old_c', 'index_old_p', 'from_ind', 'cur_ind', 'to_ind']
    indices_new = dict(zip(columns, list(range(0, len(columns)))))

    col_num = len(columns)

    superedges = torch.empty( ( inds.shape[0] ,col_num), dtype=torch.float64,device=device)

    first_edges = edges[ inds[:,0], : ]
    second_edges = edges[ inds[:,1], : ]

    diff_indices = torch.tensor([ indices[key] for key in [ 'dx','dy','dz' ] ])

    ba = first_edges [:, diff_indices ]
    cb = second_edges[:, diff_indices ]

    superedges[:, indices_new['weight']] = torch.norm( ba-cb )

    superedges[:, indices_new['index_old_c']] = first_edges[:,indices['index']]
    superedges[:, indices_new['index_old_p']] = second_edges[:,indices['index']]

    superedges[:, indices_new['from_ind']] = first_edges[:,indices['index_old_p'] ]
    superedges[:, indices_new['cur_ind']] = first_edges[:, indices['index_old_c'] ]
    superedges[:, indices_new['to_ind']] = second_edges[:, indices['index_old_c']]

    superedges[:, indices_new['is_true']] = first_edges[:,indices['is_true'] ].bool() & second_edges[:,indices['is_true'] ].bool()
    superedges[:, indices_new['weight']] = torch.norm(ba - cb,dim=1)
    #superedges_df = pd.DataFrame(superedges[ superedges[:,indices_new['is_true']].bool() ].numpy() ,columns=indices_new)
    return apply_weight_restriction(superedges,indices_new),indices_new

def apply_nodes_restrictions(nodes,
                             restrictions_0, restrictions_1,indices):
                             
    mask1 = nodes[:, indices['dy']] > restrictions_0[0]
    mask2 = nodes[:, indices['dy']] < restrictions_0[1]
    mask3 = nodes[:, indices['dz']] > restrictions_1[0]
    mask4 = nodes[:, indices['dz']] < restrictions_1[1]
    
    return nodes[ mask1 & mask2 & mask3 & mask4 ]

import  torch

@gin.configurable()
def apply_edge_rescriction_new(merged: torch.Tensor,
        indices_merged,
        device='cpu',
        restrictions_0=None,
        restrictions_1=None,
        suffixes=None,
        compute_is_true_track: bool = True,
        save_index: bool = True):
    '''
    merged_pd = pd.DataFrame(merged, columns=indices_merged)

    fake_num_old = merged_pd[ (merged_pd['track_p'] == torch.tensor([-1], dtype=torch.float64) ) & (merged_pd['track_c'] == torch.tensor([-1], dtype=torch.float64) ) ].shape[0]
    fake_num_old +=merged_pd[ (merged_pd['track_p'] == torch.tensor([-1], dtype=torch.float64) ) & (merged_pd['track_c'] != torch.tensor([-1], dtype=torch.float64) ) ].shape[0]
    fake_num_old += merged_pd[(merged_pd['track_p'] != torch.tensor([-1], dtype=torch.float64)) & (  merged_pd['track_c'] == torch.tensor([-1], dtype=torch.float64))].shape[0]

    true_edges = merged[(merged[:, indices_merged['track_p']] == merged[:, indices_merged['track_c']])]
    true_edges = true_edges[true_edges[:, indices_merged['track_c']] != torch.tensor([-1], dtype=torch.float64)]
    '''
    filtered = merged[(merged[:, indices_merged['dy']] > restrictions_0[0]) & (merged[:, indices_merged['dy']] < restrictions_0[1]) & \
                      (merged[:, indices_merged['dz']] > restrictions_1[0]) & (merged[:, indices_merged['dz']] < restrictions_1[1])]

    return filtered

    filtered_pd = pd.DataFrame(filtered, columns=indices_merged)

    true_filtered = filtered[(filtered[:, indices_merged['track_p']] == filtered[:, indices_merged['track_c']])]
    true_filtered = true_filtered[
        true_filtered[:, indices_merged['track_c']] != torch.tensor([-1], dtype=torch.float64)]



    ratio = true_filtered.shape[0] / true_edges.shape[0]

    print(ratio)


    fake_num_new = filtered_pd[(filtered_pd['track_p'] == torch.tensor([-1], dtype=torch.float64)) & (
                merged_pd['track_c'] == torch.tensor([-1], dtype=torch.float64))].shape[0]
    fake_num_new += filtered_pd[(filtered_pd['track_p'] == torch.tensor([-1], dtype=torch.float64)) & (
                filtered_pd['track_c'] != torch.tensor([-1], dtype=torch.float64))].shape[0]
    fake_num_new += filtered_pd[(filtered_pd['track_p'] != torch.tensor([-1], dtype=torch.float64)) & (
                filtered_pd['track_c'] == torch.tensor([-1], dtype=torch.float64))].shape[0]

    print(fake_num_new/fake_num_old)

    with open('fake_ratio','a') as out:
        out.write(str(fake_num_new/fake_num_old) + ',\n')
    return filtered


@gin.configurable()
def get_plain_graph_old(
        event_data: torch.Tensor,
        column_index,
        device='cpu',
        restrictions_0=None,
        restrictions_1=None,
        suffixes=None,
        compute_is_true_track: bool = True,
        save_index: bool = True
) -> pd.DataFrame:
    if suffixes is None:
        suffixes = ['_prev', '_current']

    STATION_COUNT = 35

    col_num = event_data.shape[1]

    idx = torch.tensor(column_index['station']).to(device)

    a = event_data[:, idx]#.to(torch.int16)

    res = a[:, None] == a - 1

    inds = torch.nonzero(res)

    indices_merged = {**{k + suffixes[0]: v for k, v in column_index.items()},
                      **{k + suffixes[1]: v + len(column_index) for k, v in column_index.items()}}

    extra_columns = ['dx','dy','dz','is_true','index']
    indices_merged.update( zip(extra_columns,list(range(len(indices_merged) ,len(indices_merged) + len(extra_columns))) ) )

    merged = torch.empty(inds.shape[0], col_num * 2 + len(extra_columns), dtype=torch.float64,device=device)

    merged[:, 0:col_num], merged[:, col_num:2 * col_num] = event_data[inds[:, 0], :], event_data[inds[:, 1], :]

    merged[:, indices_merged['dy'] ] = calc_dphi(merged[:, indices_merged['y'+suffixes[0]] ] , merged[:, indices_merged['y'+suffixes[1]]] )
    merged[:, indices_merged['dz'] ] = merged[:,indices_merged['z'+ suffixes[1]]] - merged[:, indices_merged['z' + suffixes[0]]]
    merged[:, indices_merged['dx'] ] = merged[:,indices_merged['x'+ suffixes[1]]] - merged[:, indices_merged['x' + suffixes[0]]]

    #if compute_is_true_track:
    merged[:, indices_merged['is_true'] ] = ( merged[:, indices_merged['track'+suffixes[0] ]]==merged[:, indices_merged[ 'track'+suffixes[1] ]] ) & \
                                               (merged[:, indices_merged['track' + suffixes[1]]] != torch.tensor([-1.]) )

    #print( merged[ merged[:,indices_merged['is_true']] == True ][:,indices_merged['dz'] ].min() )

    merged[:, indices_merged['index']  ] = torch.arange( merged.shape[0] )

    return event_data, merged, column_index,indices_merged



@gin.configurable()
def get_plain_graph_new(
        event_data: torch.Tensor,
        column_index,
        device='cpu',
        restrictions_0=None,
        restrictions_1=None,
        suffixes=None,
        compute_is_true_track: bool = True,
        save_index: bool = True
) -> pd.DataFrame:
    if suffixes is None:
        suffixes = ['_prev', '_current']

    STATION_COUNT = 35
    #event_data = torch.from_numpy(  np.c_[single_event_df.to_numpy(), single_event_df.index])
    #column_index = dict(zip(single_event_df.columns, list(range(0, len(single_event_df.columns)))))

    #column_index['index_old'] = event_data.shape[1] - 1

    #df = pd.DataFrame(event_data.numpy(), columns=column_index)

    col_num = event_data.shape[1]

    idx = torch.tensor(column_index['station']).to(device)

    a = event_data[:, idx]#.to(torch.int16)

    res = a[:, None] == a - 1

    inds = torch.nonzero(res)

    indices_merged = {**{k + suffixes[0]: v for k, v in column_index.items()},
                      **{k + suffixes[1]: v + len(column_index) for k, v in column_index.items()}}

    extra_columns = ['dx','dy','dz','z','is_true','index']
    indices_merged.update( zip(extra_columns,list(range(len(indices_merged) ,len(indices_merged) + len(extra_columns))) ) )

    merged = torch.empty(inds.shape[0], col_num * 2 + len(extra_columns), dtype=torch.float64,device=device)

    merged[:, 0:col_num], merged[:, col_num:2 * col_num] = event_data[inds[:, 0], :], event_data[inds[:, 1], :]

    merged[:, indices_merged['dy'] ] = calc_dphi(merged[:, indices_merged['y'+suffixes[0]] ] , merged[:, indices_merged['y'+suffixes[1]]] )
    merged[:, indices_merged['dz'] ] = merged[:,indices_merged['z'+ suffixes[1]]] - merged[:, indices_merged['z' + suffixes[0]]]
    merged[:, indices_merged['dx'] ] = merged[:,indices_merged['x'+ suffixes[1]]] - merged[:, indices_merged['x' + suffixes[0]]]

    merged[:, indices_merged['z'] ] = ( merged[:, indices_merged['station'+suffixes[0] ] ]  +1 ) / STATION_COUNT

    #if compute_is_true_track:
    merged[:, indices_merged['is_true'] ] = ( merged[:, indices_merged['track'+suffixes[0] ]]==merged[:, indices_merged[ 'track'+suffixes[1] ]] ) & \
                                               (merged[:, indices_merged['track' + suffixes[1]]] != -1. ) #torch.tensor([-1.])!!!

    #print( merged[ merged[:,indices_merged['is_true']] == True ][:,indices_merged['dz'] ].min() )

    #df = pd.DataFrame(merged.numpy(),columns=indices_merged)

    merged[:, indices_merged['index']  ] = torch.arange( merged.shape[0] )

    return event_data, merged, column_index,indices_merged

import collections
graph = collections.namedtuple('Graph', ['X', 'Ri', 'Ro', 'y','edges','index'])

def construct_output_graph_new(nodes,
                           edges,
                           feature_indices,
                           feature_scale:float,
                           index_nodes: int,
                           is_true_index:int,
                           index_prev: int,
                           index_current: int):

    n_edges = edges.shape[0]
    X = (nodes[:,feature_indices] / feature_scale)
    y = edges[:,is_true_index].bool()#torch.zeros(n_edges, dtype=torch.float32)

    node_idx = nodes[:,index_nodes]

    Ri = (node_idx[..., None] == edges[:, index_current]) #.long()

    Ro = (node_idx[..., None] == edges[:, index_prev]) #.long()

    idx = torch.tensor([index_prev,index_current])
    return graph(X, Ri, Ro, y,edges[:,idx], node_idx )#,nodes

#################################################################################################################################
@gin.configurable()
def get_plain_graph(
        single_event_df: pd.DataFrame,
        restrictions_0=None,
        restrictions_1=None,
        suffixes=None,
        compute_is_true_track: bool = True,
        save_index: bool = True
) -> pd.DataFrame:
    if suffixes is None:
        suffixes = ['_prev', '_current']

    my_df = np.c_[single_event_df.to_numpy(), single_event_df.index]
    column_index = dict(zip(single_event_df.columns, list(range(0, len(single_event_df.columns)))))
    column_index['index_old'] = my_df.shape[1] - 1

    stations = np.unique(my_df[:, column_index['station']])

    by_stations = []

    for station in stations:
        station_hits = my_df[my_df[:, column_index['station']] == station]

        by_stations.append(station_hits)

    indices_merged = {**{k + suffixes[0]: v for k, v in column_index.items()},
                      **{k + suffixes[1]: v + len(column_index) for k, v in column_index.items()}}

    indices_merged['dy'] = len(indices_merged)
    indices_merged['dz'] = len(indices_merged)

    full_size = 10000
    full = np.empty(shape=(full_size, len(indices_merged)))

    last_row_nodes = 0
    ev_ind = column_index['event']

    for i in range(1, len(by_stations)):

        a, b = by_stations[i - 1], by_stations[i]

        to_merge = np.empty(shape=(b.shape[0] * a.shape[0], 2), dtype=np.int32)
        to_merge[:, 0] = np.tile(b[:, ev_ind], (1, a.shape[0]))
        for i in range(a.shape[0]):
            to_merge[i * b.shape[0]:(i + 1) * b.shape[0], 1] = a[i, ev_ind]

        merge_ind = [(ind // b.shape[0], ind % b.shape[0]) for ind in np.where(to_merge[:, 0] == to_merge[:, 1])[0]]

        line_graph_edges = np.empty(shape=(len(merge_ind), a.shape[1] + b.shape[1] + 2))
        for k, (i, j) in enumerate(merge_ind):
            line_graph_edges[k, :-2] = np.hstack((a[i, :], b[j, :]))

        line_graph_edges[:, indices_merged['dz']] = line_graph_edges[:,
                                                    indices_merged['z' + suffixes[1]]] - line_graph_edges[:,
                                                                                         indices_merged[
                                                                                             'z' + suffixes[0]]]
        line_graph_edges[:, indices_merged['dy']] = calc_dphi(line_graph_edges[:, indices_merged['phi' + suffixes[0]]],
                                                              line_graph_edges[:, indices_merged['phi' + suffixes[1]]])

        full[last_row_nodes: last_row_nodes + line_graph_edges.shape[0], :] = line_graph_edges

        last_row_nodes = last_row_nodes + line_graph_edges.shape[0]

    full = apply_nodes_restrictions(full[0:last_row_nodes, :], restrictions_0, restrictions_1, indices_merged)

    return single_event_df, pd.DataFrame(full, columns=indices_merged)



def get_supernodes_df(one_station_segments, column_index,
                      axes,
                      suffix_p, suffix_c,
                      station=-1,
                      STATION_COUNT=0,
                      pi_fix=False, ):
    x0_y0_z0_ = one_station_segments[:,
                [column_index[axes[0] + suffix_p], column_index[axes[1] + suffix_p], column_index[axes[2] + suffix_p]]]
    x1_y1_z1_ = one_station_segments[:,
                [column_index[axes[0] + suffix_c], column_index[axes[1] + suffix_c], column_index[axes[2] + suffix_c]]]

    dx_dy_dz = x1_y1_z1_ - x0_y0_z0_

    # if we are in the cylindrical coordinates, we need to fix the radian angle values overflows

    if pi_fix:
        dy_fixed = calc_dphi(one_station_segments[:, column_index[axes[1] + suffix_p]],
                             one_station_segments[:, column_index[axes[1] + suffix_c]])

    ret_ = np.column_stack([dx_dy_dz[:, 0], dy_fixed, one_station_segments[:, column_index[axes[2] + suffix_p]] \
                               , one_station_segments[:, column_index[axes[2] + suffix_c]], \
                            one_station_segments[:, column_index[axes[1] + suffix_p]], \
                            one_station_segments[:, column_index[axes[1] + suffix_c]], \
                            dx_dy_dz[:, 2], np.full((one_station_segments.shape[0], 1), (station + 1) / STATION_COUNT), \
                            one_station_segments[:, column_index['index_old' + suffix_p]], \
                            one_station_segments[:, column_index['index_old' + suffix_c]], \
                            np.full((one_station_segments.shape[0], 1), -1), \
                            np.full((one_station_segments.shape[0], 1), station), \
                            one_station_segments[:, column_index['index']]])
    ret_columns = ['dx', 'dy', 'z_p', 'z_c', 'y_p', 'y_c', 'dz', 'z', 'from_ind', 'to_ind', 'track', 'station', 'index']

    ret_columns = dict(zip(ret_columns, list(range(0, len(ret_columns)))))

    # saving this information in the dataframe for later use
    index_true_superedge = one_station_segments[
        np.logical_and(one_station_segments[:, column_index['track' + suffix_p]] == one_station_segments[:,
                                                                                    column_index['track' + suffix_c]], \
                       one_station_segments[:, column_index['track' + suffix_p]] != -1)]

    ret_[np.isin(ret_[:, ret_columns['index']], index_true_superedge[:, column_index['index']]), ret_columns[
        'track']] = index_true_superedge[:,
                    column_index['track' + suffix_p]]

    return ret_, ret_columns


def get_edges_from_supernodes(a, b, indices):
    indices_prev = indices.copy()
    indices_next = indices.copy()

    indices_prev['cur_ind'] = indices_prev.pop('to_ind')
    indices_next['cur_ind'] = indices_next.pop('from_ind')

    indices_prev['edge_index'] = indices_prev.pop('index')
    indices_next['edge_index'] = indices_next.pop('index')

    cur_ind_a = indices_prev['cur_ind']
    cur_ind_b = indices_next['cur_ind']

    to_merge = np.empty(shape=(b.shape[0] * a.shape[0], 2), dtype=np.int32)
    to_merge[:, 0] = np.tile(b[:, cur_ind_b], (1, a.shape[0]))

    for i in range(a.shape[0]):
        to_merge[i * b.shape[0]:(i + 1) * b.shape[0], 1] = a[i, cur_ind_a]

    merge_ind = [(ind // b.shape[0], ind % b.shape[0]) for ind in np.where(to_merge[:, 0] == to_merge[:, 1])[0]]

    line_graph_edges = np.empty(shape=(len(merge_ind), a.shape[1] + b.shape[1] + 2))
    for k, (i, j) in enumerate(merge_ind):
        line_graph_edges[k, :-2] = np.hstack((a[i, :], b[j, :]))
    # merge segments by their common hit in the middle

    indices_merged = {**{k + '_p': v for k, v in indices_prev.items()},
                      **{k + '_c': v + len(indices_prev) for k, v in indices_next.items()}}

    indices_merged['from_ind'] = indices_merged.pop('from_ind_p')
    indices_merged['to_ind'] = indices_merged.pop('to_ind_c')
    indices_merged['cur_ind'] = indices_merged.pop('cur_ind_c')

    line_graph_edges[:, -2] = -1
    indices_merged['true_superedge'] = line_graph_edges.shape[1] - 2
    indices_merged['weight'] = line_graph_edges.shape[1] - 1

    # track the 'true' superedge. true_superedge is a combination of 2 consecutive true edges of 1 track
    index_true_superedge = (line_graph_edges[:, indices_merged['track_p']] == line_graph_edges[:,
                                                                              indices_merged['track_c']]) & \
                           (line_graph_edges[:, indices_merged['track_p']] != -1)

    line_graph_edges[index_true_superedge, indices_merged['true_superedge']] = line_graph_edges[
        index_true_superedge, indices_merged['track_p']]

    ba = line_graph_edges[:, [indices_merged['dx_p'], indices_merged['dy_p'], indices_merged['dz_p']]]
    cb = line_graph_edges[:, [indices_merged['dx_c'], indices_merged['dy_c'], indices_merged['dz_c']]]

    w = np.linalg.norm((ba - cb).astype(np.float64), axis=1)

    line_graph_edges[:, indices_merged['weight']] = w
    indexation = ['weight', 'true_superedge', 'edge_index_p', 'edge_index_c', 'from_ind', 'cur_ind', 'to_ind']
    res = line_graph_edges[:, [indices_merged[ind] for ind in indexation]]
    return line_graph_edges[:, [indices_merged[ind] for ind in indexation]], indexation


