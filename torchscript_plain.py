import torch
import math

import collections

graph = collections.namedtuple('Graph', ['X', 'Ri', 'Ro', 'y','edges','index'])
from typing import Dict, List, Tuple

def construct_output_graph(hits,
                           edges,
                           feature_indices,
                           feature_scale:float,
                           index_prev: int,
                           index_current: int):

    n_edges = edges.shape[0]
    X = (hits[:,feature_indices] / feature_scale)
    y = torch.zeros(n_edges, dtype=torch.float32)

    hit_idx = hits[:,-1]

    Ri = (hit_idx[..., None] == edges[:, index_current]) #.long()

    Ro = (hit_idx[..., None] == edges[:, index_prev]) #.long()

    idx = torch.tensor([index_prev,index_current], device=device)
    return graph(X, Ri, Ro, y,edges[:,idx], hits[:,-1] )


class ProcessSingleEvent(torch.nn.Module):

    def __init__(self,device,node_indices:Dict, edge_indices:Dict):
        super(ProcessSingleEvent, self).__init__()
        self.device = torch.device(device)

        self.edge_indices = edge_indices
        self.node_indices = node_indices

        self.restrictions_0= torch.tensor( [-0.005, 0.005],device=self.device)#.to(self.device )
        self.restrictions_1= torch.tensor( [-0.05, 0.05],device=self.device)#.to(self.device )
        #self.restrictions_0 = torch.tensor([-100, 100], device=self.device)  # .to(self.device )
        #self.restrictions_1 = torch.tensor([-100, 100], device=self.device)  # .to(self.device )

        self.constraints=torch.tensor( [[269., 581.], [-3.15, 3.15],[-2386.0, 2386.0]] ,device=self.device)#.to(self.device)

    def calc_dphi(self,phi1, phi2):

        pi = math.pi
        dphi = (phi2 - phi1)
        dphi[(dphi > pi)] -= 2 * pi
        dphi[(dphi < -pi)] += 2 * pi
        return dphi

    def convert(self,data):

        r = torch.sqrt(data[:, 1] ** 2 + data[:, 2] ** 2)

        phi = torch.atan2(data[:, 1], data[:, 2])

        z = data[:, 3]

        data[:, 1] = r
        data[:, 2] = phi
        data[:, 3] = z

    def normalize(self,data, indices: List[int], mode: str):

        if mode == 'node':

            data[:, indices] = 2 * (data[:, indices] - self.constraints[:, 0]) / ( self.constraints[:, 1] - self.constraints[:, 0]) - 1  #

        elif mode == 'edge':

            data[:, indices] = 2 * data[:, indices] / (self.constraints[:, 1] - self.constraints[:, 0])  #

    def apply_edge_restriction(self,merged):

        return merged[(merged[:, self.edge_indices['dy']] > self.restrictions_0[0]) & (merged[:, self.edge_indices['dy']] < self.restrictions_0[1]) & \
                      (merged[:, self.edge_indices['dz']] > self.restrictions_1[0]) & (merged[:, self.edge_indices['dz']] < self.restrictions_1[1])]

    def get_graph(self,event_data):

        suffixes = ['_p', '_c']

        STATION_COUNT = 35

        col_num = event_data.shape[1]

        a = event_data[:, self.node_indices['station'] ]  # .to(torch.int16)

        res = a[:, None] == a - 1

        inds = torch.nonzero(res)

        merged = torch.empty(inds.shape[0], len(self.edge_indices), dtype=torch.float64, device=self.device) #

        merged[:, 0:col_num], merged[:, col_num:2 * col_num] = event_data[inds[:, 0], :], event_data[inds[:, 1], :]

        merged[:, self.edge_indices['dy']] = self.calc_dphi(merged[:, self.edge_indices['y' + suffixes[0]]],
                                                    merged[:, self.edge_indices['y' + suffixes[1]]])

        merged[:, self.edge_indices['dz']] = merged[:, self.edge_indices['z' + suffixes[1]]] - merged[:, self.edge_indices[
                                                                                                       'z' + suffixes[
                                                                                                           0]]]
        merged[:, self.edge_indices['dx']] = merged[:, self.edge_indices['x' + suffixes[1]]] - merged[:, self.edge_indices[
                                                                                                       'x' + suffixes[
                                                                                                           0]]]

        merged[:, self.edge_indices['z']] = (merged[:, self.edge_indices['station' + suffixes[0]]] + 1) / STATION_COUNT
        return merged
    def construct_output_graph(self,nodes,edges,
                           feature_indices,
                           feature_scale:float,
                           index_nodes: int,
                           is_true_index:int,
                           index_prev: int,
                           index_current: int):
        X = (nodes[:, feature_indices] / feature_scale)
        y = edges[:, is_true_index]

        node_idx = nodes[:, index_nodes]

        Ri = (node_idx[..., None] == edges[:, index_current])  # .long()

        Ro = (node_idx[..., None] == edges[:, index_prev])  # .long()

        idx = torch.tensor([index_prev, index_current])
        return graph(X, Ri, Ro, y, edges[:, idx], node_idx)  # ,nodes

    def forward(self,event_data):

        self.convert(event_data)
        nodes = event_data
        edges = self.get_graph(nodes)

        node_index = self.node_indices['index_old']

        self.normalize(nodes,[self.node_indices[x] for x in ['x', 'y', 'z'] ],mode = 'node' )

        self.normalize(edges,[self.edge_indices['x_p'], self.edge_indices['y_p'], self.edge_indices['z_p']],  mode='node')
        self.normalize(edges,[self.edge_indices['x_c'], self.edge_indices['y_c'], self.edge_indices['z_c']],  mode='node')

        self.normalize(edges,[self.edge_indices['dx'], self.edge_indices['dy'], self.edge_indices['dz'] ],mode = 'edge')

        edges = self.apply_edge_restriction(edges)


        graph_with_inds = self.construct_output_graph(nodes, edges, \
                                                         torch.tensor( [self.node_indices[x] for x in ['x', 'y', 'z'] ]),\
                                                         1., node_index, self.edge_indices['is_true'],
                                                         self.edge_indices['index_old_p'], self.edge_indices['index_old_c'])

        return graph_with_inds
'''
import pandas as pd
df = pd.read_csv("D:/ariadne_clone/ariadne/input_data/fakes/input_1000_fake_0.tsv",delimiter='\t',names= ['event',  'x', 'y', 'z', 'station',
                              'track', 'px', 'py', 'pz', 'X0', 'Y0', 'Z0'] )
event_df = df[df['event']==0]

event_df['index_old'] = event_df.index
event_df = event_df[['event', 'x', 'y', 'z', 'station', 'track', 'index_old']]
column_index = dict(zip(event_df.columns, list(range(0, len(event_df.columns)))))

suffixes = ['_p', '_c']

indices_merged = {**{k + suffixes[0]: v for k, v in column_index.items()},
                          **{k + suffixes[1]: v + len(column_index) for k, v in column_index.items()}}
extra_columns = ['dx', 'dy', 'dz', 'z', 'is_true', 'index']
indices_merged.update(
      zip(extra_columns, list(range(len(indices_merged), len(indices_merged) + len(extra_columns)))))

event_data = torch.from_numpy(event_df.to_numpy()).to(device)
by_event = []

for n in range(0,nevents,events_per_iter):
    ev = torch.empty( (0,data.shape[1]),device=device )
    for k in range( n,n+events_per_iter ):
        ev = torch.cat( (ev,  data[data[:,0] == k] ) )

    by_event.append( ev )
obj = ProcessSingleEvent(device,column_index,indices_merged)

traced = torch.jit.script(obj, (event_data ) )

total_time = 0
start = time.perf_counter()
res = traced( event_data )
end = time.perf_counter()

print(end-start)
torch.jit.save(traced, 'E:/plain.pt')
'''
