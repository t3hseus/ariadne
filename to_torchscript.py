import torch
import math
from typing import List

def apply_nodes_restrictions(nodes,
                             restrictions_0, restrictions_1):

    mask1 = nodes[:, -2 ] > restrictions_0[0]
    mask2 = nodes[:, -2 ] < restrictions_0[1]
    mask3 = nodes[:, -1 ] > restrictions_1[0]
    mask4 = nodes[:, -1 ] < restrictions_1[1]
    res =  nodes[mask1 & mask2 & mask3 & mask4]
    return res

def convert(data):

   r = torch.sqrt(data[:,1] ** 2 + data[:,2] ** 2)

   phi = torch.atan2(data[:,1], data[:,2])

   z = data[:,3]

   data[:, 1] = r
   data[:, 2] = phi
   data[:, 3] = z

def normalize(data, constraints):

    x_min, x_max = constraints[0,0],constraints[0,1]
    y_min, y_max = constraints[1,0],constraints[1,1]
    z_min, z_max = constraints[2,0],constraints[2,1]

    x_norm = 2 * (data[:,1] - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (data[:,2] - y_min) / (y_max - y_min) - 1
    z_norm = 2 * (data[:,3] - z_min) / (z_max - z_min) - 1

    data[:,1] = x_norm
    data[:,2] = y_norm
    data[:,3] = z_norm

def calc_dphi(phi1, phi2):

    pi = math.pi
    dphi = (phi2 - phi1 )
    dphi[ (dphi > pi) ] -= 2 * pi
    dphi[ (dphi < -pi) ] += 2 * pi
    return dphi

import collections

graph = collections.namedtuple('Graph', ['X', 'Ri', 'Ro', 'y','edges'])

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

    Ri = (hit_idx[..., None] == edges[:, index_current]) .long()

    Ro = (hit_idx[..., None] == edges[:, index_prev]) .long()

    idx = torch.tensor([index_prev,index_current], device='cuda')
    return graph(X, Ri, Ro, y,edges[:,idx] )

class ProcessSingleEvent(torch.nn.Module):

    def __init__(self,device,chunk_size):
        super(ProcessSingleEvent, self).__init__()
        self.device = torch.device(device)
        self.chunk_size = chunk_size

        self.restrictions_0= torch.tensor( [-0.04, 0.04],device=self.device)#.to(self.device )
        self.restrictions_1= torch.tensor( [-0.03, 0.03],device=self.device)#.to(self.device )

        self.constraints=torch.tensor( [[269., 581.], [-3.15, 3.15],[-2386.0, 2386.0]] ,device=self.device)#.to(self.device)

        self.convert= convert
        self.normalize= normalize

    def forward(self,event_data):

        col_num = event_data.shape[1]

        self.convert(event_data)
        self.normalize(event_data, self.constraints)

        idx = torch.tensor([0, 4],device=self.device)

        a = event_data[:,idx].to(torch.int16)

        b = a - torch.cat( [torch.zeros( event_data.shape[0],1,device=self.device,dtype=torch.int16),torch.ones( event_data.shape[0],1,device=self.device,dtype=torch.int16)],dim=1 )

        res =  a[:, None] == b

        res = res[:, :, 0] & res[:, :, 1]

        chunks = torch.split(res, self.chunk_size,dim=0 )

        all_filtered = torch.empty( 0,col_num*2+2,device=self.device )
        all:List[graph] = []
        for chunk_num, chunk in enumerate(chunks):

            if True: # just placeholder for an additional for-loop that was here

                inds = torch.nonzero(chunk)
                inds += torch.cat( [torch.ones(inds.shape[0],1,device=self.device,dtype=torch.long) \
                                    * chunk_num*self.chunk_size, torch.zeros(inds.shape[0],1,device=self.device,dtype=torch.long)],dim=1 )

                merged = torch.empty( inds.shape[0],col_num*2+2 ,dtype=torch.float64,device=self.device )

                merged[:,0:col_num],merged[:,col_num:2*col_num]  = event_data[inds[:, 0], :],event_data[inds[:, 1], :]

                merged[:,-2]= merged[:,2+col_num] - merged[:,2]
                merged[:,-1]= merged[:,3+col_num] - merged[:,3] #


                all_filtered = torch.cat( [all_filtered,apply_nodes_restrictions(merged, self.restrictions_0, self.restrictions_1)],dim=0)

        for ev in torch.unique( event_data[:,0].detach() ):

            output_graph = construct_output_graph(event_data[ event_data[:,0] == ev ], all_filtered[ all_filtered[:,0] == ev ], torch.tensor([1, 2, 3],device=self.device),\
                                                                                                                                             1., col_num - 1, col_num * 2 - 1)
            all.append(output_graph)


        return all

import pandas as pd
import numpy as np

df = pd.read_csv("D:/output_fake_5000.tsv",delimiter='\t',nrows=500000000,names= ['event',  'x', 'y', 'z', 'station',
                              'track', 'px', 'py', 'pz', 'X0', 'Y0', 'Z0'] )
df = df[['event', 'x', 'y', 'z', 'station', 'track']]
df['index'] = df.index

#device = 'cpu'
device = 'cuda'
single_ev_obj = ProcessSingleEvent(device,chunk_size=1000)
data = torch.from_numpy( df.to_numpy() ).to(device)
traced = torch.jit.script(single_ev_obj, (data[0:10000,:] ) )

torch.jit.save(traced, 'D:/scriptmodule_cuda.pt')
