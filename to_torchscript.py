import torch
import math
from typing import List
device = 'cpu'
def apply_nodes_restrictions(nodes,
                             restrictions_0, restrictions_1):

    mask1 = nodes[:, -2 ] > restrictions_0[0]
    mask2 = nodes[:, -2 ] < restrictions_0[1]
    mask3 = nodes[:, -1 ] > restrictions_1[0]
    mask4 = nodes[:, -1 ] < restrictions_1[1]
    res =  nodes[mask1 & mask2 & mask3 & mask4]
    return res
import gin


def convert(data):

   r = torch.sqrt(data[:,1] ** 2 + data[:,2] ** 2)

   phi = torch.atan2(data[:,1] , data[:,2])

   z = data[:,3]

   data[:, 1] = r
   data[:, 2] = phi
   data[:, 3] = z
import pandas as pd
@gin.configurable
def normalize(data, indices,constraints=[], mode='node'):

    constraints= constraints#.to('cuda')
    x_min, x_max = constraints[0,0],constraints[0,1]
    y_min, y_max = constraints[1,0],constraints[1,1]
    z_min, z_max = constraints[2,0],constraints[2,1]

    def norm_coords(  ):

        x_norm = 2 * (data[:, 1] - x_min) / (x_max - x_min) - 1
        y_norm = 2 * (data[:, 2] - y_min) / (y_max - y_min) - 1
        z_norm = 2 * (data[:, 3] - z_min) / (z_max - z_min) - 1
        return  torch.stack ( [x_norm,y_norm,z_norm],axis=1)

    '''
    def norm_diff(  ):


        x_norm = 2 * data[:, -3] / (x_max - x_min)
        y_norm = 2 * data[:, -2]  / (y_max - y_min)
        z_norm = 2 * data[:, -1]  / (z_max - z_min)
        return torch.stack([x_norm,y_norm, z_norm], axis=1)
    '''
    if mode == 'node':

        data[:,indices] = 2 * (  data[:,indices] -  constraints[:,0] ) /  ( constraints[:,1]-constraints[:,0]) -1 #

        #data[:,indices] = norm_coords()#torch.stack ( [x_norm,y_norm,z_norm],axis=1)

    elif mode == 'edge':
        #df = pd.DataFrame(data, columns=edge_indices)
        data[:,indices] = 2 * data[:,indices] / ( constraints[:,1]-constraints[:,0])  #
        #df = pd.DataFrame(data,columns=edge_indices)
        #print('feff')
def calc_dphi(phi1, phi2):

    pi = math.pi
    dphi = (phi2 - phi1 )
    dphi[ (dphi > pi) ] -= 2 * pi
    dphi[ (dphi < -pi) ] += 2 * pi
    return dphi

import collections

graph = collections.namedtuple('Graph', ['X', 'Ri', 'Ro', 'y','edges','index'])

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
import time

class ProcessSingleEvent(torch.nn.Module):

    def __init__(self,device,chunk_size=1000):
        super(ProcessSingleEvent, self).__init__()
        self.device = torch.device(device)
        self.chunk_size = chunk_size

        self.restrictions_0= torch.tensor( [-0.01, 0.01],device=self.device)#.to(self.device )
        self.restrictions_1= torch.tensor( [-0.05, 0.05],device=self.device)#.to(self.device )
        #self.restrictions_0 = torch.tensor([-0.004, 0.004], device=self.device)  # .to(self.device )
        #self.restrictions_1 = torch.tensor([-0.03, 0.03], device=self.device)  # .to(self.device )

        self.constraints=torch.tensor( [[269., 581.], [-3.15, 3.15],[-2386.0, 2386.0]] ,device=self.device)#.to(self.device)

        self.convert= convert
        self.normalize= normalize

    def forward(self,event_data):
        '''
        prod = torch.cartesian_prod
        event_data=event_data.to(torch.float16)
        cartesian = torch.cat( [ prod( event_data[:,i],event_data[:,i]) for i in [0,1,2,3,4,-1] ],dim=1 ) #range(event_data.shape[1])

        cartesian = cartesian[ ( cartesian[:,0]==cartesian[:,1] ) & ( cartesian[:,8]==cartesian[:,9]-1 ) ]
        '''
        col_num = event_data.shape[1]

        self.convert(event_data)
        self.normalize(event_data, self.constraints)

        idx = torch.tensor([4], device=self.device)

        a = event_data[:, idx].to(torch.int16)

        res = a[:, None] == a-1

        inds = torch.nonzero(res)

        merged = torch.empty(inds.shape[0], col_num * 2 + 2, dtype=torch.float64, device=self.device)

        merged[:, 0:col_num], merged[:, col_num:2 * col_num] = event_data[inds[:, 0], :], event_data[inds[:, 1], :]



        merged[:, -2] = merged[:, 2 + col_num] - merged[:, 2]
        merged[:, -1] = merged[:, 3 + col_num] - merged[:, 3]  #
        print(merged[:, -1].min(),merged[:, -1].max() )

        filtered = merged[ (merged[:,-2] > self.restrictions_0[0]) & (merged[:,-2] < self.restrictions_0[1]) & \
                         (merged[:,-1] > self.restrictions_1[0]) & (merged[:,-1] < self.restrictions_1[1]) ]



        return [construct_output_graph(event_data, filtered, \
                                                  torch.tensor([1, 2, 3],device=self.device),\
                                                  1., col_num - 1, col_num * 2 - 1)]

'''
import pandas as pd

nevents = 1000
events_per_iter =17
df = pd.read_csv("D:/output_fake_5000.tsv",delimiter='\t',names= ['event',  'x', 'y', 'z', 'station',
                              'track', 'px', 'py', 'pz', 'X0', 'Y0', 'Z0'] )
df = df[['event', 'x', 'y', 'z', 'station', 'track']]
df['index'] = df.index


obj = ProcessSingleEvent(device)
data = torch.from_numpy( df.to_numpy() ).to(device)

by_event = []

for n in range(0,nevents,events_per_iter):
    ev = torch.empty( (0,data.shape[1]),device=device )
    for k in range( n,n+events_per_iter ):
        ev = torch.cat( (ev,  data[data[:,0] == k] ) )

    by_event.append( ev )

total_time = 0
start = time.perf_counter()
for ev in by_event:

    res = obj( ev )
end = time.perf_counter()
print(end-start)

obj = ProcessSingleEvent(device)
traced = torch.jit.script(obj, (data[data[:,0] == 0] ) )

#torch._C._jit_set_bailout_depth(1)
torch.jit.save(traced, 'E:/scriptmodule_cpu.pt')
'''