import os
import sys
import gin
import numpy as np
import pandas as pd
import cupy as cp

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

#from IPython.core.display import clear_output, display

import logging

logging.getLogger().setLevel(logging.DEBUG)

from eval.event_evaluation_old import EventEvaluator
from ariadne_v2.transformations import Compose, ConstraintsNormalize, ToCylindrical, DropSpinningTracks, DropShort, \
    DropEmpty,DropFakes

parse_cfg = {
    'csv_params': {
        "sep": '\s+',
        # "nrows": 15000,
        "encoding": 'utf-8',
        "names": ['event', 'x', 'y', 'z', 'station', 'track', 'px', 'py', 'pz', 'X0', 'Y0', 'Z0']
    },

   # 'input_file_mask': "./input_data/fakes/input_1000_fake_0.tsv", #
    'input_file_mask': "./input_data/input_10000_fake_0.tsv",
    'events_quantity': '0..100'
}

global_transformer = Compose([
    #DropSpinningTracks(),
    DropShort(num_stations=35),
    #DropEmpty()
    #DropFakes()
])

import scripts.clean_cache

# to clean cache if needed
# scripts.clean_cache.clean_jit_cache('20d')


from ariadne.graph_net.graph_utils.graph_prepare_utils import get_plain_graph,get_plain_graph_new,apply_edge_rescriction_new, construct_output_graph_new,\
                                                                                                                            get_edges_from_supernodes_new,get_plain_graph_old
from ariadne.transformations import Compose, ConstraintsNormalize, ToCylindrical

from ariadne_v2.inference import IModelLoader
from to_torchscript import normalize,convert
from to_torchscript_old import ProcessSingleEvent_old
from collections import namedtuple
import torch
from to_torchscript import graph as GraphTuple

suff_df = ('_p', '_c')

gin.bind_parameter('normalize.constraints',torch.tensor( [[269., 581.], [-3.15, 3.15],[-2386.0, 2386.0]]))


gin.bind_parameter('apply_edge_rescriction_new.restrictions_0',(-100,100))
gin.bind_parameter('apply_edge_rescriction_new.restrictions_1',(-100,100))

#gin.bind_parameter('apply_edge_rescriction_new.restrictions_0',(-0.005, 0.005))
#gin.bind_parameter('apply_edge_rescriction_new.restrictions_1',(-0.05, 0.05))

#gin.bind_parameter('get_pd_line_graph.suffix_c', '_c')
#gin.bind_parameter('get_pd_line_graph.suffix_p', '_p')
#gin.bind_parameter('get_pd_line_graph.spec_kwargs', {'suffix_c': '_c',
#                                                     'suffix_p': '_p',
#                                                     'axes': ['r', 'phi', 'z']})

class GraphModelLoader(IModelLoader):
    def __call__(self):
        from ariadne.graph_net.model import GraphNet_v1

        gin.bind_parameter('GraphNet_v1.input_dim', 3) #5
        gin.bind_parameter('GraphNet_v1.hidden_dim', 128)
        gin.bind_parameter('GraphNet_v1.n_iters', 1)

        def weights_update_g(model, checkpoint):
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['state_dict']
            real_dict = {}
            for (k, v) in model_dict.items():
                needed_key = None
                for pretr_key in pretrained_dict:
                    if k in pretr_key:
                        needed_key = pretr_key
                        break
                assert needed_key is not None, "key %s not in pretrained_dict %r!" % (k, pretrained_dict.keys())
                real_dict[k] = pretrained_dict[needed_key]

            model.load_state_dict(real_dict)
            model.eval()
            return model

        path_g =  'lightning_logs/GraphNet_v1/no_restriction_fakes/epoch=4-step=9999.ckpt' #_fakes

        #path_g = 'lightning_logs/GraphNet_v1/version_195/epoch=1-step=4999.ckpt'  # _fakes



        checkpoint_g = torch.load(path_g) if torch.cuda.is_available() else torch.load(path_g,
                                                                                       map_location=torch.device('cuda'))
        model_g = weights_update_g(model=GraphNet_v1(),
                                   checkpoint=checkpoint_g).cuda()
        model_hash = {"path_g": path_g, 'gin': gin.config_str(), 'model': '%r' % model_g }#'edge': _edge_restriction}
        return model_hash, model_g


GraphWithIndices = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y','edges','index'])

transformer_g = Compose([
    DropSpinningTracks(),
    DropShort(),
    DropEmpty(),
    ToCylindrical(),
    ConstraintsNormalize(
        columns=('r', 'phi', 'z'),
        constraints={'r': [269., 581.], 'phi': [-3.15, 3.15], 'z': [-2386.0, 2386.0]},
        use_global_constraints=True
    ),
])


def construct_graph_with_indices(graph, v1v2,index):

    return GraphTuple(graph.X, graph.Ri, graph.Ro, graph.y, v1v2, index)
    #return GraphWithIndices(graph.X, graph.Ri, graph.Ro, graph.y, graph.index, v1v2, ev_id )
def construct_graph_with_indices_torchscript(graph):
    return GraphWithIndices( graph.X.cpu().numpy(), graph.Ri.cpu().numpy(), graph.Ro.cpu().numpy(), graph.y.cpu().numpy(),  graph.edges.cpu().numpy()  )
#process_single = torch.jit.load('scriptmodule.pt')


from torchscript_plain import ProcessSingleEvent

loaded = torch.jit.load('E:/plain.pt')
process_single = None
def get_graph_cupy(event):

    event = event[:, [0, 1, 2, 3, 4, 5, 12]]
    return ProcessSingleEvent().forward( cp.asarray(event) )


def get_graph_torchscript(event,node_indices,edge_indices,_):
    global process_single
    if process_single == None:
        process_single = ProcessSingleEvent('cpu', node_indices, edge_indices)

    with torch.no_grad():

        graphs = process_single(event)

    return graphs
def get_graph_torchscript_old(event):

    return ProcessSingleEvent_old('cuda')(event)

def get_graph(event_data,column_index,*_):

    #event = event[['event', 'x', 'y', 'z', 'station', 'track', 'index_old']]

    device = 'cpu'
    #event_data = torch.from_numpy(event.to_numpy() ).to(device)
    #column_index = dict(zip(event.columns, list(range(0, len(event.columns)))))

    convert(event_data)

    '''
    try:
        event = transformer_g(event)
    except AssertionError as err:
        print("ASS error %r" % err)
        return None
    '''

    nodes, edges, node_indices,edge_indices = get_plain_graph_new(event_data, column_index, device=device, suffixes=suff_df, compute_is_true_track=False)

    node_index = node_indices['index_old']

    normalize(nodes,[node_indices[x] for x in ['x', 'y', 'z'] ],mode = 'node' )

    normalize(edges,[edge_indices['x_p'], edge_indices['y_p'], edge_indices['z_p']],  mode='node')
    normalize(edges,[edge_indices['x_c'], edge_indices['y_c'], edge_indices['z_c']],  mode='node')

    normalize(edges,[edge_indices['dx'], edge_indices['dy'], edge_indices['dz'] ],mode = 'edge')

    #df = pd.DataFrame(edges,columns=edge_indices)
    #dz =edges[edges[:, edge_indices['is_true']].bool() == True][:, edge_indices['dz']]
    #dy =edges[edges[:, edge_indices['is_true']].bool() == True][:, edge_indices['dy']]
    #if ( dy.min() < -0.1 )or ( dy.max() > 0.1 ) or ( dz.min() < -0.1 )or ( dz.max() > 0.1 ):
    #print(dy.min(),dy.max(),dz.min(),dz.max() )

    edges = apply_edge_rescriction_new(edges, edge_indices)


    graph_with_inds = construct_output_graph_new(nodes, edges, \
                                                     torch.tensor( [node_indices[x] for x in ['x', 'y', 'z'] ]),\
                                                     1., node_index, edge_indices['is_true'],
                                                     edge_indices['index_old_p'], edge_indices['index_old_c'])

    return graph_with_inds
from ariadne.graph_net.dataset import collate_fn

N_STATIONS = 35

def bfs__(graph, start, end):
    paths = []
    # maintain a queue of paths
    queue = []
    # push the first path into the queue
    queue.append([start])
    while queue:
        # get the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        # path found
        if node == end:
            paths.append ( path )
        # enumerate all adjacent nodes, construct a
        # new path and push it into the queue
        for adjacent in graph.get(node, []):
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)
    return np.array( paths,dtype=np.uint32 )

def dfs(data, path, paths ,ev_id = 0):
    datum = path[-1]
    if datum in data:
        for val in data[datum]:

            new_path = path + [val]

            paths = dfs(data, new_path, paths)
    else:
        paths += [path]

    return paths

import random

from bfs import find_paths

from timeit import default_timer as timer

def eval_event(tgt_graph, model_g,ev_id):

    #tgt_graph =  GraphWithIndices( *map(lambda x:x.get(), tgt_graph ) )
    tgt_graph = tgt_graph

    tgt_graph = GraphWithIndices( *map(lambda x:x.cuda(), tgt_graph ) )
    
    batch_input, batch_target = collate_fn([tgt_graph])
    
    with torch.no_grad():
        res = model_g(batch_input['inputs']).cpu().numpy()
        y_pred = res.flatten() > 0.5

        #y_pred = np.array( [True]*(tgt_graph.Ri.shape[1]) )#
        #y_pred = tgt_graph.y

    adj = torch.t(torch.matmul(tgt_graph.Ri[:,y_pred].float(), torch.t(tgt_graph.Ro[:,y_pred].float())) )

    '''
        tracks_ =[]
        real_index = np.vectorize( lambda x:tgt_graph.index.cpu().numpy()[x] )

        tracks = find_paths(adj )

        for track in tracks:
            tracks_.append( real_index(track.cpu().numpy() ) )
        return pd.DataFrame(tracks_, columns=[f"hit_id_{n}" for n in range(1, N_STATIONS + 1)])
    '''

    power = 34
    adj_to_34 = torch.matrix_power(adj, power)
    track_num = adj_to_34.sum().int()


    v_all = np.empty((track_num, N_STATIONS), dtype=np.int32)


    start_end_idx = torch.nonzero(adj_to_34).cpu().numpy()



    v1v2 = tgt_graph.edges [ y_pred ].cpu().numpy()

    #np.save('saved_graphs/v1v2v3' + str(ev_id), v1v2)

    graph_dict = dict()
    #start = timer()
    hits = np.unique( v1v2 )
    for hit in hits :
        res = v1v2[np.where(hit == v1v2[:, 0])][:, 1]
        if res.shape[0] > 0:
            graph_dict[hit] = res
    #end = timer()


    last_row = 0


    for first,last in start_end_idx  :

        if adj_to_34[first,last] >5:

            continue

        paths = bfs__(graph_dict, tgt_graph.index [ first ].item(), tgt_graph.index [last ].item() )#dfs(graph_dict, [hit],[],ev_id=ev_id)


        v_all[ last_row:last_row+paths.shape[0],: ] = paths

        last_row+=paths.shape[0]

    eval_df = pd.DataFrame(v_all[0:last_row,:], columns=[f"hit_id_{n}" for n in range(1, N_STATIONS + 1)])

    return eval_df

evaluator = EventEvaluator(parse_cfg, global_transformer, N_STATIONS)
events = evaluator.prepare(model_loader=GraphModelLoader()) #[0]
all_results = evaluator.build_all_tracks()
model_results = evaluator.run_model(get_graph_torchscript, eval_event)
results_graphnet = evaluator.solve_results(model_results, all_results)

