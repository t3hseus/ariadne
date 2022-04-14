import os
import sys
import gin
import numpy as np
import pandas as pd


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

#from IPython.core.display import clear_output, display

import logging

logging.getLogger().setLevel(logging.DEBUG)

from eval.event_evaluation import EventEvaluator
from ariadne_v2.transformations import Compose, ConstraintsNormalize, ToCylindrical, DropSpinningTracks, DropShort, \
    DropEmpty,DropFakes

parse_cfg = {
    'csv_params': {
        "sep": '\s+',
        # "nrows": 15000,
        "encoding": 'utf-8',
        "names": ['event', 'x', 'y', 'z', 'station', 'track', 'px', 'py', 'pz', 'X0', 'Y0', 'Z0']
    },

    'input_file_mask': "D:/output_fake_5000.tsv", #
    'events_quantity': '0..500'
}

global_transformer = Compose([
    #DropSpinningTracks(),
    #DropShort(num_stations=3),
    #DropEmpty()
    DropFakes()
])

import scripts.clean_cache

# to clean cache if needed
# scripts.clean_cache.clean_jit_cache('20d')


from ariadne.graph_net.graph_utils.graph_prepare_utils import get_plain_graph, construct_output_graph
from ariadne.transformations import Compose, ConstraintsNormalize, ToCylindrical

from ariadne_v2.inference import IModelLoader

suff_df = ('_p', '_c')

gin.bind_parameter('get_plain_graph.restrictions_0',(-0.04, 0.04))
gin.bind_parameter('get_plain_graph.restrictions_1',(-0.03, 0.03))

gin.bind_parameter('get_pd_line_graph.suffix_c', '_c')
gin.bind_parameter('get_pd_line_graph.suffix_p', '_p')
gin.bind_parameter('get_pd_line_graph.spec_kwargs', {'suffix_c': '_c',
                                                     'suffix_p': '_p',
                                                     'axes': ['r', 'phi', 'z']})

class GraphModelLoader(IModelLoader):
    def __call__(self):
        from ariadne.graph_net.model import GraphNet_v1

        gin.bind_parameter('GraphNet_v1.input_dim', 3)
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

        path_g =  'lightning_logs/GraphNet_v1/version_6/epoch=19-step=999.ckpt'

        checkpoint_g = torch.load(path_g) if torch.cuda.is_available() else torch.load(path_g,
                                                                                       map_location=torch.device('cpu'))
        model_g = weights_update_g(model=GraphNet_v1(),
                                   checkpoint=checkpoint_g)
        model_hash = {"path_g": path_g, 'gin': gin.config_str(), 'model': '%r' % model_g }#'edge': _edge_restriction}
        return model_hash, model_g


from collections import namedtuple

GraphWithIndices = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y','edges'])

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
import torch
loaded = torch.jit.load('D:/scriptmodule_cuda.pt')
def construct_graph_with_indices(graph, v1v2, ev_id):

    return GraphWithIndices(graph.X, graph.Ri, graph.Ro, graph.y, v1v2, ev_id)
def construct_graph_with_indices_torchscript(graph):
    return GraphWithIndices( graph.X.cpu().numpy(), graph.Ri.cpu().numpy(), graph.Ro.cpu().numpy(), graph.y.cpu().numpy(),  graph.edges.cpu().numpy()  )
#process_single = torch.jit.load('scriptmodule.pt')

def get_graph_torcscipt(event):

    event = event[:,[0,1,2,3,4,5,12] ]

    with torch.no_grad():
        graphs = loaded( event )

    return graphs
def get_graph(event):

    event = event[['event', 'x', 'y', 'z', 'station', 'track', 'index_old']]

    event['index'] = event.index

    try:
        event = transformer_g(event)
    except AssertionError as err:
        print("ASS error %r" % err)
        return None


    event.index = event['index_old'].values
    event = event[['event', 'r', 'phi', 'z', 'station', 'track']]

    nodes, edges = get_plain_graph(event, suffixes=suff_df, compute_is_true_track=False)
    ev_id = event.event.values[0]


    graph = construct_output_graph(nodes, edges, ['r', 'phi', 'z'],
                                   [ 1., 1., 1.] ,index_label_prev='index_old_p',
                                                  index_label_current='index_old_c')


    graph_with_inds = construct_graph_with_indices(graph, edges[ ['index_old_p','index_old_c'] ].values,
                                                    ev_id)


    return graph_with_inds


from ariadne.graph_net.dataset import collate_fn

N_STATIONS = 35

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
from timeit import default_timer as timer
def eval_event(tgt_graph, model_g,start_ind,ev_id):
    start_ = timer()
    tgt_graph = GraphWithIndices( *map(lambda x:x.cpu(), tgt_graph ) )

    batch_input, batch_target = collate_fn([tgt_graph])

    with torch.no_grad():
        res = model_g(batch_input['inputs']).numpy()
        y_pred = res.flatten() > 0.9
    end_ = timer()
    start = timer()
    v1v2 = tgt_graph.edges [ y_pred ].numpy()

    hits = np.unique(v1v2)

    #np.save('saved_graphs/v1v2v3' + str(ev_id), v1v2)

    graph_dict = dict()

    v_all = None

    for hit in hits:
        res = v1v2[np.where(hit == v1v2[:, 0])][:, 1]
        if res.shape[0] > 0:

            graph_dict[hit] = res

    for hit in start_ind:

        #if hit not in graph_dict: continue
        paths = dfs(graph_dict, [hit],[],ev_id=ev_id)

        for path in paths:
             if len(path) == N_STATIONS:
                 if v_all is None:
                     v_all = np.array([path])
                 else:
                     v_all = np.concatenate((v_all, np.array([path]) ), axis=0)
    end= timer()
    #print('searching for tracks', end-start,'evaluation',end_-start_)
    eval_df = pd.DataFrame(v_all, columns=[f"hit_id_{n}" for n in range(1, N_STATIONS + 1)])

    eval_df[ [f"hit_id_{n}" for n in range(1,N_STATIONS+1)]] = v_all


    return eval_df

evaluator = EventEvaluator(parse_cfg, global_transformer, N_STATIONS)
events = evaluator.prepare(model_loader=GraphModelLoader())[0]
all_results = evaluator.build_all_tracks()
model_results = evaluator.run_model(get_graph_torcscipt, eval_event)
results_graphnet = evaluator.solve_results(model_results, all_results)

