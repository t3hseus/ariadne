import os
import sys
import gin
import numpy as np
import pandas as pd

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from IPython.core.display import clear_output, display

import matplotlib.pyplot as plt

import logging

logging.getLogger().setLevel(logging.DEBUG)

from eval.event_evaluation import EventEvaluator
from ariadne_v2.transformations import Compose, ConstraintsNormalize, ToCylindrical, DropSpinningTracks, DropShort, \
    DropEmpty

parse_cfg = {
    'csv_params': {
        "sep": '\s+',
        # "nrows": 15000,
        "encoding": 'utf-8',
        "names": ['event', 'x', 'y', 'z', 'station', 'track', 'px', 'py', 'pz', 'X0', 'Y0', 'Z0']
    },

    'input_file_mask': "./output_fake_5000.tsv", #_fake
    'events_quantity': '0..100'
}

global_transformer = Compose([
    DropSpinningTracks(),
    DropShort(num_stations=3),
    DropEmpty()
])

import scripts.clean_cache

# to clean cache if needed
# scripts.clean_cache.clean_jit_cache('20d')


from ariadne.graph_net.graph_utils.graph_prepare_utils import get_plain_graph, construct_output_graph
from ariadne.transformations import Compose, ConstraintsNormalize, ToCylindrical

from ariadne_v2.inference import IModelLoader

import torch

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
        import torch

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

GraphWithIndices = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y','v1v2','ev_id'])

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

def construct_graph_with_indices(graph, v1v2, ev_id):
    return GraphWithIndices(graph.X, graph.Ri, graph.Ro, graph.y, v1v2, ev_id)

def get_graph(event):

    event = event[['event', 'x', 'y', 'z', 'station', 'track', 'index_old']]

    try:
        event = transformer_g(event)
    except AssertionError as err:
        print("ASS error %r" % err)
        return None

    event.index = event['index_old'].values
    event = event[['event', 'r', 'phi', 'z', 'station', 'track']]

    nodes, edges = get_plain_graph(event, suffixes=suff_df, compute_is_true_track=False)

    graph = construct_output_graph(nodes, edges, ['r', 'phi', 'z'],
                                   [ 1., 1., 1.] ,index_label_prev='index_old_p',
                                                  index_label_current='index_old_c')
    ev_id = event.event.values[0]

    graph_with_inds = construct_graph_with_indices(graph,edges[ ['index_old_p','index_old_c'] ].values,
                                                    ev_id)


    return graph_with_inds


from ariadne.graph_net.dataset import collate_fn
import matplotlib
matplotlib.use('TKAgg')

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
def eval_event(tgt_graph, model_g,start_ind,ev_id):
    batch_input, batch_target = collate_fn([tgt_graph])
    with torch.no_grad():
        res = model_g(batch_input['inputs']).numpy()
        y_pred = res.flatten() > 0.9

        v1v2 = tgt_graph.v1v2 [ y_pred ]
        if v1v2.size == 0:return 0

        hits = np.unique(v1v2)

    #np.save('saved_graphs/v1v2v3' + str(ev_id), v1v2)

    graph_dict = dict()

    v_all = None

    for hit in hits:

        if ( res:=v1v2[ np.where( hit == v1v2[:,0] ) ][:,1] ).size > 0:

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

    eval_df = pd.DataFrame(v_all, columns=[f"hit_id_{n}" for n in range(1, N_STATIONS + 1)])

    eval_df[ [f"hit_id_{n}" for n in range(1,N_STATIONS+1)]] = v_all


    return eval_df

evaluator = EventEvaluator(parse_cfg, global_transformer, N_STATIONS)
events = evaluator.prepare(model_loader=GraphModelLoader())[0]
all_results = evaluator.build_all_tracks()
model_results = evaluator.run_model(get_graph, eval_event)
results_graphnet = evaluator.solve_results(model_results, all_results)

