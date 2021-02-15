import torch
import numpy as np
import itertools
import pandas as pd
from copy import deepcopy
from ariadne.tracknet_v2.metrics import point_in_ellipse
import faiss
def fix_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.deterministic = True #torch.backends.cudnn.determenistic
        torch.benchmark = False #torch.backends.cudnn.benchmark

def cartesian(df1, df2):
    rows = itertools.product(df1.iterrows(), df2.iterrows())
    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    df_fakes = df[(df['track_left'] == -1) & (df['track_right'] == -1)]
    df = df[(df['track_left'] != df['track_right'])]
    df = pd.concat([df, df_fakes], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    return df.reset_index(drop=True)

def weights_update(model, checkpoint):
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

def find_nearest_hit_no_faiss(ellipses, y, return_numpy=False):
    centers = ellipses[:, :2]
    last_station_hits = deepcopy(y)
    dists = torch.cdist(last_station_hits.float(), centers.float())
    minimal = last_station_hits[torch.argmin(dists, dim=0)]
    is_in_ellipse = point_in_ellipse(ellipses, minimal)
    if return_numpy:
        minimal = minimal.detach().cpu().numpy()
        is_in_ellipse  = is_in_ellipse.detach().cpu().numpy()
    return minimal, is_in_ellipse

def find_nearest_hit(ellipses, last_station_hits):
    #numpy, numpy -> numpy, numpy
    index = faiss.IndexFlatL2(2)
    index.add(last_station_hits.astype('float32'))
    #ellipses = torch_ellipses.detach().cpu().numpy()
    centers = ellipses[:,:2]
    d, i = index.search(np.ascontiguousarray(centers.astype('float32')), 1)
    x_part = d.flatten() / ellipses[:, 2].flatten()**2
    y_part = d.flatten() / ellipses[:, 3].flatten()**2
    left_side = x_part + y_part
    is_in_ellipse = left_side <= 1
    return last_station_hits[i.flatten()], is_in_ellipse