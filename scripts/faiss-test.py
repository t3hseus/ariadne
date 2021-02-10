import logging
import sys
import os
sys.path.append(os.path.abspath('./'))
import torch

import pandas as pd
import numpy as np
import torch
import gin
import faiss
import itertools
import time

from tqdm import tqdm
from absl import flags
from absl import app
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.autograd.profiler import profile
from ariadne.tracknet_v2.model import TrackNETv2
from ariadne.tracknet_v2.dataset import TrackNetV2Dataset, TrackNetV2ExplicitDataset
from ariadne.tracknet_v2.metrics import point_in_ellipse
from ariadne.tracknet_v2_1.model import Classifier
from ariadne.tracknet_v2.data_loader import TrackNetV2DataLoader

import faiss


def find_nearest_hit_base(ellipses, y):
    centers = ellipses[:, :2]
    # print(centers)
    last_station_hits = deepcopy(y)
    # print(last_station_hits.size())
    # print('center', centers)
    # print(torch.isclose(torch.tensor(y, dtype=torch.double), last_station_hits))
    # print(self.last_station_hits[0:10])
    dists = torch.cdist(last_station_hits.float(), centers.float())
    # print(dists)
    min_ids = torch.argmin(dists, dim=0)
    minimal = last_station_hits[min_ids]
    # print(minimal)
    is_in_ellipse = point_in_ellipse(ellipses, minimal)
    return minimal, is_in_ellipse, min_ids

def cartesian(df1, df2):
    rows = itertools.product(df1.iterrows(), df2.iterrows())
    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    df_fakes = df[(df['track_left'] == -1) & (df['track_right'] == -1)]
    df = df[(df['track_left'] != df['track_right'])]
    df = pd.concat([df, df_fakes], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    return df.reset_index(drop=True)



gin.bind_parameter('TrackNETv2.input_features', 3)
gin.bind_parameter('TrackNETv2.conv_features', 32)
gin.bind_parameter('TrackNETv2.rnn_type', 'gru')
gin.bind_parameter('TrackNETv2.batch_first', True)

def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict =  checkpoint['state_dict']
    real_dict = {}
    for (k,v) in model_dict.items():
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

model = weights_update(model=TrackNETv2(), checkpoint=torch.load('../lightning_logs/version_112/checkpoints/epoch=161.ckpt'))
model.cuda()
class_model = weights_update(model=Classifier(), checkpoint=torch.load('../lightning_logs/class_epoch=1.ckpt'))
class_model.cuda()
use_classifier = False

gin.bind_parameter('TrackNetV2ExplicitDataset.input_dir', '../output/cgem_t_plain_explicit_test')
gin.bind_parameter('TrackNetV2ExplicitDataset.input_file', 'tracknet_test_all_explicit.npz')
data_loader = TrackNetV2DataLoader(
    batch_size=500,
    dataset=TrackNetV2ExplicitDataset,
    valid_size=1.0
).get_val_dataloader()

all_last_station_coordinates = np.load('../output/cgem_t_tracknet_valid/test_data_events_last_station.npz')
last_station_hits = all_last_station_coordinates['hits']
last_station_hits_events = all_last_station_coordinates['events']

all_tracks_df = pd.DataFrame(columns=['event', 'track', 'hit_0_id', 'hit_1_id', 'hit_2_id', 'px', 'py', 'pz', 'pred'])
reco_df = all_tracks_df.copy()

from ariadne.tracknet_v2.metrics import point_in_ellipse


def find_nearest_hit(ellipses,hits):
    centers = ellipses[:,:2]
    dists = torch.cdist(hits.to(torch.float), centers.to(torch.float))
    minimal = hits[torch.argmin(dists, dim=0)]
    is_in_ellipse = point_in_ellipse(ellipses, minimal)
    return minimal, is_in_ellipse

def find_nearest_hit_faiss(torch_ellipses,index):
    ellipses = torch_ellipses.detach().cpu().numpy()
    centers = ellipses[:,:2]
    d, i = index.search(np.ascontiguousarray(centers), 1)
    #print('distance: \n', d)
    x_part = d.flatten() / ellipses[:, 2].flatten()**2
    #print('x_part: \n', x_part)
    y_part = d.flatten() / ellipses[:, 3].flatten()**2
    left_side = x_part + y_part
    is_in_ellipse = left_side <= 1
    #print('is_in_ellipse: \n', is_in_ellipse)
    return i, is_in_ellipse

def handle_event_to_df(batch_input, batch_len, batch_target, batch_real_flag, batch_last_station):
    t0 = time.time()
    all_last_y = batch_last_station
    with torch.no_grad():
        temp_dict = {}
        temp_df = pd.DataFrame(columns=['found_nothing', 'found_right_point', 'is_real_track'])
        if batch_input.ndim < 3:
            batch_input = np.expand_dims(batch_input, axis=0)
            batch_len = np.expand_dims(batch_len, axis=0)
            batch_target = np.expand_dims(batch_target, axis=0)
            batch_real_flag = np.expand_dims(batch_real_flag, axis=0)
        if all_last_y.ndim < 2:
            all_last_y = np.expand_dims(all_last_y, axis=0)
            #print('last_y \n', all_last_y)
        all_last_y = torch.from_numpy(all_last_y).cuda()
        batch_target = torch.from_numpy(batch_target).cuda()
        test_pred = model(inputs=torch.from_numpy(batch_input).cuda(), input_lengths=torch.from_numpy(batch_len).cuda())
        batch_real_flag = torch.from_numpy(batch_real_flag).cuda()
        num_real_tracks = batch_real_flag.sum()
        t1 = time.time()
        nearest_points, is_point_in_ellipse = find_nearest_hit(test_pred, all_last_y)
        t2 = time.time()
        nearest_points[~is_point_in_ellipse] = -2.
        if use_classifier:
            pred_classes = class_model(model.last_gru_output, nearest_points)
            softmax = torch.nn.Softmax()
            pred_classes = torch.argmax(softmax(pred_classes), dim=1)
        # print(pred_classes)
        is_prediction_true = (batch_target == nearest_points)
        is_prediction_true = is_prediction_true.to(torch.int).sum(dim = 1) / 2
        if use_classifier:
            found_points = is_point_in_ellipse & (batch_real_flag == 1) & (pred_classes == 0)
        else:
            found_points = is_point_in_ellipse & (batch_real_flag == 1)
        found_nothing = ~is_point_in_ellipse & (batch_real_flag == 1)
        found_right_points = found_points & (is_prediction_true == 1)
        num_points_for_real_tracks = found_points.sum()
        num_fales_for_real_tracks = found_nothing.sum()
        temp_dict['found_nothing'] = found_nothing.detach().cpu().numpy()
        temp_dict['found_right_point'] = found_right_points.detach().cpu().numpy()
        temp_dict['is_real_track'] = batch_real_flag.detach().cpu().numpy()
        temp_df = pd.DataFrame(temp_dict)
        t3 = time.time()
        return temp_df, t2-t1, t3-t0


def handle_event_to_df_faiss(batch_input, batch_len, batch_target, batch_real_flag, batch_last_station):
    t0 = time.time()
    all_last_y = batch_last_station.astype('float32')
    index = faiss.IndexFlatL2(2)
    index.add(all_last_y)
    with torch.no_grad():
        temp_dict = {}
        temp_df = pd.DataFrame(columns=['found_nothing', 'found_right_point', 'is_real_track'])
        if batch_input.ndim < 3:
            batch_input = np.expand_dims(batch_input, axis=0)
            batch_len = np.expand_dims(batch_len, axis=0)
            batch_target = np.expand_dims(batch_target, axis=0)
            batch_real_flag = np.expand_dims(batch_real_flag, axis=0)
        if all_last_y.ndim < 2:
            all_last_y = np.expand_dims(all_last_y, axis=0)
            #print('last_y \n', all_last_y)
        test_pred = model(inputs=torch.from_numpy(batch_input).cuda(), input_lengths=torch.from_numpy(batch_len).cuda())
        num_real_tracks = batch_real_flag.sum()
        t1 = time.time()
        nearest_points_index, is_point_in_ellipse = find_nearest_hit_faiss(test_pred, index)
        t2 = time.time()
        nearest_points = all_last_y[nearest_points_index.flatten()]
        if use_classifier:
            pred_classes = class_model(model.last_gru_output, nearest_points)
            softmax = torch.nn.Softmax()
            pred_classes = torch.argmax(softmax(pred_classes), dim=1).detach().cpu().numpy()
        # print(pred_classes)
        is_prediction_true = np.logical_and(batch_target, nearest_points)
        is_prediction_true = is_prediction_true.all(1)
        #is_point_in_ellipse = is_point_in_ellipse.all(1)

        if use_classifier:
            found_points = is_point_in_ellipse & (batch_real_flag == 1) & (pred_classes == 0)
        else:
            found_points = is_point_in_ellipse & (batch_real_flag == 1)
        found_nothing = ~is_point_in_ellipse & (batch_real_flag == 1)
        found_right_points = found_points & (is_prediction_true == 1)
        num_points_for_real_tracks = found_points.sum()
        num_fales_for_real_tracks = found_nothing.sum()
        temp_dict['found_nothing'] = found_nothing
        temp_dict['found_right_point'] = found_right_points
        temp_dict['is_real_track'] = batch_real_flag
        temp_df = pd.DataFrame(temp_dict)
        t3 = time.time()
        return temp_df, t2-t1, t3-t0


result_df = pd.DataFrame(columns=['found_nothing', 'found_right_point','is_real_track', 'px','py','pz','p'])
result_df_faiss = pd.DataFrame(columns=['found_nothing', 'found_right_point','is_real_track', 'px','py','pz','p'])

all_time_no_faiss = 0
all_time_faiss = 0
search_time_no_faiss = 0
search_time_faiss = 0
num_batches = 0

events = np.load('../output/cgem_t_tracknet_valid/test_data_events.npz')
for batch_event in tqdm(np.unique(events['events'])):
    batch_x = events['x'][events['events'] == batch_event]
    batch_len = events['len'][events['events'] == batch_event]
    batch_y = events['y'][events['events'] == batch_event]
    batch_labels = events['is_real'][events['events'] == batch_event]
    last_station_data = last_station_hits[last_station_hits_events == batch_event]
    df, elapsed_time, total_elapsed = handle_event_to_df(batch_x, batch_len, batch_y, batch_labels, last_station_data)
    search_time_no_faiss += elapsed_time
    all_time_no_faiss += total_elapsed
    df_faiss, elapsed_time_faiss, total_elapsed_faiss = handle_event_to_df_faiss(batch_x, batch_len, batch_y, batch_labels, last_station_data)
    search_time_faiss += elapsed_time_faiss
    all_time_faiss += total_elapsed_faiss
    result_df = pd.concat([result_df,df], axis=0)
    result_df_faiss = pd.concat([result_df, df_faiss], axis=0)
    num_batches +=1
real_tracks = deepcopy(result_df.loc[result_df['is_real_track']==1, ])

recall = real_tracks['found_right_point'].sum() / float(len(real_tracks))
precision = real_tracks['found_right_point'].sum() / float(len(result_df) - result_df['found_nothing'].sum())
fale_rate = (real_tracks['found_nothing'].sum()) / float(len(real_tracks))

print('Test set results:')
#print('Accuracy:  %.4f' % sklearn.metrics.accuracy_score(y_true, y_pred))
print('Precision: %.4f' % precision)
print('Recall:    %.4f' % recall)
print('Fale rate:    %.4f' % fale_rate)
print('Time for search: %.4f sec' % search_time_no_faiss)
print('Search time per batch: %.4f sec' % (search_time_no_faiss/num_batches))

print('Total time for search: %.4f sec' % all_time_no_faiss)
print('Total time per batch: %.4f sec' % (all_time_no_faiss/num_batches))

result_df.loc[result_df['is_real_track']==1,].found_right_point.sum()

real_tracks = deepcopy(result_df_faiss.loc[result_df_faiss['is_real_track']==1, ])

recall = real_tracks['found_right_point'].sum() / float(len(real_tracks))
precision = real_tracks['found_right_point'].sum() / float(len(result_df_faiss) - result_df_faiss['found_nothing'].sum())
fale_rate = (real_tracks['found_nothing'].sum()) / float(len(real_tracks))

print('Test set results:')
#print('Accuracy:  %.4f' % sklearn.metrics.accuracy_score(y_true, y_pred))
print('Precision: %.4f' % precision)
print('Recall:    %.4f' % recall)
print('Fale rate:    %.4f' % fale_rate)
print('Time for search: %.4f sec' % search_time_faiss)
print('Search time per batch: %.4f sec' % (search_time_faiss/num_batches))
print('Total time: %.4f sec' % all_time_faiss)
print('Total time per batch: %.4f sec' % (all_time_faiss/num_batches))
result_df.loc[result_df['is_real_track']==1,].found_right_point.sum()
