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
import matplotlib.pyplot as plt
from ariadne.tracknet_v2.model import TrackNETv2
from ariadne.tracknet_v2_1.model import TrackNetClassifier
from ariadne.utils import weights_update, find_nearest_hit, find_nearest_hit_no_faiss,\
    get_diagram_arr_linspace,draw_for_col,draw_from_data
from ariadne.tracknet_v2.metrics import point_in_ellipse
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

gin.bind_parameter('TrackNETv2.input_features', 3)
gin.bind_parameter('TrackNETv2.conv_features', 32)
gin.bind_parameter('TrackNETv2.rnn_type', 'gru')
gin.bind_parameter('TrackNETv2.batch_first', True)

model = weights_update(model=TrackNETv2(), checkpoint=torch.load('../lightning_logs/TrackNETv2/version_44/epoch=34-step=40144.ckpt'))
model.to(DEVICE)
class_model = weights_update(model=TrackNetClassifier(), checkpoint=torch.load('../lightning_logs/TrackNetClassifier/version_52/epoch=1-step=6249.ckpt'))
class_model.to(DEVICE)
use_classifier = False

events = np.load('../output/cgem_t_tracknet_valid/tracknet_all_3.txt.npz')
all_last_station_coordinates = np.load('../output/cgem_t_tracknet_valid/tracknet_all_3.txt_last_station.npz')
last_station_hits = all_last_station_coordinates['hits']
last_station_hits_events = all_last_station_coordinates['events']


all_tracks_df = pd.DataFrame(columns=['event', 'track', 'hit_0_id', 'hit_1_id', 'hit_2_id', 'px', 'py', 'pz', 'pred'])
reco_df = all_tracks_df.copy()

def handle_event_to_df(batch_input, batch_len, batch_target, batch_real_flag, batch_last_station, batch_momentum):
    t0 = time.time()
    all_last_y = batch_last_station
    with torch.no_grad():
        temp_dict = {}
        temp_df = pd.DataFrame(columns=['found', 'found_right_point', 'is_real_track','px', 'py','pz', 'p'])
        if batch_input.ndim < 3:
            batch_input = np.expand_dims(batch_input, axis=0)
            batch_len = np.expand_dims(batch_len, axis=0)
            batch_target = np.expand_dims(batch_target, axis=0)
            batch_real_flag = np.expand_dims(batch_real_flag, axis=0)
        if all_last_y.ndim < 2:
            all_last_y = np.expand_dims(all_last_y, axis=0)
        all_last_y = torch.from_numpy(all_last_y).to(DEVICE)
        batch_target = torch.from_numpy(batch_target).to(DEVICE)
        test_pred,last_gru_output = model(inputs=torch.from_numpy(batch_input).to(DEVICE),
                          input_lengths=torch.from_numpy(batch_len).to(DEVICE),
                          return_gru_state=True)
        batch_real_flag = torch.from_numpy(batch_real_flag).to(DEVICE)
        num_real_tracks = batch_real_flag.sum()
        t1 = time.time()
        nearest_points, is_point_in_ellipse = find_nearest_hit_no_faiss(test_pred, all_last_y)
        t2 = time.time()
        nearest_points[~is_point_in_ellipse] = -2.
        if use_classifier:
            pred_classes = class_model(last_gru_output, nearest_points)
            softmax = torch.nn.Softmax()
            pred_classes = torch.argmax(softmax(pred_classes), dim=1)
        is_prediction_true = (batch_target == nearest_points)
        is_prediction_true = is_prediction_true.to(torch.int).sum(dim = 1) / 2.0
        if use_classifier:
            found_points = is_point_in_ellipse & (pred_classes == 1)
        else:
            found_points = is_point_in_ellipse
        found_right_points = found_points & (is_prediction_true == 1) & (batch_real_flag == 1)
        temp_dict['found'] = found_points.detach().cpu().numpy()
        temp_dict['found_right_point'] = found_right_points.detach().cpu().numpy()
        temp_dict['is_real_track'] = batch_real_flag.detach().cpu().numpy()
        temp_dict['px'] = batch_momentum[:, 0]
        temp_dict['py'] = batch_momentum[:, 1]
        temp_dict['pz'] = batch_momentum[:, 2]
        temp_dict['p'] = np.linalg.norm(batch_momentum, axis=1)
        temp_df = pd.DataFrame(temp_dict)
        t3 = time.time()
        return temp_df, t2-t1, t3-t0


def handle_event_to_df_faiss(batch_input, batch_len, batch_target, batch_real_flag, batch_last_station, batch_momentum):
    t0 = time.time()
    all_last_y = batch_last_station.astype('float32')
    with torch.no_grad():
        temp_dict = {}
        temp_df = pd.DataFrame(columns=['found', 'found_right_point', 'is_real_track', 'px', 'py','pz', 'p'])
        if batch_input.ndim < 3:
            batch_input = np.expand_dims(batch_input, axis=0)
            batch_len = np.expand_dims(batch_len, axis=0)
            batch_target = np.expand_dims(batch_target, axis=0)
            batch_real_flag = np.expand_dims(batch_real_flag, axis=0)
        if all_last_y.ndim < 2:
            all_last_y = np.expand_dims(all_last_y, axis=0)
        test_pred, last_gru_output = model(inputs=torch.from_numpy(batch_input).to(DEVICE),
                                           input_lengths=torch.from_numpy(batch_len).to(DEVICE),
                                           return_gru_state=True)
        num_real_tracks = batch_real_flag.sum()
        t1 = time.time()
        nearest_points, is_point_in_ellipse = find_nearest_hit(test_pred.detach().cpu().numpy(), all_last_y)
        t2 = time.time()
        #nearest_points = nearest_points.detach().cpu().numpy()
        #is_point_in_ellipse = is_point_in_ellipse.detach().cpu().numpy()
        if use_classifier:
            pred_classes = class_model(last_gru_output,
                                       torch.from_numpy(nearest_points.astype('float')).to(DEVICE))
            softmax = torch.nn.Softmax()
            pred_classes = torch.argmax(softmax(pred_classes), dim=1).detach().cpu().numpy()
        is_prediction_true = np.logical_and(batch_target, nearest_points)
        is_prediction_true = is_prediction_true.all(1)
        if use_classifier:
            found_points = is_point_in_ellipse  & (pred_classes == 1)
        else:
            found_points = is_point_in_ellipse
        found_right_points = found_points & (is_prediction_true == 1) & (batch_real_flag == 1)
        temp_dict['found'] = found_points
        temp_dict['found_right_point'] = found_right_points
        temp_dict['is_real_track'] = batch_real_flag
        temp_dict['px'] = batch_momentum[:, 0]
        temp_dict['py'] = batch_momentum[:, 1]
        temp_dict['pz'] = batch_momentum[:, 2]
        temp_dict['p'] = np.linalg.norm(batch_momentum, axis=1)
        temp_df = pd.DataFrame(temp_dict)
        t3 = time.time()
        return temp_df, t2-t1, t3-t0


result_df = pd.DataFrame(columns=['found', 'found_right_point', 'is_real_track', 'px', 'py', 'pz', 'p'])
result_df_faiss = pd.DataFrame(columns=['found', 'found_right_point', 'is_real_track', 'px', 'py', 'pz', 'p'])

all_time_no_faiss = 0
all_time_faiss = 0
search_time_no_faiss = 0
search_time_faiss = 0
times = {i: [] for i in range(1, 20)}
num_events = 0
for batch_event in tqdm(np.unique(events['events'])):
    num_events += 1
    batch_x = events['x'][events['events'] == batch_event]
    batch_len = events['len'][events['events'] == batch_event]
    batch_y = events['y'][events['events'] == batch_event]
    batch_labels = events['is_real'][events['events'] == batch_event]
    batch_momentums = events['moments'][events['events'] == batch_event]
    last_station_data = last_station_hits[last_station_hits_events == batch_event]
    df, elapsed_time, total_elapsed = handle_event_to_df(batch_x,
                                                         batch_len,
                                                         batch_y,
                                                         batch_labels,
                                                         last_station_data,
                                                         batch_momentums)
    #print('\n df \n', df[['found', 'found_right_point', 'is_real_track']])
    search_time_no_faiss += elapsed_time
    all_time_no_faiss += total_elapsed
    df_faiss, elapsed_time_faiss, total_elapsed_faiss = handle_event_to_df_faiss(batch_x,
                                                                                 batch_len,
                                                                                 batch_y,
                                                                                 batch_labels,
                                                                                 last_station_data,
                                                                                 batch_momentums)
    #print('\n df_faiss \n', df_faiss[['found', 'found_right_point', 'is_real_track']])
    search_time_faiss += elapsed_time_faiss
    all_time_faiss += total_elapsed_faiss
    times[int(batch_labels.sum())].append(total_elapsed_faiss)
    result_df = pd.concat([result_df, df], axis=0)
    result_df_faiss = pd.concat([result_df_faiss, df_faiss], axis=0)
    if num_events == 3000:
        break

real_tracks = deepcopy(result_df.loc[result_df['is_real_track'] == 1, ])
from numpy import linalg as LA
recall = real_tracks['found_right_point'].sum() / float(len(real_tracks))
precision = real_tracks['found_right_point'].sum() / float(result_df['found'].sum())
result_df['pt'] = LA.norm(result_df[['px','py']].values, axis=1)
result_df['cos_t'] = (result_df[['pz']].values / LA.norm(result_df[['px', 'py', 'pz']].values, axis=1, keepdims=True))
result_df['a_phi'] = np.arctan2(result_df[['px']].values, result_df[['py']].values)
print('\n ===> NO FAISS:')
print('Test set results:')
print(f'Precision: {precision}')
print(f'Recall:    {recall}')
print('-'*10)
print(f'Search time: {search_time_no_faiss} sec')
print(f'Search time per batch: {(search_time_no_faiss / num_events)} sec')
print('-'*10)
print(f'Total time: {all_time_no_faiss} sec')
print(f'Total time per batch: {(all_time_no_faiss / num_events)} sec')

real_tracks = deepcopy(result_df_faiss.loc[result_df_faiss['is_real_track'] == 1, ])
recall = real_tracks['found_right_point'].sum() / float(len(real_tracks))
precision = result_df_faiss['found_right_point'].sum() / float(result_df_faiss['found'].sum())

print('\n ===> FAISS:')
print('Test set results:')
print(f'Precision: {precision}')
print(f'Recall:    {recall}')
print('-'*10)
print(f'Search time: {search_time_faiss} sec')
print(f'Search time per batch: {(search_time_faiss / num_events)} sec')
print('-'*10)
print(f'Total time: {all_time_faiss} sec')
print(f'Total time per batch: {(all_time_faiss/num_events)} sec')

result_df_faiss['pt'] = LA.norm(result_df_faiss[['px','py']].values, axis=1)
result_df_faiss['cos_t'] = (result_df_faiss[['pz']].values /
                            LA.norm(result_df_faiss[['px', 'py', 'pz']].values, axis=1, keepdims=True))
result_df_faiss['a_phi'] = np.arctan2(result_df_faiss[['px']].values, result_df_faiss[['py']].values)


true_tracks_result_df = result_df_faiss[result_df_faiss.is_real_track == 1]
tracks_pred_true = true_tracks_result_df[true_tracks_result_df.found_right_point]
tracks_real = true_tracks_result_df[(true_tracks_result_df.found_right_point == False)  |
                                    (true_tracks_result_df.found == False)]
#draw_for_col(true_tracks_result_df, tracks_pred_true, 'pt', '$pt$', 2000, 175)
draw_for_col(true_tracks_result_df, tracks_pred_true, 'pt', '$pt$', num_events, 175, style='plot')
#draw_for_col(true_tracks_result_df, tracks_pred_true, 'a_phi', '$a_\\phi$', 2000, 175)
draw_for_col(true_tracks_result_df, tracks_pred_true, 'a_phi', '$a_\\phi$', num_events, 175, style='plot')
#draw_for_col(true_tracks_result_df, tracks_pred_true, 'cos_t', '$cos_t$', 2000, 175)
draw_for_col(true_tracks_result_df, tracks_pred_true,'cos_t', '$cos_t$', num_events, 175, style='plot')
times_mean = {k: np.mean(v) for k, v in times.items()}
times_std = {k: np.std(v) for k, v in times.items()}
draw_from_data(title = "TrackNETv2.1 Mean Processing Time vs Multiplicity (ms)",
               data_x=list(times_mean.keys()),
               data_y=[v * 1000 for v in times_mean.values()],
               data_y_err=[v * 1000 for v in times_std.values()],
               axis_x="multiplicity",
               mean_label='mean'
               )


