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
from ariadne.tracknet_v2_1.model import Classifier
from ariadne.utils import cartesian, weights_update, find_nearest_hit, find_nearest_hit_no_faiss
from ariadne.tracknet_v2.metrics import point_in_ellipse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

gin.bind_parameter('TrackNETv2.input_features', 3)
gin.bind_parameter('TrackNETv2.conv_features', 32)
gin.bind_parameter('TrackNETv2.rnn_type', 'gru')
gin.bind_parameter('TrackNETv2.batch_first', True)

model = weights_update(model=TrackNETv2(), checkpoint=torch.load('../lightning_logs/version_112/checkpoints/epoch=161.ckpt'))
model.to(DEVICE)
class_model = weights_update(model=Classifier(), checkpoint=torch.load('../lightning_logs/Classifier/version_39/_epoch=99-step=0.ckpt'))
class_model.to(DEVICE)
use_classifier = True

events = np.load('../output/cgem_t_tracknet_valid/test_data_events.npz')
all_last_station_coordinates = np.load('../output/cgem_t_tracknet_valid/test_data_events_last_station.npz')
last_station_hits = all_last_station_coordinates['hits']
last_station_hits_events = all_last_station_coordinates['events']


all_tracks_df = pd.DataFrame(columns=['event', 'track', 'hit_0_id', 'hit_1_id', 'hit_2_id', 'px', 'py', 'pz', 'pred'])
reco_df = all_tracks_df.copy()

def handle_event_to_df(batch_input, batch_len, batch_target, batch_real_flag, batch_last_station, batch_moment):
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
        test_pred = model(inputs=torch.from_numpy(batch_input).to(DEVICE), input_lengths=torch.from_numpy(batch_len).to(DEVICE))
        batch_real_flag = torch.from_numpy(batch_real_flag).to(DEVICE)
        num_real_tracks = batch_real_flag.sum()
        t1 = time.time()
        nearest_points, is_point_in_ellipse = find_nearest_hit_no_faiss(test_pred, all_last_y)
        t2 = time.time()
        nearest_points[~is_point_in_ellipse] = -2.
        if use_classifier:
            pred_classes = class_model(model.last_gru_output, nearest_points)
            softmax = torch.nn.Softmax()
            pred_classes = torch.argmax(softmax(pred_classes), dim=1)
        is_prediction_true = (batch_target == nearest_points)
        is_prediction_true = is_prediction_true.to(torch.int).sum(dim = 1) / 2.0
        if use_classifier:
            found_points = is_point_in_ellipse & (pred_classes == 0)
        else:
            found_points = is_point_in_ellipse
        found_right_points = found_points & (is_prediction_true == 1) & (batch_real_flag == 1)
        temp_dict['found'] = found_points.detach().cpu().numpy()
        temp_dict['found_right_point'] = found_right_points.detach().cpu().numpy()
        temp_dict['is_real_track'] = batch_real_flag.detach().cpu().numpy()
        temp_dict['px'] = batch_moment[:, 0]
        temp_dict['py'] = batch_moment[:, 1]
        temp_dict['pz'] = batch_moment[:, 2]
        temp_dict['p'] = np.linalg.norm(batch_moment, axis=1)
        temp_df = pd.DataFrame(temp_dict)
        t3 = time.time()
        return temp_df, t2-t1, t3-t0


def handle_event_to_df_faiss(batch_input, batch_len, batch_target, batch_real_flag, batch_last_station, batch_moment):
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
        test_pred = model(inputs=torch.from_numpy(batch_input).to(DEVICE), input_lengths=torch.from_numpy(batch_len).to(DEVICE))
        num_real_tracks = batch_real_flag.sum()
        t1 = time.time()
        nearest_points, is_point_in_ellipse = find_nearest_hit(test_pred.detach().cpu().numpy(), all_last_y)
        t2 = time.time()
        #nearest_points = nearest_points.detach().cpu().numpy()
        #is_point_in_ellipse = is_point_in_ellipse.detach().cpu().numpy()
        if use_classifier:
            pred_classes = class_model(model.last_gru_output, torch.from_numpy(nearest_points.astype('float')).to(DEVICE))
            softmax = torch.nn.Softmax()
            pred_classes = torch.argmax(softmax(pred_classes), dim=1).detach().cpu().numpy()
        is_prediction_true = np.logical_and(batch_target, nearest_points)
        is_prediction_true = is_prediction_true.all(1)
        if use_classifier:
            found_points = is_point_in_ellipse  & (pred_classes == 0)
        else:
            found_points = is_point_in_ellipse
        found_right_points = found_points & (is_prediction_true == 1) & (batch_real_flag == 1)
        temp_dict['found'] = found_points
        temp_dict['found_right_point'] = found_right_points
        temp_dict['is_real_track'] = batch_real_flag
        temp_dict['px'] = batch_moment[:, 0]
        temp_dict['py'] = batch_moment[:, 1]
        temp_dict['pz'] = batch_moment[:, 2]
        temp_dict['p'] = np.linalg.norm(batch_moment, axis=1)
        temp_df = pd.DataFrame(temp_dict)
        t3 = time.time()
        return temp_df, t2-t1, t3-t0


result_df = pd.DataFrame(columns=['found', 'found_right_point','is_real_track', 'px','py','pz','p'])
result_df_faiss = pd.DataFrame(columns=['found', 'found_right_point','is_real_track', 'px','py','pz','p'])

all_time_no_faiss = 0
all_time_faiss = 0
search_time_no_faiss = 0
search_time_faiss = 0
num_batches = 0

for batch_event in tqdm(np.unique(events['events'])):
    batch_x = events['x'][events['events'] == batch_event]
    batch_len = events['len'][events['events'] == batch_event]
    batch_y = events['y'][events['events'] == batch_event]
    batch_labels = events['is_real'][events['events'] == batch_event]
    batch_moments = events['moments'][events['events'] == batch_event]
    last_station_data = last_station_hits[last_station_hits_events == batch_event]
    df, elapsed_time, total_elapsed = handle_event_to_df(batch_x, batch_len, batch_y, batch_labels, last_station_data, batch_moments)
    search_time_no_faiss += elapsed_time
    all_time_no_faiss += total_elapsed
    df_faiss, elapsed_time_faiss, total_elapsed_faiss = handle_event_to_df_faiss(batch_x, batch_len, batch_y, batch_labels, last_station_data, batch_moments)
    search_time_faiss += elapsed_time_faiss
    all_time_faiss += total_elapsed_faiss
    result_df = pd.concat([result_df,df], axis=0)
    result_df_faiss = pd.concat([result_df, df_faiss], axis=0)
    num_batches +=1

real_tracks = deepcopy(result_df.loc[result_df['is_real_track']==1, ])
from numpy import linalg as LA
recall = real_tracks['found_right_point'].sum() / float(len(real_tracks))
precision = real_tracks['found_right_point'].sum() / float(result_df['found'].sum())
result_df['pt'] = LA.norm(result_df[['px','py']].values, axis=1)
result_df['cos_t'] = (result_df[['pz']].values/ LA.norm(result_df[['px','py','pz']].values, axis=1, keepdims=True))
result_df['a_phi'] = np.arctan2(result_df[['px']].values, result_df[['py']].values)
print('\n ===> NO FAISS:')
print('Test set results:')
print(f'Precision: {precision}')
print(f'Recall:    {recall}')
print('-'*10)
print(f'Search time: {search_time_no_faiss} sec')
print(f'Search time per batch: {(search_time_no_faiss/num_batches)} sec')
print('-'*10)
print(f'Total time: {all_time_no_faiss} sec')
print(f'Total time per batch: {(all_time_no_faiss/num_batches)} sec')

result_df.loc[result_df['is_real_track']==1,].found_right_point.sum()
real_tracks = deepcopy(result_df_faiss.loc[result_df_faiss['is_real_track']==1, ])
recall = real_tracks['found_right_point'].sum() / float(len(real_tracks))
precision = result_df_faiss['found_right_point'].sum() / float(result_df_faiss['found'].sum())

print('\n ===> FAISS:')
print('Test set results:')
print(f'Precision: {precision}')
print(f'Recall:    {recall}')
print('-'*10)
print(f'Search time: {search_time_faiss} sec')
print(f'Search time per batch: {(search_time_faiss/num_batches)} sec')
print('-'*10)
print(f'Total time: {all_time_faiss} sec')
print(f'Total time per batch: {(all_time_faiss/num_batches)} sec')

from scipy.interpolate import make_interp_spline, BSpline


def get_diagram_arr_linspace(all_real_hits, found_hits, start, end, num, col):
    spac = np.linspace(start, end, num=num)

    arr = []

    for i in range(len(spac) - 1):
        beg = spac[i]
        end = spac[i + 1]
        elems_real = all_real_hits[(all_real_hits[col] > beg) & (all_real_hits[col] < end)]
        elems_pred = found_hits[(found_hits[col] > beg) & (found_hits[col] < end)]
        if elems_real.empty:
            arr.append(np.NaN)
            continue
        arr.append(len(elems_pred) / len(elems_real))

    return arr, spac[:-1]


def draw_for_col(tracks_real, tracks_pred_true,
                 col, col_pretty, total_events, n_ticks=150, save_disk=True):
    start = tracks_real[tracks_real[col] > -np.inf][col].min()
    end = tracks_real[tracks_real[col] < np.inf][col].max()

    initial, spac = get_diagram_arr_linspace(tracks_real, tracks_pred_true, start, end, n_ticks, col)

    # mean line
    # find number of ticks until no nans present
    second = np.array([np.nan])
    count_start = n_ticks // 5
    while np.isnan(second).any():
        second, spac2 = get_diagram_arr_linspace(tracks_real, tracks_pred_true, start, end, count_start, col)
        count_start = count_start - count_start // 2

    xnew = np.linspace(spac2.min(), spac2.max(), count_start)

    spl = make_interp_spline(spac2, second, k=3)  # type: BSpline
    power_smooth = spl(xnew)

    maxX = end
    plt.figure(figsize=(8, 7))

    plt.subplot(111)
    plt.ylabel('Track efficiency', fontsize=12)
    plt.xlabel(col_pretty, fontsize=12)
    # plt.axis([0, maxX, 0, 1.005])
    plt.plot(spac, initial, alpha=0.8, lw=0.8)
    plt.title('GraphNet_V1 track efficiency vs impulse (%d events)' % total_events, fontsize=14)
    plt.plot(xnew, power_smooth, ls='--', label='mean', lw=2.5)
    plt.xticks(np.linspace(start, maxX, 8))
    plt.yticks(np.linspace(0, 1, 9))
    plt.legend(loc=0)
    plt.grid()
    plt.tight_layout()
    plt.rcParams['savefig.facecolor'] = 'white'
    os.makedirs('../output', exist_ok=True)
    plt.savefig('../output/img_track_eff_%s_ev%r_t%d.png' % (col, total_events, n_ticks), dpi=300)
    plt.show()



