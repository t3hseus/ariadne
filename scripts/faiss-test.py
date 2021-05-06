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
from numpy import linalg as LA

from tqdm import tqdm
from absl import flags
from absl import app
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.nn import functional as F
from ariadne.tracknet_v2.model import TrackNETv2
from ariadne.tracknet_v2_1.model import TrackNetClassifier
from ariadne.utils import weights_update, find_nearest_hit, find_nearest_hit_old, find_nearest_hit_no_faiss,\
    get_diagram_arr_linspace, draw_for_col, draw_from_data, get_checkpoint_path, load_data
from ariadne.tracknet_v2.metrics import point_in_ellipse

LOGGER = logging.getLogger('ariadne.test')
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name='config', default=None,
    help='Path to the config file to use.'
)
flags.DEFINE_enum(
    name='log', default='INFO',
    enum_values=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
    help='Level of logging'
)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@gin.configurable()
def faiss_test(tracknet_ckpt_path_dict,
               classifier_ckpt_path_dict,
               max_num_events=1000,
               input_dir='output/cgem_t_plain_valid_v48_valid',
               file_mask='tracknet_all_3.npz',
               last_station_file_mask='tracknet_all_3_last_station.npz',
               tracknet_input_features=3,
               tracknet_conv_features=32,
               use_classifier=True,
               draw_figures=True):
    path_to_tracknet_ckpt = get_checkpoint_path(**tracknet_ckpt_path_dict)
    path_to_classifier_ckpt = get_checkpoint_path(**classifier_ckpt_path_dict)

    model = weights_update(model=TrackNETv2(input_features=tracknet_input_features,
                                            conv_features=tracknet_conv_features,
                                            rnn_type='gru',
                                            batch_first=True),
                           checkpoint=torch.load(path_to_tracknet_ckpt))
    model.to(DEVICE)
    class_model = weights_update(model=TrackNetClassifier(),
                                 checkpoint=torch.load(path_to_classifier_ckpt))
    class_model.to(DEVICE)

    events = load_data(input_dir, file_mask, None)
    all_last_station_coordinates = load_data(input_dir, last_station_file_mask, None)

    last_station_hits = all_last_station_coordinates['hits']
    last_station_hits_events = all_last_station_coordinates['events']

    all_tracks_df = pd.DataFrame(columns=['event', 'track', 'hit_0_id', 'hit_1_id', 'hit_2_id', 'px', 'py', 'pz', 'pred'])
    reco_df = all_tracks_df.copy()

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
            nearest_points, is_point_in_ellipse = find_nearest_hit_old(test_pred.detach().cpu().numpy(), all_last_y)
            t2 = time.time()
            if use_classifier:
                pred_classes = class_model(last_gru_output[is_point_in_ellipse],
                                           torch.from_numpy(nearest_points[is_point_in_ellipse].astype('float')).to(DEVICE))
                confidence = deepcopy(F.sigmoid(pred_classes))
                pred_classes = (F.sigmoid(pred_classes) > 0.6).squeeze().detach().cpu().numpy()
            is_prediction_true = np.logical_and(batch_target, nearest_points)
            is_prediction_true = is_prediction_true.all(1)
            if use_classifier:
                found_points = is_point_in_ellipse
                found_points[is_point_in_ellipse] = (pred_classes == 1)
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
        batch_momentums = events['momentums'][events['events'] == batch_event]
        last_station_data = last_station_hits[last_station_hits_events == batch_event]
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
        result_df_faiss = pd.concat([result_df_faiss, df_faiss], axis=0)
        if num_events == max_num_events:
            break

    real_tracks = deepcopy(result_df_faiss.loc[result_df_faiss['is_real_track'] == 1, ])
    recall = result_df_faiss['found_right_point'].sum() / (result_df_faiss['is_real_track'].sum() + 1e-6)
    precision = result_df_faiss['found_right_point'].sum() / (float(result_df_faiss['found'].sum()) + 1e-6)

    LOGGER.info('\n ===> FAISS: \n'
                'Test set results: \n'
                f'Precision: {precision} \n' 
                f'Recall:    {recall} \n' 
                f'{"-"*10} '
                f'\nSearch time: {search_time_faiss} sec \n'
                f'Search time per batch: {(search_time_faiss / num_events)} sec \n'
                f'{"-"*10} '
                f'Total time: {all_time_faiss} sec \n'
                ""f'Total time per batch: {(all_time_faiss/num_events)} sec')

    result_df_faiss['pt'] = LA.norm(result_df_faiss[['px','py']].values, axis=1)
    result_df_faiss['cos_t'] = (result_df_faiss[['pz']].values /
                                LA.norm(result_df_faiss[['px', 'py', 'pz']].values, axis=1, keepdims=True))
    result_df_faiss['a_phi'] = np.arctan2(result_df_faiss[['px']].values, result_df_faiss[['py']].values)


    real_tracks_result_df = result_df_faiss[result_df_faiss['is_real_track'].astype('int') == 1]
    found_tracks_result_df = result_df_faiss[result_df_faiss['found'].astype('int') == 1]
    tracks_pred_true = result_df_faiss[result_df_faiss['found_right_point'].astype('int') == 1]

    if draw_figures:
        draw_for_col(real_tracks_result_df, tracks_pred_true, 'pt', '$pt$', num_events, 175, style='plot')
        draw_for_col(real_tracks_result_df, tracks_pred_true, 'a_phi', '$a_\\phi$', num_events, 175, style='plot')
        draw_for_col(real_tracks_result_df, tracks_pred_true,'cos_t', '$cos_t$', num_events, 175, style='plot')
        draw_for_col(found_tracks_result_df, tracks_pred_true, 'pt', '$pt$', num_events, 175, metric='precision', style='plot')
        draw_for_col(found_tracks_result_df, tracks_pred_true, 'a_phi', '$a_\\phi$', num_events, 175, style='plot', metric='precision')
        draw_for_col(found_tracks_result_df, tracks_pred_true, 'cos_t', '$cos_t$', num_events, 175, style='plot', metric='precision')
        times_mean = {k: np.mean(v) for k, v in times.items()}
        times_std = {k: np.std(v) for k, v in times.items()}
        draw_from_data(title="TrackNETv2.1 Mean Processing Time vs Multiplicity (ms)",
                       data_x=list(times_mean.keys()),
                       data_y=[v * 1000 for v in times_mean.values()],
                       data_y_err=[v * 1000 for v in times_std.values()],
                       axis_x="multiplicity",
                       mean_label='mean'
                       )


def main(argv):
    del argv
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    LOGGER.info("CONFIG: %s" % gin.config_str())
    faiss_test()

if __name__ == '__main__':
    app.run(main)