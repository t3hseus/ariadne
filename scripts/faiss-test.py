import logging
import sys
import os
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('.'))
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
from ariadne.utils import *
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
               draw_figures=True,
               plot_treshold=True,
               treshold_min=0.5,
               treshold_max=0.95,
               treshold_step=0.05
               ):
    path_to_tracknet_ckpt = get_checkpoint_path(**tracknet_ckpt_path_dict)
    path_to_classifier_ckpt = get_checkpoint_path(**classifier_ckpt_path_dict)

    model = weights_update(model=TrackNETv2(input_features=tracknet_input_features,
                                            conv_features=tracknet_conv_features,
                                            rnn_type='gru',
                                            batch_first=True),
                           checkpoint=torch.load(path_to_tracknet_ckpt, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    if use_classifier:
        class_model = weights_update(model=TrackNetClassifier(coord_size=2),
                                     checkpoint=torch.load(path_to_classifier_ckpt, map_location=torch.device(DEVICE)))
        class_model.to(DEVICE)

    events = load_data(input_dir, file_mask, None)
    all_last_station_coordinates = load_data(input_dir, last_station_file_mask, None)

    last_station_hits = all_last_station_coordinates['hits']
    last_station_hits_events = all_last_station_coordinates['events']

    all_tracks_df = pd.DataFrame(columns=['event', 'track', 'hit_0_id', 'hit_1_id', 'hit_2_id', 'px', 'py', 'pz', 'pred'])
    reco_df = all_tracks_df.copy()

    def handle_event_to_df_faiss(batch_input,
                                 batch_len,
                                 batch_target,
                                 batch_real_flag,
                                 batch_last_station,
                                 batch_momentum,
                                 treshold=0.5):
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
            last_station_index = store_in_index(all_last_y, num_components=2)
            test_pred, last_gru_output = model(inputs=torch.from_numpy(batch_input).to(DEVICE),
                                               input_lengths=torch.from_numpy(batch_len).to(DEVICE),
                                               return_gru_states=True)
            num_real_tracks = batch_real_flag.sum()
            print(batch_input[batch_real_flag])
            print(batch_target[batch_real_flag])
            print(test_pred[batch_real_flag])
            test_pred = test_pred[:, -1, :]
            last_gru_output = last_gru_output[:, -1, :]
            nearest_hits_index = search_in_index(test_pred.detach().cpu().numpy()[:, :2], last_station_index, 1, n_dim=2)
            nearest_hits = all_last_y[nearest_hits_index]
            nearest_hits, is_point_in_ellipse = filter_hits_in_ellipses(test_pred.cpu().detach().numpy(),
                                                                    nearest_hits,
                                                                    nearest_hits_index,
                                                                    filter_station=True,
                                                                    z_last=True,
                                                                    find_n=nearest_hits_index.shape[1],
                                                                    n_dim=2)

            print(nearest_hits[batch_real_flag])
            print(is_point_in_ellipse[batch_real_flag])
            last_gru_output = last_gru_output.unsqueeze(1).repeat(1, is_point_in_ellipse.shape[-1], 1, 1).reshape(-1, last_gru_output.shape[-1])
            nearest_points = nearest_hits.reshape(-1, nearest_hits.shape[-1])
            print(nearest_points.shape)
            print(is_point_in_ellipse.shape)
            batch_target = np.expand_dims(batch_target, 1).repeat(is_point_in_ellipse.shape[-1]).reshape(-1,2)
            batch_momentum = np.expand_dims(batch_momentum, 1).repeat(is_point_in_ellipse.shape[-1]).reshape(-1, 3)
            batch_real_flag = np.expand_dims(batch_real_flag, 1).repeat(is_point_in_ellipse.shape[-1]).reshape(-1)
            print(batch_real_flag.sum())
            is_point_in_ellipse = is_point_in_ellipse.reshape(-1)
            t1 = time.time()
            print(is_point_in_ellipse.shape)
            t2 = time.time()
            if use_classifier:
                pred_classes = class_model(last_gru_output[is_point_in_ellipse][:,-1],
                                           torch.from_numpy(nearest_points[is_point_in_ellipse].astype('float')).to(DEVICE))
                confidence = deepcopy(F.sigmoid(pred_classes))
                pred_classes = (F.sigmoid(pred_classes) > treshold).squeeze().detach().cpu().numpy()
            is_prediction_true = np.logical_and(batch_target, nearest_points)
            is_prediction_true = is_prediction_true.all(1)
            print(is_prediction_true)
            print(nearest_points)
            if use_classifier:
                found_points = is_point_in_ellipse
                found_points[is_point_in_ellipse] = (pred_classes == 1)
            else:
                found_points = is_point_in_ellipse
            print(found_points[batch_real_flag])
            print(is_prediction_true[batch_real_flag])
            found_right_points = found_points & (is_prediction_true ==
1) & (batch_real_flag == 1)
            print(found_right_points)
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

    num_tresholds = int((treshold_max - treshold_min) / treshold_step)
    tresholds = np.linspace(treshold_min, treshold_max, num_tresholds)

    metrics = {'precision': [], 'recall': []}
    treshold = treshold_min
    for i in tqdm(range(num_tresholds)):
        num_events = 0
        for batch_event in np.unique(events['events'])[:max_num_events]:
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
                                                                                         batch_momentums,
                                                                                         treshold=treshold)
            #print('\n df_faiss \n', df_faiss[['found', 'found_right_point', 'is_real_track']])
            search_time_faiss += elapsed_time_faiss
            all_time_faiss += total_elapsed_faiss
            times[int(batch_labels.sum())].append(total_elapsed_faiss)
            result_df_faiss = pd.concat([result_df_faiss, df_faiss], axis=0)

        real_tracks = deepcopy(result_df_faiss.loc[result_df_faiss['is_real_track'] == 1, ])
        recall = result_df_faiss['found_right_point'].sum() / (result_df_faiss['is_real_track'].sum() + 1e-6)
        precision = result_df_faiss['found_right_point'].sum() / (float(result_df_faiss['found'].sum()) + 1e-6)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)

        LOGGER.info('\n ===> FAISS: \n'
                    'Test set results: \n'
                    f'------> Treshold: {treshold}, {num_events} events\n'
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
            draw_for_col(real_tracks_result_df, tracks_pred_true, 'pt', '$pt$', num_events, 175, style='boxplot', treshold=treshold)
            draw_for_col(real_tracks_result_df, tracks_pred_true, 'a_phi', '$a_\\phi$', num_events, 175, style='boxplot', treshold=treshold)
            draw_for_col(real_tracks_result_df, tracks_pred_true,'cos_t', '$cos_t$', num_events, 175, style='boxplot', treshold=treshold)
            draw_for_col(found_tracks_result_df, tracks_pred_true, 'pt', '$pt$', num_events, 175, metric='precision', style='plot', treshold=treshold)
            draw_for_col(found_tracks_result_df, tracks_pred_true, 'a_phi', '$a_\\phi$', num_events, 175, style='plot', metric='precision', treshold=treshold)
            draw_for_col(found_tracks_result_df, tracks_pred_true, 'cos_t', '$cos_t$', num_events, 175, style='plot', metric='precision', treshold=treshold)
            times_mean = {k: np.mean(v) for k, v in times.items()}
            times_std = {k: np.std(v) for k, v in times.items()}
            draw_from_data(title=f"TrackNETv2.1 Mean Processing Time vs Multiplicity (ms), treshold {treshold}",
                           data_x=list(times_mean.keys()),
                           data_y=[v * 1000 for v in times_mean.values()],
                           data_y_err=[v * 1000 for v in times_std.values()],
                           axis_x="multiplicity",
                           mean_label='mean'
                           )
        if not use_classifier:
            break
        treshold += treshold_step

    if use_classifier and plot_treshold:
        draw_treshold_plots(title=f"TrackNETv2.1 Metric values vs Treshold, {max_num_events} events per value",
                            data_x=tresholds,
                            data_y_dict=metrics,
                            model_name='tracknet_v2_1',
                            total_events=max_num_events)
    if not use_classifier and plot_treshold:
        LOGGER.warning('You can apply treshold only on classifier logits, not TrackNETv2. Skipping plotting')



def main(argv):
    del argv
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    LOGGER.info("CONFIG: %s" % gin.config_str())
    faiss_test()

if __name__ == '__main__':
    app.run(main)