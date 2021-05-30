import logging
import itertools
import os
from typing import List, Tuple, Optional, Iterable
from copy import deepcopy

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import gin

from ariadne.tracknet_v2.model import TrackNETv2
from ariadne.tracknet_v2.metrics import point_in_ellipse
from ariadne.preprocessing import (
    BaseTransformer,
    DataProcessor,
    DataChunk,
    ProcessedDataChunk,
    ProcessedData
)
from ariadne.utils import brute_force_hits_two_first_stations_cart, \
    find_nearest_hit, \
    weights_update, \
    store_in_index,\
    filter_hits_in_ellipse,\
    search_in_index, \
    is_ellipse_intersects_station,\
    filter_hits_in_ellipses

from ariadne.tracknet_v2_1.processor import TrackNetV21Processor, ProcessedTracknetDataChunk, ProcessedTracknetData, TracknetDataChunk
from ariadne.tracknet_v2_bmn.processor_bmn import TrackNetBMNProcessor

LOGGER = logging.getLogger('ariadne.prepare')

@gin.configurable(denylist=['data_df'])
class TrackNetV21BMNProcessorWithModel(TrackNetV21Processor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 name_suffix: str,
                 valid_size: float,
                 device: str,
                 tracknet_v2_model: TrackNETv2,
                 num_stations: int = 8,
                 station_params = None,
                 n_times_oversampling: int = 4,
                 tracknet_v2_checkpoint: str = '',
                 transforms: List[BaseTransformer] = None):
        super().__init__(
            data_df=data_df,
            output_dir=output_dir,
            transforms=transforms,
            name_suffix=name_suffix,
            valid_size=valid_size,
            n_times_oversampling=n_times_oversampling
            )
        self.output_name = f'{self.output_dir}/{name_suffix}'
        self.n_times_oversampling = n_times_oversampling
        self.valid_size = valid_size
        self.chunks = []
        self.device = torch.device(device)

        self.model = tracknet_v2_model()
        #self.z_values = {0: -0.9998595, 1: -0.953343, 2: -0.841152, 3: -0.675508, 4: -0.379517, 5: 0.180436}
        #self.model = TrackNETv2(input_features=3)
        print(tracknet_v2_checkpoint)
        if tracknet_v2_checkpoint is not None:
            print('update weights')
            self.model = weights_update(model=self.model, checkpoint=torch.load(tracknet_v2_checkpoint))
            print('updated')
        self.model.eval()
        self.model.to(self.device)
        self.stations_sizes = np.array(station_params)
        self.max_n_stations = len(self.stations_sizes)#snum_stations
        self.z_values = [15.229, 18.499, 21.604, 39.702, 64.535, 112.649, 135.330, 160.6635, 183.668]
        self.z_values = self.z_values[9-self.max_n_stations:]
        self.det_indices = [0, 1]

    def get_seeds(self, hits):
        st0_hits = hits[hits.station == 0][['x', 'y', 'z']].values
        st1_hits = hits[hits.station == 1][['x', 'y', 'z']].values
        # all possible combinations
        idx0 = range(len(st0_hits))
        idx1 = range(len(st1_hits))
        idx_comb = itertools.product(idx0, idx1)
        # unpack indices
        idx0, idx1 = zip(*idx_comb)
        idx0 = list(idx0)
        idx1 = list(idx1)
        # create seeds array
        seeds = np.zeros((len(idx0), 2, 3))
        seeds[:, 0, ] = st0_hits[idx0]
        seeds[:, 1, ] = st1_hits[idx1]
        return seeds

    def generate_chunks_iterable(self) -> Iterable[TracknetDataChunk]:
        if len(self.det_indices) > 1:
            self.data_df.loc[self.data_df.det == 1, 'station'] = self.data_df.loc[self.data_df.det == 1, 'station'].values + 3
            if self.max_n_stations < 9:
                self.data_df = self.data_df.loc[self.data_df['station'] > 8-self.max_n_stations, :]
                self.data_df.loc[:, 'station'] = self.data_df.loc[:, 'station'].values - (9-self.max_n_stations)
        LOGGER.info(f"range of x: {self.data_df['x'].min()} - {self.data_df['x'].max()} ")
        LOGGER.info(f"range of y: {self.data_df['y'].min()} - {self.data_df['y'].max()} ")
        LOGGER.info(f"range of z: {self.data_df['z'].min()} - {self.data_df['z'].max()} ")
        LOGGER.info(f'stations: {self.data_df["station"].min()}-{self.data_df["station"].max()}')
        return self.data_df.groupby('event')

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> TracknetDataChunk:
        processed = self.transformer(chunk_df)
        tracks = processed[processed['track'] != -1].groupby('track')
        return TracknetDataChunk(processed)

    def preprocess_chunk(self,
                         chunk: TracknetDataChunk,
                         idx: str) -> ProcessedTracknetDataChunk:
        chunk_df = chunk.df_chunk_data
        if chunk_df.empty:
            return ProcessedTracknetDataChunk(None, '', -1)
        chunk_id = int(chunk_df.event.values[0])
        output_name = f'{self.output_dir}/tracknet_with_model_{idx.replace(".txt", "")}'
        self.chunks.append(chunk_id)
        return ProcessedTracknetDataChunk(chunk_df, output_name, chunk_id)

    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedTracknetData:
        track_nums = {3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
        track_right_pred_nums = {3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
        track_all_pred_nums = {3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
        track_hit_eff = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9:0, 10:0}
        precisions_vs_multiplicity = {i: [] for i in range(30)}
        recalls_vs_multiplicity = {i: [] for i in range(30)}
        hit_eff_vs_multiplicity = {i: [] for i in range(30)}

        for chunk in chunks:

            if chunk.processed_object is None:
                continue
            LOGGER.info(f'=====> id {chunk.id}')
            df = chunk.processed_object
            tracks = df[df['track'] != -1].groupby('track')
            multiplicity = tracks.ngroups
            if multiplicity == 0:
                LOGGER.warning(f'Multiplicity in chunk #{chunk.id} is 0! This chunk is skipped')
                chunk.processed_object = None
                continue
            track_lists = {3: [], 4: [], 5: [], 6: [], 7: [], 8:[], 9:[]}
            track_indices = {3: [], 4: [], 5: [], 6: [], 7: [], 8:[], 9:[]}
            for i, track in tracks:
                temp_track = track[['x', 'y', 'z']].values
                if len(temp_track)>3:
                    track_lists[len(temp_track)].append(temp_track)
                    track_nums[len(temp_track)] += 1
            for stations_in_track, this_track_list in track_lists.items():
                if len(this_track_list) > 0:
                    track_lists[stations_in_track] = np.stack(this_track_list, 0)
                else:
                    track_lists[stations_in_track] = np.expand_dims(np.zeros((stations_in_track, 3)), 0)
            #chunk_data_x, chunk_data_real = brute_force_hits_two_first_stations_cart(df)  # initial seeds
            chunk_data_x = self.get_seeds(df)
            chunk_data_len = np.full(len(chunk_data_x), 2)
            chunk_data_event = np.full(len(chunk_data_x), chunk.id)
            event_hits = np.ascontiguousarray(df[['x', 'y', 'z']].values)
            event_hits[:, -1] *= 1000.
            event_index = store_in_index(event_hits)  # z is last!
            for i, track in tracks:
                temp_track = track[['x', 'y', 'z']].values
                temp_track[:, -1] *= 1000.
                temp_index = search_in_index(temp_track, event_index, 1)
                track_indices[len(temp_track)].append(temp_index.flatten())
            gru_candidates = []
            track_labels = []
            track_hit_labels = []
            x_candidates = []
            max_batch_size = 10000
            if chunk.id == 2:
                temp_track = torch.tensor([[[ 3.382228,   4.909874,  21.604   ],
                                         [  5.183506,  10.62523,   39.702   ],
                                         [  8.42668 ,  19.67131 ,  64.535   ],
                                         [ 16.08211 ,  36.84681 , 112.649   ]
                    #[[ -4.095308,    2.703733,   21.604  ],
                        #                    [ -6.530911,   6.497071,  39.702 ],
                                            # [ -9.494087,  12.82823,   64.535   ],
                                            # [-10.98873,   24.71189,  112.649   ],
                                            # [ -9.889756,  30.28516,  135.33    ],
                                            #  [ -7.072373,  36.8528,   160.6635  ]
                                            ]])
                this_chunk_prediction, this_chunk_gru = self.model(torch.tensor(temp_track).to(self.device),
                                                                   torch.tensor([temp_track.shape[1]],dtype=torch.int64).to(self.device),
                                                                   return_gru_states=True)

                new_pred = torch.full((len(this_chunk_prediction), 5), self.z_values[(temp_track.shape[1])] * 1000.)
                new_pred[:, :2] = this_chunk_prediction[:, -1, :2]
                new_pred[:, 3:] = this_chunk_prediction[:, -1, 2:]
                nearest_hits_index = search_in_index(new_pred.detach().cpu().numpy()[:, :3], event_index, 15)
                found_hits = event_hits[nearest_hits_index]
                h, el = filter_hits_in_ellipses(new_pred.cpu().detach().numpy(), found_hits, np.array(0),
                                        filter_station=True, z_last=True, find_n=1)
                #is_on_station = is_ellipse_intersects_station(this_chunk_prediction[0, -1].tolist(), self.stations_sizes[(temp_track.shape[1])].flatten().tolist())


            for stations_gone in range(2, self.max_n_stations):
                batch_xs_new = np.array(0)
                flag = 0
                num_batches = int(len(chunk_data_x) / max_batch_size) + 1
                chunk_data_xs = []
                chunk_data_lens = []
                if len(chunk_data_x) == 0:
                    LOGGER.info('Have zero ellipces on this station! Skipping other stations')
                    break
                for batch_num in range(num_batches):
                    min_i = max_batch_size*batch_num
                    max_i = min(max_batch_size + min_i, len(chunk_data_x))
                    if min_i==max_i:
                        LOGGER.info('Have zero ellipces on this station! Skipping other stations')
                        break
                    this_chunk_data_x = chunk_data_x[min_i: max_i]
                    this_chunk_data_len = chunk_data_len[min_i: max_i]
                    chunk_prediction, chunk_gru = self.model(torch.tensor(this_chunk_data_x).to(self.device),
                                                                  torch.tensor(this_chunk_data_len, dtype=torch.int64).to(self.device),
                                                                  return_gru_states=True)
                    new_pred = torch.full((len(chunk_prediction), 5), self.z_values[stations_gone]*1000.)
                    new_pred[:, :2] = chunk_prediction[:, -1, :2]
                    new_pred[:, 3:] = chunk_prediction[:, -1, 2:]
                    chunk_prediction = new_pred
                    nearest_hits_index = search_in_index(chunk_prediction.detach().cpu().numpy()[:, :3], event_index, 15)
                    found_hits = event_hits[nearest_hits_index]
                    nearest_hits, in_ellipse = filter_hits_in_ellipses(chunk_prediction.cpu().detach().numpy(),
                                                                       found_hits,
                                                                       nearest_hits_index,
                                                                       filter_station=True,
                                                                       z_last=True,
                                                                       find_n=nearest_hits_index.shape[1])

                    nearest_hits[:, :, -1] /= 1000.
                    temp = (nearest_hits[:, :, -1] == self.z_values[stations_gone])
                    nearest_hits_mask = in_ellipse * (temp)
                    is_empty = (nearest_hits_mask.sum(axis=-1) == 0)
                    empty_ellipses = chunk_prediction[is_empty].detach().cpu().numpy()
                    empty_xs = this_chunk_data_x[is_empty]
                    if stations_gone == 4 and chunk.id==2:
                        print('im okey')

                    empty_grus = chunk_gru[is_empty].detach().cpu().numpy()
                    xs = np.expand_dims(this_chunk_data_x, 1).repeat(nearest_hits_mask.shape[-1], 1)
                    grus = np.expand_dims(chunk_gru.detach().cpu().numpy(), 1).repeat(nearest_hits_mask.shape[-1], 1)
                    batch_xs_new = np.zeros((len(xs), nearest_hits_mask.shape[-1], xs.shape[2]+1, 3))
                    batch_xs_new[:, :, :xs.shape[2], :] = xs
                    batch_xs_new[:, :, xs.shape[2], :] = nearest_hits
                    batch_xs_new = batch_xs_new[nearest_hits_mask].reshape(-1, stations_gone+1, 3)
                    grus = grus[nearest_hits_mask].reshape(-1, grus.shape[-2], grus.shape[-1])
                    if stations_gone > 3:
                         ellipse_to_use_list  = []
                         temp_el_list = []
                         for el_num, ellipse in enumerate(empty_ellipses):
                             orig_ellipse = np.concatenate((ellipse[:2], ellipse[3:]), 0)
                             is_ellipse_on_station = is_ellipse_intersects_station(orig_ellipse.tolist(),
                                                                                   self.stations_sizes[stations_gone].tolist())
                             temp_el_list.append(is_ellipse_on_station)
                             ellipse_to_use_list.append(True) #is_ellipse_on_station == False) # вот тут я убрала фильтр по станции!
                         print(empty_xs.shape)
                         empty_xs = empty_xs[ellipse_to_use_list]
                         empty_grus = empty_grus[ellipse_to_use_list]
                         labels=[]
                         hit_labels=[]
                         for track in track_lists[stations_gone]:
                             expanded_track = np.expand_dims(track, 0).repeat(len(empty_xs), 0)
                             labels_this_track = np.all(np.all(np.equal(expanded_track, empty_xs), axis=-1),
                                                          axis=-1)
                             if sum(labels_this_track)==0 and track.sum()>0:
                                 print(track)

                             if len(labels) > 0:
                                 labels = labels + labels_this_track

                                 hit_labels =  np.maximum(hit_labels, np.sum(np.all(np.equal(expanded_track, empty_xs), axis=-1), axis=-1))
                             else:
                                 labels = labels_this_track

                                 hit_labels = np.sum(np.all(np.equal(expanded_track, empty_xs), axis=-1), axis=-1)
                         hit_labels = hit_labels.astype('float') / (stations_gone-1)
                         if len(ellipse_to_use_list) > 0:
                             track_right_pred_nums[stations_gone] += labels.sum()
                             track_all_pred_nums[stations_gone] += len(labels)
                             track_hit_eff[stations_gone] += np.mean(hit_labels)
                             track_hit_labels.extend(hit_labels)
                             gru_candidates.extend(empty_grus[:, -2])
                             track_labels.extend(labels)
                             x_candidates.extend(empty_xs[:, -1])
                    if (stations_gone == (self.max_n_stations-1)):
                        print(stations_gone == (self.max_n_stations-1))
                        labels = []
                        hit_labels = []
                        for track in track_lists[stations_gone+1]:
                            expanded_track = np.expand_dims(track, 0).repeat(len(batch_xs_new), 0)
                            if len(labels) > 0:
                                labels = labels + np.all(np.all(np.equal(expanded_track, batch_xs_new), axis=-1), axis=-1)
                                hit_labels = np.maximum(hit_labels,np.sum(np.all(np.equal(expanded_track, batch_xs_new), axis=-1),
                                                                 axis=-1))
                            else:
                                labels = np.all(np.all(np.equal(expanded_track, batch_xs_new),axis=-1), axis=-1)
                                hit_labels = np.sum(np.all(np.equal(expanded_track, batch_xs_new), axis=-1),
                                                    axis=-1)
                        hit_labels = hit_labels.astype('float') / (stations_gone + 1)
                        track_right_pred_nums[stations_gone+1] += labels.sum()
                        track_all_pred_nums[stations_gone+1] += len(labels)
                        track_hit_eff[stations_gone+1] += np.mean(hit_labels)
                        gru_candidates.extend(grus[:, -1])
                        track_labels.extend(labels)
                        track_hit_labels.extend(hit_labels)
                        x_candidates.extend(batch_xs_new[:, -1])
                    chunk_data_xs.append(batch_xs_new)
                    chunk_data_lens.append(np.full(len(batch_xs_new), stations_gone+1))
                if len(chunk_data_xs) > 1:
                    chunk_data_x = np.concatenate(chunk_data_xs, 0)
                    chunk_data_len = np.concatenate(chunk_data_lens, 0)
                else:
                    chunk_data_x = chunk_data_xs[0]
                    chunk_data_len = chunk_data_lens[0]
            recalls_vs_multiplicity[multiplicity].append(np.sum(track_labels) / multiplicity)
            precisions_vs_multiplicity[multiplicity].append(np.sum(track_labels) / len(track_labels))
            hit_eff_vs_multiplicity[multiplicity].append(np.mean(track_hit_labels))
            chunk_data = {'x': {'grus': gru_candidates, 'x': x_candidates},
                          'label': track_labels,
                          'event': chunk.id,
                          'multiplicity': multiplicity}
            chunk.processed_object = chunk_data

        precision_lens = {3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0.}
        recall_lens = {3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0.}
        for stations_gone in precision_lens.keys():
            precision_lens[stations_gone] = track_right_pred_nums[stations_gone] / (track_all_pred_nums[stations_gone] + 1e-6)
            recall_lens[stations_gone] = track_right_pred_nums[stations_gone] / (track_nums[stations_gone] + 1e-6)
            track_hit_eff[stations_gone] = np.mean(track_hit_eff[stations_gone])
        LOGGER.info(f'precision vs len {precision_lens}')
        LOGGER.info(f'recall vs len {recall_lens}')
        LOGGER.info(f'hit accuracy vs len { track_hit_eff}')

        for i in precisions_vs_multiplicity.keys():
            if len(precisions_vs_multiplicity[i]) > 0:
                LOGGER.info(f'precision for mult {i}: {np.mean(precisions_vs_multiplicity[i])}, '
                      f'recall: {np.mean(recalls_vs_multiplicity[i])}, '
                      f'track accuracy: {np.mean(hit_eff_vs_multiplicity[i])}')
        return ProcessedTracknetData(chunks, chunks[0].output_name)


    def save_on_disk(self,
                     processed_data: ProcessedTracknetData):
        train_data_x = []
        train_data_inputs = []
        train_data_labels = []
        train_data_events = []

        valid_data_x = []
        valid_data_inputs = []
        valid_data_labels = []
        valid_data_events = []
        train_chunks = np.random.choice(self.chunks, int(len(self.chunks) * (1-self.valid_size)), replace=False)
        for data_chunk in processed_data.processed_data:
            if data_chunk.processed_object is None:
                continue
            gru = np.stack(data_chunk.processed_object['x']['grus'], 0)
            x = np.stack(data_chunk.processed_object['x']['x'], 0)
            labels = np.array(data_chunk.processed_object['label'])
            events = np.full(len(labels), data_chunk.processed_object['event'])
            if data_chunk.id in train_chunks:
                initial_len = len(data_chunk.processed_object['label'])
                multiplicity = data_chunk.processed_object['multiplicity']
                max_len = int(multiplicity + (initial_len / self.n_times_oversampling))
                to_use = data_chunk.processed_object['label'] | (np.arange(initial_len) < max_len)
                train_data_inputs.extend(gru[to_use])
                train_data_x.extend(x[to_use])
                train_data_labels.extend(labels[to_use])
                train_data_events.extend(events[to_use])
            else:
                valid_data_inputs.extend(gru)
                valid_data_inputs.extend(x)
                valid_data_labels.extend(labels)
                valid_data_events.extend(events)
        np.savez(
            f'{processed_data.output_name}_train',
            grus=train_data_inputs,
            preds=train_data_x,
            labels=train_data_labels,  # predicted right point and point was real
            events=train_data_events
        )
        np.savez(
            f'{processed_data.output_name}_valid',
            grus=valid_data_inputs,
            preds=valid_data_x,
            labels=valid_data_labels,
            events=valid_data_events
        )
        LOGGER.info(f'Saved train hits to: {processed_data.output_name}_train.npz')
        LOGGER.info(f'Saved valid hits to: {processed_data.output_name}_valid.npz')
