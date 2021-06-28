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

from ariadne.utils import *
from ariadne.tracknet_v2_1.processor import TrackNetV21Processor, ProcessedTracknetDataChunk, ProcessedTracknetData, TracknetDataChunk

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
                 min_len: int= 4,
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
        self.max_n_stations = len(self.stations_sizes)
        #self.z_values = [15.229, 18.499, 21.604, 39.702, 64.535, 112.649, 135.330, 160.6635, 183.668]
        self.z_values = [12.344, 15.614, 24.499, 39.702, 64.535, 112.649, 135.330, 160.6635, 183.668]
        self.z_values = self.z_values[9-self.max_n_stations:]
        self.det_indices = [0, 1]
        self.min_len = min_len



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

    def initialize_all(self):
        self.track_right_pred_nums = {i: 0 for i in range(3, self.max_n_stations + 1)}
        self.track_all_pred_nums = {i: 0 for i in range(3, self.max_n_stations + 1)}
        self.track_count_vs_len = {i: 0 for i in range(3, self.max_n_stations + 1)}
        self.track_precision_vs_len_event = {i: [] for i in range(3, self.max_n_stations + 1)}
        self.track_recall_vs_len_event = {i: [] for i in range(3, self.max_n_stations + 1)}
        self.track_hit_eff = {i: 0 for i in range(3, self.max_n_stations + 1)}
        self.precisions_vs_multiplicity = {i: [] for i in range(100)}
        self.recalls_vs_multiplicity = {i: [] for i in range(100)}
        self.hit_eff_vs_multiplicity = {i: [] for i in range(100)}
        self.recalls = []
        self.candidate_counts = []
        self.candidate_3_counts = []
        self.multiplicities = []
        self.num_hits_in_events = []


    def get_tracks(self, df):
        tracks = df[df['track'] != -1].groupby('track')
        multiplicity = tracks.ngroups
        self.multiplicities.append(multiplicity)
        self.num_hits_in_events.append(len(df))
        tracks_vs_len = {3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        all_tracks = []
        for i, track in tracks:
            temp_track = track[['x', 'y', 'z']].values
            if len(temp_track) >= self.min_len:
                tracks_vs_len[len(temp_track)].append(temp_track)
                self.track_count_vs_len[len(temp_track)] += 1
                all_tracks.append(temp_track)
        for stations_in_track, this_track_list in tracks_vs_len.items():
            if len(this_track_list) > 0:
                tracks_vs_len[stations_in_track] = np.stack(this_track_list, 0)
        return tracks_vs_len, multiplicity


    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedTracknetData:
        self.initialize_all()
        for chunk in chunks:
            torch.cuda.empty_cache()
            if chunk.processed_object is None:
                continue
            if chunk.id == 343 or chunk.id == 560 or chunk.id == 621 or chunk.id == 675:
                chunk.processed_object = None
                continue
            LOGGER.info(f'=====> id {chunk.id}')
            df = chunk.processed_object
            tracks_vs_len, multiplicity = self.get_tracks(df)
            if multiplicity == 0:
                LOGGER.warning(f'Multiplicity in chunk #{chunk.id} is 0! This chunk is skipped')
                chunk.processed_object = None
                continue
            stations = df.groupby('station').size().max()
            #max_station = max(stations)
            if stations > 1000:
                chunk.processed_object = None
                continue
            chunk_data_x = get_seeds(df)
            self.candidate_counts.append(len(chunk_data_x))
            chunk_data_len = np.full(len(chunk_data_x), 2)
            chunk_data_event = np.full(len(chunk_data_x), chunk.id)
            event_hits = np.ascontiguousarray(df[['x', 'y', 'z']].values)
            event_hits[:, -1] *= 1000.
            event_indexes = {}
            for z_value in np.unique(event_hits[:,2]):
                event_indexes[z_value] = store_in_index(event_hits[event_hits[:,2] == z_value])  # z is last!
            gru_candidates = []
            candidate_labels = []
            x_candidates = []
            max_batch_size = 256
            for stations_gone in range(2, self.max_n_stations):
                num_batches = int(len(chunk_data_x) / max_batch_size) + 1
                num_right_batches = 0
                num_all_batches = 0
                next_stage_xs = []
                next_stage_lens = []
                if len(chunk_data_x) == 0:
                    LOGGER.info('Have zero ellipces on this station! Skipping other stations')
                    break
                print(num_batches)
                labels_for_empty_el_batch = []
                labels_for_full_el_batch = []
                for batch_num in range(num_batches):
                    min_i = max_batch_size*batch_num
                    max_i = min(max_batch_size + min_i, len(chunk_data_x))
                    if min_i==max_i:
                        LOGGER.info('Have zero ellipces on this station! Skipping other stations')
                        break
                    this_batch_x =torch.tensor( chunk_data_x[min_i: max_i]).to(self.device)
                    this_batch_len = chunk_data_len[min_i: max_i]
                    batch_prediction, batch_gru = self.model(this_batch_x,
                                                             torch.tensor(this_batch_len, dtype=torch.int64).to(self.device),
                                                             return_gru_states=True)
                    new_pred = torch.full((len(batch_prediction), 5), self.z_values[stations_gone]*1000., device=self.device)
                    new_pred[:, :2] = batch_prediction[:, -1, :2]
                    new_pred[:, 3:] = batch_prediction[:, -1, 2:]
                    batch_prediction = new_pred
                    #print(self.z_values[stations_gone]*1000)
                    this_station_hits = event_hits[event_hits[:, 2] == self.z_values[stations_gone] * 1000.]
                    if batch_num == 0:
                        print(f'station {stations_gone+1}, on this station: {len(this_station_hits)} hits' )
                    if len(this_station_hits) == 0:
                        LOGGER.info('Have zero hits on this station! Skipping other stations')
                        break
                   #nearest_hits_index = find_nearest_hits_in_index(batch_prediction.detach().cpu().numpy()[:, :3],
                    #                                     event_indexes[self.z_values[stations_gone]*1000],
                    #                                     60)
                    #nearest_hits = this_station_hits[nearest_hits_index]

                    nearest_hits, in_ellipse_mask = find_hits_in_ellipse(batch_prediction, torch.tensor(this_station_hits, device=batch_prediction.device), return_numpy=False)
                    #print('ok')
                    #nearest_hits, in_ellipse_mask = filter_hits_in_ellipses(batch_prediction.cpu().detach().numpy(),
                    #                                                   nearest_hits,
                    #                                                  nearest_hits_index,
                    #                                                   filter_station=True,
                    #                                                   z_last=True,
                    #                                                  find_n=nearest_hits_index.shape[1])
#
                    nearest_hits[:, :, -1] /= 1000.
                    station_mask = (nearest_hits[:, :, -1] == self.z_values[stations_gone])
                    nearest_hits_mask = in_ellipse_mask #* (station_mask)
                    # here empty ellipses and all inputs for them are saved
                    empty_ellipses_mask = (nearest_hits_mask.sum(dim=-1) == 0) # if nothing is found in ellipse
                    empty_ellipses = batch_prediction[empty_ellipses_mask]#.detach().cpu().numpy()
                    empty_xs = this_batch_x[empty_ellipses_mask]
                    empty_grus = batch_gru[empty_ellipses_mask]#.detach().cpu().numpy()
                    #here filtered candidates are prolonged and all_data is saved
                    #prolonged_batch_xs_1, prolonged_grus_0 = prolong(this_batch_x.detach().cpu().numpy(),
                    #                                                  batch_gru,
                    #                                                  nearest_hits_mask.detach().cpu().numpy(),
                    #                                                  nearest_hits.detach().cpu().numpy(),
                    #                                                  stations_gone)
                    prolonged_batch_xs, prolonged_grus = prolong(this_batch_x,
                                                                 batch_gru,
                                                                 nearest_hits_mask,
                                                                 nearest_hits,
                                                                 stations_gone,
                                                                 use_gpu=True)
                    next_stage_xs.append(prolonged_batch_xs)
                    next_stage_lens.append(np.full(len(prolonged_batch_xs), stations_gone + 1))
                    if stations_gone > 3:
                         use_empty_ellipses = []
                         empty_ellipses_station_intersections = []
                         orig_ellipses = torch.zeros((len(empty_ellipses), 4))
                         orig_ellipses[:, :2] = empty_ellipses[:, :2]
                         orig_ellipses[:, 2:] = empty_ellipses[:, 3:]
                         #are_ellipses_on_station = is_ellipse_intersects_station_tensor(orig_ellipses,
                         #                                                               self.stations_sizes[stations_gone].tolist())
                         #use_empty_ellipses.append(True)  # вот тут я убрала фильтр по станции!
                         use_empty_ellipses = torch.ones(len(orig_ellipses), dtype=torch.bool)
                         empty_xs = empty_xs[use_empty_ellipses]
                         empty_grus = empty_grus[use_empty_ellipses]
                         if len(empty_xs) > 0:
                             labels_for_empty_ellipses = get_labels(torch.tensor(tracks_vs_len[stations_gone],
                                                                                 device=self.device),
                                                                    empty_xs,
                                                                    use_gpu=True)
                             labels_for_empty_el_batch.append(labels_for_empty_ellipses)
                             gru_candidates.extend(empty_grus[:, -2].detach().cpu().numpy())  # because we don't need gru for empty ellipse
                             candidate_labels.extend(labels_for_empty_ellipses)
                             x_candidates.extend(empty_xs[:, -1].detach().cpu().numpy())
                             del empty_xs

                    if (stations_gone == (self.max_n_stations-1)): # if we are predicting for last station
                        if len(prolonged_batch_xs) > 0:  # now we prolong candidates with all hits *in* ellipses
                            labels_for_ellipses = get_labels(torch.tensor(tracks_vs_len[stations_gone + 1],
                                                                          device=self.device),
                                                             prolonged_batch_xs,
                                                             use_gpu=True)
                            labels_for_full_el_batch.append(labels_for_ellipses)
                            gru_candidates.extend(prolonged_grus[:, -1])  # because we need gru for predicted ellipse
                            candidate_labels.extend(labels_for_ellipses)
                            x_candidates.extend(prolonged_batch_xs[:, -1])
                            del prolonged_batch_xs

                if len(labels_for_empty_el_batch) > 0:
                    labels_for_empty_el_batch = np.concatenate(labels_for_empty_el_batch)
                    self.track_right_pred_nums[stations_gone], self.track_all_pred_nums[stations_gone] = \
                        compute_metrics(tracks_vs_len[stations_gone],
                                        labels_for_empty_el_batch,
                                        self.track_right_pred_nums[stations_gone],
                                        self.track_all_pred_nums[stations_gone],
                                        self.track_precision_vs_len_event[stations_gone],
                                        self.track_recall_vs_len_event[stations_gone])
                if len(labels_for_full_el_batch) > 0:
                    labels_for_full_el_batch = np.concatenate(labels_for_full_el_batch)
                    self.track_right_pred_nums[stations_gone+1], self.track_all_pred_nums[stations_gone+1] = compute_metrics(
                        tracks_vs_len[stations_gone+1],
                        labels_for_full_el_batch,
                        self.track_right_pred_nums[stations_gone+1],
                        self.track_all_pred_nums[stations_gone+1],
                        self.track_precision_vs_len_event[stations_gone+1],
                        self.track_recall_vs_len_event[stations_gone+1])
                if len(next_stage_xs) > 1:
                    chunk_data_x = np.concatenate(next_stage_xs, 0)
                    chunk_data_len = np.concatenate(next_stage_lens, 0)
                else:
                    try:
                        chunk_data_x = next_stage_xs[0]
                        chunk_data_len = next_stage_lens[0]
                    except:
                        continue
            print('recall in this event is: ', np.sum(candidate_labels)/multiplicity)
            self.recalls.append(np.sum(candidate_labels)/multiplicity)
            print(np.sum(candidate_labels), multiplicity)
            print('Running mean recall is: ', np.mean(self.recalls))
            print(self.track_precision_vs_len_event)
            print(self.track_recall_vs_len_event)
            print(self.track_right_pred_nums)
            print(self.track_all_pred_nums)
            self.recalls_vs_multiplicity[multiplicity].append(np.sum(candidate_labels) / multiplicity)
            self.precisions_vs_multiplicity[multiplicity].append(np.sum(candidate_labels) / len(candidate_labels))
            chunk_data = {'x': {'grus': gru_candidates, 'x': x_candidates},
                          'label': candidate_labels,
                          'event': chunk.id,
                          'multiplicity': multiplicity}
            chunk.processed_object = chunk_data
        precision_lens = {}
        recall_lens = {}
        print(self.track_precision_vs_len_event)
        print(self.track_recall_vs_len_event)
        print(self.track_right_pred_nums)
        print(self.track_all_pred_nums)
        print(self.track_count_vs_len)
        for stations_gone in self.track_right_pred_nums.keys():
            if self.track_all_pred_nums[stations_gone] > 0:
                precision_lens[stations_gone] = self.track_right_pred_nums[stations_gone] / (self.track_all_pred_nums[stations_gone] + 1e-6)
                recall_lens[stations_gone] = self.track_right_pred_nums[stations_gone] / (self.track_count_vs_len[stations_gone] + 1e-6)
                self.track_precision_vs_len_event[stations_gone] = np.mean(self.track_precision_vs_len_event[stations_gone])
                self.track_recall_vs_len_event[stations_gone] = np.mean(self.track_recall_vs_len_event[stations_gone])
        LOGGER.info(f'precision vs len {precision_lens}')
        LOGGER.info(f'recall vs len {recall_lens}')
        LOGGER.info(f'precision event-wise vs len {self.track_precision_vs_len_event}')
        LOGGER.info(f'recall event-wise vs len {self.track_recall_vs_len_event}')
        LOGGER.info(f'hit accuracy vs len { self.track_hit_eff}')
        for i in self.precisions_vs_multiplicity.keys():
            if len(self.precisions_vs_multiplicity[i]) > 0:
                LOGGER.info(f'Mean precision for mult {i}: {np.mean(self.precisions_vs_multiplicity[i])}, '
                      f'Mean recall: {np.mean(self.recalls_vs_multiplicity[i])}')
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
            if data_chunk.processed_object is None or len(data_chunk.processed_object['x']['grus']) == 0:
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
                valid_data_x.extend(x)
                valid_data_labels.extend(labels)
                valid_data_events.extend(events)

        train_data_inputs = np.stack(train_data_inputs, 0)
        train_data_events = np.array(train_data_events)
        train_data_x = np.stack(train_data_x, 0)
        train_data_labels = np.array(train_data_labels)

        valid_data_inputs = np.stack(valid_data_inputs, 0)
        valid_data_events = np.array(valid_data_events)
        valid_data_x = np.stack(valid_data_x, 0)
        valid_data_labels = np.array(valid_data_labels)
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
