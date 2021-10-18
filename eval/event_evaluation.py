import os

# TODO: torch multiprocessing everywhere?
# from torch import multiprocessing
import multiprocessing
import sys
import traceback
from timeit import default_timer as timer

from abc import ABCMeta, abstractmethod
from typing import Dict, Callable

import gin
import numpy as np
import pandas as pd

from tqdm import tqdm

from ariadne_v2 import jit_cacher
from ariadne_v2.data_chunk import DFDataChunk
from ariadne_v2.inference import Transformer, IPreprocessor, EventInferrer, IModelLoader
from ariadne_v2.jit_cacher import Cacher
from ariadne_v2.parsing import parse

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


def store_df_from_hash(df, hash):
    with jit_cacher.instance() as cacher:
        cacher.store_df(hash, df)


def read_df_from_hash(hash):
    with jit_cacher.instance() as cacher:
        result_df = cacher.read_df(hash)
        if result_df is not None and not result_df.empty:
            return result_df
    return None


class EventEvaluator:
    def __init__(self,
                 parse_cfg: Dict,
                 global_transformer: Transformer,
                 n_stations: int):

        self.parse_cfg = parse_cfg
        self.loaded_model_state = None
        self.model_loader = None

        self.n_stations = n_stations

        self.global_transformer = global_transformer

        global_lock = multiprocessing.Lock()
        jit_cacher.init_locks(global_lock)

        with jit_cacher.instance() as cacher:
            cacher.init()

        self.data_df_transformed = []
        self.data_df_hashes = []

    def prepare(self, model_loader: IModelLoader):
        assert isinstance(model_loader, IModelLoader)
        for data_df, basename, df_hash in parse(**self.parse_cfg):
            print(f"[prepare]: started processing a df {basename} with {len(data_df)} rows:")
            data_df, hash = self.global_transformer(DFDataChunk.from_df(data_df, df_hash), return_hash=True)
            data_df.rename(columns={'index': 'index_old'}, inplace=True)
            self.data_df_transformed.append(data_df)
            self.data_df_hashes.append(hash)
        print("[prepare] finished")
        print("[prepare] loading your model(s)...")
        self.model_loader = model_loader
        self.loaded_model_state = self.model_loader()
        print("[prepare] finished loading your model(s)...")
        return self.data_df_transformed

    def _hash_run_model(self, funcs):
        assert len(self.data_df_hashes) > 0 and self.loaded_model_state, "call prepare() first"
        return Cacher.build_hash_callable(funcs, *self.data_df_hashes, hits='hits', **self.loaded_model_state[0])

    def _hash_run_model_events(self, funcs):
        assert len(self.data_df_hashes) > 0 and self.loaded_model_state, "call prepare() first"
        return Cacher.build_hash_callable(funcs, *self.data_df_hashes, events='events', **self.loaded_model_state[0])

    def _hash_all_tracks_df(self):
        assert len(self.data_df_hashes) > 0 and self.loaded_model_state, "call prepare() first"
        return Cacher.build_hash(*self.data_df_hashes, all_tracks_df='__all_tracks_df')

    def _hash_all_events_df(self):
        assert len(self.data_df_hashes) > 0 and self.loaded_model_state, "call prepare() first"
        return Cacher.build_hash(*self.data_df_hashes, all_events_df='__all_events_df')

    ## todo: typing, batching for preprocessing
    def run_model(self, model_preprocess_func, model_run_func):
        assert len(self.data_df_transformed) > 0 and self.loaded_model_state, "call prepare() first"
        print('[run model] start')
        COLUMNS = ['event_id', 'track_pred', 'px', 'py', 'pz'] + [f"hit_id_{n}" for n in range(self.n_stations)]
        result_df_arr = []
        result_event_arr = []
        run_model_hash = self._hash_run_model([self.model_loader, model_run_func, model_preprocess_func])
        result_df = read_df_from_hash(run_model_hash)
        events_hash = self._hash_run_model_events([self.model_loader, model_run_func, model_preprocess_func])
        result_event_df = read_df_from_hash(events_hash)
        if result_df is not None and result_event_df is not None:
            print('[run model] cache hit, finish')
            return result_df, result_event_df

        print('\n')
        for events in self.data_df_transformed:
            with tqdm(total=events.event.nunique(), file=sys.stdout) as pbar:
                for ev_id, event_df in events.groupby('event'):
                    pbar.set_description('processed: %d' % ev_id)
                    pbar.update(1)
                    cpu_time_for_event = 0.0
                    gpu_time_for_event = 0.0
                    try:
                        start = timer()
                        preprocess_result = model_preprocess_func(event_df)
                        end = timer()
                        cpu_time_for_event = end - start

                        if preprocess_result is None:
                            continue
                    except KeyboardInterrupt as ex:
                        break
                    except:
                        error_message = traceback.format_exc()

                        print(f"got exception for preprocessing:\n message={error_message} \n\
                                            on \nevent_id={ev_id}")
                        continue

                    try:
                        start = timer()
                        model_run_df = model_run_func(preprocess_result, self.loaded_model_state[1])
                        end = timer()
                        gpu_time_for_event = end - start

                    except KeyboardInterrupt as ex:
                        break
                    except:
                        error_message = traceback.format_exc()
                        print(f"got exception for model run:\n message={error_message} \n\
                                           on \nevent_id={ev_id}")
                        continue

                    model_run_df['event_id'] = ev_id
                    tracks = event_df[event_df.track != -1]
                    model_run_df['px'] = tracks.px.min()
                    model_run_df['py'] = tracks.py.min()
                    model_run_df['pz'] = tracks.pz.min()

                    result_event_arr.append(pd.DataFrame({
                        'event_id': ev_id,
                        'cpu_time': cpu_time_for_event,
                        'gpu_time': gpu_time_for_event,
                        'multiplicity': tracks.track.nunique()}, index=[0]))
                    model_run_df = model_run_df[COLUMNS]
                    result_df_arr.append(model_run_df)

        result_df = pd.concat(result_df_arr, ignore_index=True)
        result_event_df = pd.concat(result_event_arr, ignore_index=True)

        store_df_from_hash(result_df, run_model_hash)
        store_df_from_hash(result_event_df, events_hash)
        print('[run model] cache miss, finish')
        return result_df, result_event_df

    def build_all_tracks(self):
        print('[build_all_tracks] start')
        all_tracks_df = read_df_from_hash(self._hash_all_tracks_df())
        all_events_df = read_df_from_hash(self._hash_all_events_df())
        if all_events_df is not None and all_tracks_df is not None:
            print('[build_all_tracks] cache hit, finish')
            return all_tracks_df, all_events_df

        STATION_COLUMNS = [f"hit_id_{n}" for n in range(self.n_stations)]

        # COLUMNS_DF = ['event', 'track', 'px', 'py', 'pz', 'pred', 'multiplicity'] + STATION_COLUMNS

        true_tracks_arr = []
        all_events_arr = []
        print('\n')
        for events in self.data_df_transformed:
            with tqdm(total=events.event.nunique(), file=sys.stdout) as pbar:
                for ev_id, event in events.groupby('event'):
                    pbar.set_description('processed: %d' % ev_id)
                    pbar.update(1)

                    if event.empty:
                        continue
                    ev_id_real = event.event.values[0]

                    px_min_general = event[event.track != -1].px.min()
                    py_min_general = event[event.track != -1].py.min()
                    pz_min_general = event[event.track != -1].pz.min()
                    hits_in_event = set()
                    tracks_in_event = event[event.track != -1].track.nunique()

                    for tr_id, track in event.groupby('track'):
                        if tr_id != -1:
                            local_index_values = track.index.values
                            global_index_values = track.index_old.values

                            assert len(local_index_values) >= 3, f"track len <3 for event {idx} tr_id {tr_id}"
                            px_py_pz = track[['px', 'py', 'pz']].values[0]
                            hits_in_event.update(global_index_values)

                            new_dict = {
                                'event_id': int(ev_id_real),
                                'track': int(tr_id),
                                'px': px_py_pz[0],
                                'py': px_py_pz[1],
                                'pz': px_py_pz[2],
                                'pred': int(0),
                            }
                            for station_id, col in enumerate(STATION_COLUMNS):
                                new_dict[col] = -1

                            for idx, index_val in enumerate(local_index_values):
                                station_id = track.loc[index_val].station
                                new_dict[f"hit_id_{int(station_id)}"] = global_index_values[idx]

                            true_tracks_arr.append(pd.DataFrame(new_dict, index=[0]))

                    all_events_arr.append(pd.DataFrame({
                        'event_id': int(ev_id_real),
                        'multiplicity': int(tracks_in_event),
                        'pred': 0,
                        'time': 0,
                        'px_min': px_min_general,
                        'py_min': py_min_general,
                        'pz_min': pz_min_general,
                    }, index=[0]))

        all_tracks_df = pd.concat(true_tracks_arr, ignore_index=True)
        all_events_df = pd.concat(all_events_arr, ignore_index=True)

        store_df_from_hash(all_tracks_df, self._hash_all_tracks_df())
        store_df_from_hash(all_events_df, self._hash_all_events_df())
        print('[build_all_tracks] cache miss, finish')
        return all_tracks_df, all_events_df

    def solve_results(self, model_results, all_data):
        print('[solve results] start')
        STATION_COLUMNS = [f"hit_id_{n}" for n in range(self.n_stations)]

        all_tracks, all_events = all_data
        reco_tracks, reco_events = model_results

        # TODO: how to solve ghost hits?
        tracks_pred = reco_tracks[reco_tracks.track_pred]
        reco_tracks_impulses = tracks_pred[['px', 'py', 'pz']]
        reco_tracks_preds = tracks_pred[['event_id', 'track_pred'] + STATION_COLUMNS]
        reco_tracks_preds['idx_old'] = tracks_pred.index

        results = pd.merge(all_tracks, reco_tracks_preds, how='outer',
                           on=['event_id'] + STATION_COLUMNS)

        not_found_tracks = (results.track_pred != False) & (results.track_pred != True)
        results.loc[not_found_tracks, 'track_pred'] = False

        results.loc[results.track_pred, ['pred']] = 1

        ghosts_idx_all = pd.isna(results.track)
        ghosts_idx_reco = results[ghosts_idx_all].idx_old.astype('int')
        ghosts_impulses = reco_tracks_impulses.loc[ghosts_idx_reco]
        results.loc[ghosts_idx_all, ['px', 'py', 'pz']] = ghosts_impulses[['px', 'py', 'pz']].values
        results.loc[ghosts_idx_all, 'track'] = -1
        results.loc[ghosts_idx_all, 'pred'] = -1
        results = results.drop(['track_pred', 'idx_old'], axis=1)
        results['pred'] = results['pred'].astype('int')
        results['track'] = results['track'].astype('int')

        print('[solve results] finish')
        print('[solve results] final stats:')
        print('=' * 10 + 'EVALUATION RESULTS' + '=' * 10)
        print(f'Total events evaluated: {results.event_id.nunique()}')

        all_tracks = results[results.pred != -1]
        print(f'Total tracks evaluated: {len(all_tracks)}')

        true_tracks = results[results.pred == 1]
        efficiency = len(true_tracks) / float(len(all_tracks))
        print(f'Track Efficiency (recall): {efficiency:.4f} ')

        reco_tracks = results[results.pred != 0]
        true_tracks = results[results.pred == 1]
        purity = len(true_tracks) / float(len(reco_tracks))
        print(f'Track Purity (precision): {purity:.4f} ')

        all_tracks = results[results.pred != -1]
        true_unique = all_tracks[['track', 'event_id']].groupby('event_id').nunique()

        reco_tracks = results[results.pred == 1]
        reco_unique = reco_tracks[['track', 'event_id']].groupby('event_id').nunique()

        all_events = pd.merge(true_unique, reco_unique, on='event_id', how='outer')

        missing_events = pd.isna(all_events.track_y)
        all_events.loc[missing_events, 'track_y'] = -1
        all_events.track_x = all_events.track_x.astype('int')
        all_events.track_y = all_events.track_y.astype('int')

        event_efficiency = len(all_events[all_events.track_x == all_events.track_y]) / len(all_events)
        print(f'Fully reconstructed event ratio: {event_efficiency:.4f}')

        print(f'Mean cpu time per event: {reco_events["cpu_time"].mean():.4f} sec'
              f' ({1. / reco_events["cpu_time"].mean():.2f} events per second) ')

        print(f'Mean gpu time per event: {reco_events["gpu_time"].mean():.4f} sec'
              f' ({1. / reco_events["gpu_time"].mean():.2f} events per second) ')
        print('=' * 10 + 'EVALUATION RESULTS' + '=' * 10)
        return results, reco_events
