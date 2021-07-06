import multiprocessing
import os
from contextlib import contextmanager
from typing import Callable, Any, Dict, Union, List

import h5py
import numpy as np
import pandas as pd

from ariadne_v2 import jit_cacher
from ariadne_v2.jit_cacher import Cacher


class AriadneDataset(object):
    class KVStorage:
        def __init__(self, ds):
            self.ds = ds
            self.__props = {}
            self.__dfs = {}

        def get_df(self, df_name):
            if df_name not in self.__dfs:
                db = self.ds.dataset_name
                hash = Cacher.build_hash(name=df_name, db=db)
                with jit_cacher.instance(self.ds.cacher) as cacher:
                    df = cacher.read_df(hash, db=db)
                self.__dfs[df_name] = df

            return self.__dfs[df_name]

        def update_df(self, df_name, df_update: Callable[[Any, Cacher], pd.DataFrame]):
            db = self.ds.dataset_name
            hash = Cacher.build_hash(name=df_name, db=db)
            with jit_cacher.instance(self.ds.cacher) as cacher:
                df = cacher.read_df(hash, db=db)
                df = df_update(df, cacher)
                cacher.store_df(hash, df, db)

            self.__dfs[df_name] = df

            return df

        def set_df(self, df_name, df):
            assert self.ds.connected or multiprocessing.parent_process() is None, \
                "Direct write to the metainfo from child process is forbidden and error-prone, " \
                "consider using modify_attr"
            db = self.ds.dataset_name
            hash = Cacher.build_hash(name=df_name, db=db)
            with jit_cacher.instance(self.ds.cacher) as cacher:
                cacher.store_df(hash, df, db=db)
            self.__dfs[df_name] = df

        def __setitem__(self, key, item):
            assert self.ds.connected or multiprocessing.parent_process() is None, \
                "Direct write to the metainfo from child process is forbidden and error-prone, " \
                "consider using modify_attr"

            with jit_cacher.instance(self.ds.cacher) as cacher:
                cacher.store_attr(self.ds.dataset_name, key, item)

            if key in self.__props:
                assert type(item) is type(self.__props[key]), \
                    f"Types are not the same! first {type(item)} second {type(self.__props[key])}"
            self.__props[key] = item

        def __getitem__(self, key):
            if key not in self.__props:
                with jit_cacher.instance(self.ds.cacher) as cacher:
                    value = cacher.read_attr(self.ds.dataset_name, key)
                self.__props[key] = value

            return self.__props[key]

        # atomic update: read-modify-write is synchronized across multiprocessing
        def modify_attr(self, key, update_meth: Callable[[Any], Any]):
            with jit_cacher.instance(self.ds.cacher) as cacher:
                value = cacher.update_attr(self.ds.dataset_name, key, update_meth)
            self.__props[key] = value
            return value

        def refresh_attr(self, key_to_refresh: str):
            with jit_cacher.instance(self.ds.cacher) as cacher:
                value = cacher.read_attr(self.ds.dataset_name, key_to_refresh)
            self.__props[key_to_refresh] = value
            return value

        def refresh_df(self, df_name_to_refresh: str):
            db = self.ds.dataset_name
            hash = Cacher.build_hash(name=df_name_to_refresh, db=db)
            with jit_cacher.instance(self.ds.cacher) as cacher:
                df = cacher.read_df(hash, db=db)
            self.__dfs[df_name_to_refresh] = df
            return df

        def refresh_all(self):
            self.__props = {}
            self.__dfs = {}
            if self.ds.connected:
                self.ds.db_conn.flush()

        def drop(self, cacher):
            assert self.ds.connected or multiprocessing.parent_process() is None, \
                "drop should be run from the main process"
            if not self.ds.db_conn:
                with cacher.handle(self.ds.dataset_name, mode='w') as f:
                    f.flush()

    LEN_KEY = "len"
    REFS_KEY = "refs"

    def __init__(self, dataset_name: str):
        self.meta = self.KVStorage(self)
        self.dataset_name = dataset_name
        self.connected = False
        self.cacher = None
        self.db_conn: Union[h5py.File, Any] = None

    @contextmanager
    def open_dataset(self, cacher: Cacher, dataset_path=None, drop_old=True):
        try:
            self.connect(cacher, dataset_path, drop_old)
            yield self
        finally:
            self.disconnect()

    def connect(self, cacher: Cacher, dataset_path: str, drop_old: bool, mode: str = None):
        # unix fork issue: erase dicts allocated from the parent process
        self.meta.refresh_all()

        if dataset_path is not None:
            temp_cache_dir = os.path.join(cacher.cache_path_dir, dataset_path)
            os.makedirs(temp_cache_dir, exist_ok=True)
            self.dataset_name = dataset_path
            self.connected = True
            self.cacher = cacher
            mode = mode if mode is not None else 'w' if drop_old else 'a'
            self.db_conn = cacher.raw_handle(dataset_path, mode=mode)
        else:
            temp_cache_dir = os.path.join(cacher.cache_path_dir, self.dataset_name)
            os.makedirs(temp_cache_dir, exist_ok=True)

        if drop_old:
            self.meta.drop(cacher)
            self.meta[self.LEN_KEY] = 0
            self.meta[self.REFS_KEY] = [self.dataset_name]

    def disconnect(self):
        if self.cacher:
            self.cacher = None
        if self.db_conn:
            self.db_conn.flush()
            self.db_conn.close()
            self.db_conn = None
        self.connected = False

    def get(self, key):
        return self.db_conn[f'{self.REFS_KEY}/{key}']

    def add(self, key, values: Dict):
        for k, v in values.items():
            self.db_conn.create_dataset(name=f'data/{key}/{k}', data=v, shape=v.shape, compression="gzip")
        self.db_conn.attrs[self.LEN_KEY] = self.db_conn.attrs["len"] + 1

    def add_dataset_reference(self, other_ds_path):
        valid = other_ds_path not in self.db_conn.attrs.keys()
        assert valid, "double reference add"

        refs = self.db_conn.attrs[self.REFS_KEY]
        self.db_conn.attrs[self.REFS_KEY] = np.append(refs, [other_ds_path])
        self.db_conn[f'{self.REFS_KEY}/{other_ds_path}'] = h5py.ExternalLink(
            filename=self.cacher.to_db_path(other_ds_path),
            path='/')

    def _submit_local_data(self):
        pass

    def _gather_local_data(self, datasets: List[str]):
        pass

    def local_submit(self):
        self._submit_local_data()

    def global_submit(self, datasets: List[str]):
        self._gather_local_data(datasets)
        self.db_conn.attrs[self.LEN_KEY] = self.db_conn.attrs[self.LEN_KEY] + \
                                           sum([self.db_conn[self.REFS_KEY][ds_name].attrs['len']
                                                for ds_name in datasets])
        self.meta.refresh_all()
