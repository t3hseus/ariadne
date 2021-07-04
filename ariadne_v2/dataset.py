import multiprocessing
import os
from contextlib import contextmanager
from typing import Callable, Any, Dict, Union

import h5py

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
                db = self.ds.dataset_path
                hash = Cacher.build_hash(name=df_name, db=db)
                with jit_cacher.instance(self.ds.cacher) as cacher:
                    df = cacher.read_df(hash, db=db)
                self.__dfs[df_name] = df

            return self.__dfs[df_name]

        def update_df(self, df_name, df_update:Callable):
            db = self.ds.dataset_path
            hash = Cacher.build_hash(name=df_name, db=db)
            with jit_cacher.instance(self.ds.cacher) as cacher:
                df = cacher.read_df(hash, db=db)
                df = df_update(df)
                cacher.store_df(hash, df, db)

            self.__dfs[df_name] = df

            return df

        def set_df(self, df_name, df):
            assert self.ds.is_forked or multiprocessing.parent_process() is None, \
                "Direct write to the metainfo from child process is forbidden and error-prone, " \
                "consider using modify_attr"
            db = self.ds.dataset_path
            hash = Cacher.build_hash(name=df_name, db=db, src=self.ds.source)
            with jit_cacher.instance(self.ds.cacher) as cacher:
                df = cacher.store_df(hash, df, db=db)
            self.__dfs[df_name] = df

        def __setitem__(self, key, item):
            assert self.ds.is_forked or multiprocessing.parent_process() is None, \
                "Direct write to the metainfo from child process is forbidden and error-prone, " \
                "consider using modify_attr"

            with jit_cacher.instance(self.ds.cacher) as cacher:
                cacher.store_attr(self.ds.dataset_path, key, item)

            if key in self.__props:
                assert type(item) is type(self.__props[key]), \
                    f"Types are not the same! first {type(item)} second {type(self.__props[key])}"
            self.__props[key] = item

        def __getitem__(self, key):
            if key not in self.__props:
                with jit_cacher.instance(self.ds.cacher) as cacher:
                    value = cacher.read_attr(self.ds.dataset_path, key)
                self.__props[key] = value

            return self.__props[key]

        # atomic update: read-modify-write is synchronized across multiprocessing
        def modify_attr(self, key, update_meth: Callable[[Any], Any]):
            with jit_cacher.instance(self.ds.cacher) as cacher:
                value = cacher.update_attr(self.ds.dataset_path, key, update_meth)
            self.__props[key] = value
            return value

        def refresh_attr(self, key_to_refresh: str):
            with jit_cacher.instance(self.ds.cacher) as cacher:
                value = cacher.read_attr(self.ds.dataset_path, key_to_refresh)
            self.__props[key_to_refresh] = value
            return value

        def refresh_df(self, df_name_to_refresh: str):
            db = self.ds.dataset_path
            hash = Cacher.build_hash(name=df_name_to_refresh, db=db, src=self.ds.source)
            with jit_cacher.instance(self.ds.cacher) as cacher:
                df = cacher.read_df(hash, db=db)
            self.__dfs[df_name_to_refresh] = df
            return df

        def refresh_all(self):
            self.__props = {}
            self.__dfs = {}

        def drop(self, cacher):
            assert self.ds.is_forked or multiprocessing.parent_process() is None, \
                "drop should be run from the main process"
            if not self.ds.db_conn:
                with cacher.handle(self.ds.dataset_path, mode='w') as f:
                    f.flush()

    LEN_KEY = "len"

    def __init__(self, dataset_path: str):
        self.meta = self.KVStorage(self)
        self.dataset_path = dataset_path
        self.temp_dir = None
        self.is_forked = False
        self.cacher = None
        self.db_conn:Union[h5py.File, Any] = None

    @contextmanager
    def create(self, cacher:Cacher, dataset_path=None):
        try:
            yield self.__create(cacher, dataset_path)
        finally:
            self.__close()

    def __create(self, cacher:Cacher, dataset_path):
        if dataset_path is not None:
            self.dataset_path = dataset_path
            self.is_forked = True
            self.cacher = cacher
            self.db_conn = cacher.raw_handle(self.dataset_path, mode='w')
        temp_cache_dir = os.path.join(cacher.cache_path_dir, self.dataset_path)
        self.temp_dir = temp_cache_dir

        os.makedirs(temp_cache_dir, exist_ok=True)
        self.meta.drop(cacher)
        self.meta[self.LEN_KEY] = 0
        return self

    def __close(self):
        if self.cacher:
            self.cacher = None
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None

        self.is_forked = False

    def add(self, key, values: Dict):
        for k, v in values.items():
            self.db_conn.create_dataset(name=f'data/{key}/{k}', data=v, shape=v.shape, compression="gzip")
        self.db_conn.attrs["len"] = self.db_conn.attrs["len"] + 1

    def _submit_local_data(self, prefix):
        pass

    def submit(self, prefix):
        self._submit_local_data(prefix=prefix)
