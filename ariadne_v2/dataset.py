import multiprocessing
import os
from typing import Callable, Any, Dict

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
                hash = Cacher.build_hash(name=df_name, db=db, src=self.ds.source)
                with jit_cacher.instance() as cacher:
                    df = cacher.read_df(hash, db=db)
                self.__dfs[df_name] = df

            return self.__dfs[df_name]

        def update_df(self, df_name, df_update:Callable):
            db = self.ds.dataset_path
            hash = Cacher.build_hash(name=df_name, db=db, src=self.ds.source)
            with jit_cacher.instance() as cacher:
                df = cacher.read_df(hash, db=db)
                df = df_update(df)
                cacher.store_df(hash, db, df)

            self.__dfs[df_name] = df

            return df

        def set_df(self, df_name, df):
            assert multiprocessing.parent_process() is None, \
                "Direct write to the metainfo from child process is forbidden and error-prone, " \
                "consider using modify_attr"
            db = self.ds.dataset_path
            hash = Cacher.build_hash(name=df_name, db=db, src=self.ds.source)
            with jit_cacher.instance() as cacher:
                df = cacher.store_df(hash, df, db=db)
            self.__dfs[df_name] = df

        def __setitem__(self, key, item):
            assert multiprocessing.parent_process() is None, \
                "Direct write to the metainfo from child process is forbidden and error-prone, " \
                "consider using modify_attr"

            with jit_cacher.instance() as cacher:
                cacher.store_attr(self.ds.dataset_path, key, item)

            if key in self.__props:
                assert type(item) is type(self.__props[key]), \
                    f"Types are not the same! first {type(item)} second {type(self.__props[key])}"
            self.__props[key] = item

        def __getitem__(self, key):
            if key not in self.__props:
                with jit_cacher.instance() as cacher:
                    value = cacher.read_attr(self.ds.dataset_path, key)
                self.__props[key] = value

            return self.__props[key]

        # atomic update: read-modify-write is synchronized across multiprocessing
        def modify_attr(self, key, update_meth: Callable[[Any], Any]):
            with jit_cacher.instance() as cacher:
                value = cacher.update_attr(self.ds.dataset_path, key, update_meth)
            self.__props[key] = value
            return value

        def refresh_attr(self, key_to_refresh: str):
            with jit_cacher.instance() as cacher:
                value = cacher.read_attr(self.ds.dataset_path, key_to_refresh)
            self.__props[key_to_refresh] = value
            return value

        def refresh_df(self, df_name_to_refresh: str):
            db = self.ds.dataset_path
            hash = Cacher.build_hash(name=df_name_to_refresh, db=db, src=self.ds.source)
            with jit_cacher.instance() as cacher:
                df = cacher.read_df(hash, db=db)
            self.__dfs[df_name_to_refresh] = df
            return df

        def refresh_all(self):
            self.__props = {}
            self.__dfs = {}

        def drop(self, cacher):
            assert multiprocessing.parent_process() is None, \
                "drop must be run from the main process"

            with cacher.handle(self.ds.dataset_path, mode='w') as f:
                f.flush()

    LEN_KEY = "len"

    def __init__(self, dataset_path: str):
        self.meta = self.KVStorage(self)
        self.dataset_path = dataset_path

    def create(self, cacher:Cacher):
        os.makedirs(os.path.join(cacher.cache_path_dir, self.dataset_path), exist_ok=True)
        self.meta.drop(cacher)
        self.meta[self.LEN_KEY] = 0

    def add(self, key, values: Dict):
        with jit_cacher.instance() as cacher:
            with cacher.handle(self.dataset_path, mode='a') as db:
                for k, v in values.items():
                    db.create_dataset(name=f'data/{key}/{k}', data=v, shape=v.shape, compression="gzip")
        self.meta.modify_attr("len", lambda len_old: len_old + 1)

    def _submit_local_data(self, cacher: Cacher):
        pass

    def submit(self):
        with jit_cacher.instance() as cacher:
            self._submit_local_data(cacher)
