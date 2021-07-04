import atexit
import functools
import hashlib
import logging
import os
import datetime
import time
from contextlib import contextmanager

from typing import Dict, Callable, Any, Union

import subprocess

import pandas as pd
import collections
import multiprocessing

import h5py

from ariadne_v2.data_chunk import DataChunk

LOGGER = logging.getLogger('ariadne.cache')


class CACHE_DATA_TYPES:
    Object = 'Object'
    Dataframe = 'Dataframe'
    Numpy = 'Numpy'


# God bless python..
def is_hashable(obj):
    return isinstance(obj, collections.Hashable)

class Cacher():
    # bump this version if you change the code, otherwise cache might be wrong
    VERSION = 1
    CACHE_INFO_FILE = 'cache_info.txt'
    CACHE_DB_FILE = 'ariadne_cache_db.h5'

    COLUMNS = ['hash', 'date', 'key', 'data_path', 'type', 'commit']

    def __init__(self,
                 cache_path='.jit',
                 df_compression_level=5,
                 with_stats=True):
        self.cache_path_dir = cache_path
        self.cache_info_path = os.path.join(self.cache_path_dir, self.CACHE_INFO_FILE)
        os.makedirs(self.cache_path_dir, exist_ok=True)
        self.df_compression_level = df_compression_level
        self.cache_db_path = os.path.join(self.cache_path_dir, self.CACHE_DB_FILE)
        self.with_stats = with_stats
        self.cache = None
        self.cached_data = {}

        try:
            self.rev = self.get_git_revision_hash()
        except Exception as ex:
            LOGGER.error('Exception when trying to get git hash... bad!')
            self.rev = 'no revision'

    def init(self):
        self._build_cache()

    @staticmethod
    def get_git_revision_hash():
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

    @staticmethod
    def generate_unique_key(*args, **kwargs) -> str:
        hashed_args = [('%r' % arg) for arg in args]
        hashed_kwargs = [('%r%r' % ((key, value))) for (key, value) in kwargs.items()]
        return hashlib.md5(':'.join(hashed_args + hashed_kwargs).encode('utf-8')).hexdigest()

    def _save(self):
        self.cache.to_csv(self.cache_info_path, sep='\t', encoding='utf-8', mode='w', header=False)

    def _build_cache(self):
        if os.path.exists(self.cache_info_path):
            self.cache = pd.read_csv(self.cache_info_path, names=self.COLUMNS, sep='\t', encoding='utf-8')
        else:
            self.cache = pd.DataFrame(columns=self.COLUMNS)
        self._save()

    def _append_cache(self, dict_to_append: Dict):
        assert 'hash' in dict_to_append
        if not self.cache[self.cache.hash == dict_to_append['hash']].empty:
            # delete old entry
            self.cache = self.cache[self.cache.hash != dict_to_append['hash']]

        self.cache = self.cache.append(dict_to_append, ignore_index=True)
        self.cache = self.cache.sort_values(by=['date'])
        self._save()

    @staticmethod
    def __save_as_datachunk(dc:DataChunk, db_path: str, hash):
        with h5py.File(db_path, 'a', libver='latest') as db:
            path = f'dc/{hash}'
            dc.to_hdf5(db, hash, path)
            db.flush()
            return path

    @staticmethod
    def __save_as_np_arr(df: pd.DataFrame, db_path: str, hash):
        with h5py.File(db_path, 'a', libver='latest') as db:
            path = f'df/{hash}'
            DataChunk.from_df(df).to_hdf5(db, hash, path)
            db.flush()
            return path

    @staticmethod
    def __load_as_np_arr(db_path: str, key) -> pd.DataFrame:
        with h5py.File(db_path, 'r', libver='latest') as db:
            if key in db:
                return DataChunk.from_hdf5(db, hash, key).as_df()
            return pd.DataFrame()

    @staticmethod
    def __load_as_datachunk(db_path: str, key) -> Union[DataChunk, None]:
        with h5py.File(db_path, 'r', libver='latest') as db:
            if key in db:
                return DataChunk.from_hdf5(db, hash, key)
            return None

    def _store_entry(self, args_hash: str, save_func: Callable, data: Any, key_override=None):
        if key_override is not None:
            key, db = key_override
            save_func(data, db, key)
            return

        if not self.cache[self.cache.hash == args_hash].empty:
            # if we hit the same hash, it means it is already stored
            if os.path.exists(self.cache[self.cache.hash == args_hash].data_path.values[0]):
                return

        key = save_func(data, self.cache_db_path, args_hash)
        self._append_cache({'hash': args_hash,
                            'date': str(datetime.datetime.now()),
                            'key': key,
                            'data_path': self.cache_db_path,
                            'type': CACHE_DATA_TYPES.Dataframe,
                            'commit': self.rev})

    def _read_entry(self, args_hash, load_func: Callable, key_override=None):
        if key_override is not None:
            key, db = key_override
            return load_func(db, key)

        if not self.cache[self.cache.hash == args_hash].empty:
            if args_hash in self.cached_data:
                return self.cached_data[args_hash]
            path = self.cache[self.cache.hash == args_hash].data_path.values[0]
            key = self.cache[self.cache.hash == args_hash].key.values[0]
            try:
                # result: pd.DataFrame = pd.read_hdf(self.cache_db_path, key=key)
                result = load_func(self.cache_db_path, key)
                # result = self.__load_as_np_arr(self.cache_db_path, key)
                # if result.empty:
                #     self.cache = self.cache[self.cache.hash != args_hash]
                #     self._save()
                #     return None

                self.cached_data[args_hash] = result
            except Exception as ex:
                LOGGER.exception(f'Exception when trying to read cache with path {path}!:\n {ex}')

                return None
            return result
        return None

    def read_df(self, args_hash):
        return self._read_entry(args_hash, self.__load_as_np_arr)

    def store_df(self, args_hash, df: pd.DataFrame):
        return self._store_entry(args_hash, self.__save_as_np_arr, df)

    def read_datachunk(self, args_hash)->Union[None, DataChunk]:
        return self._read_entry(args_hash, self.__load_as_datachunk)

    def store_datachunk(self, args_hash, dc: DataChunk):
        return self._store_entry(args_hash, self.__save_as_datachunk, dc)

    def read_custom(self, db, key, load_method:Callable[[h5py.File, str], Any]) -> Union[None, Any]:
        def save_override(db_path, key):
            db_path = os.path.join(self.cache_path_dir, db_path + "/db.h5") if os.path.isdir(db_path) else db_path
            with h5py.File(db_path, 'r', libver='latest') as db:
                load_method(db, f"custom/{key}")

        return self._read_entry('', save_override, key_override=(key, db))

    def store_custom(self, db, key, data: Any, store_method:Callable[[h5py.File, str, Any], Any]):
        def store_override(data, db_path, key):
            db_path = os.path.join(self.cache_path_dir, db_path+"/db.h5") if os.path.isdir(db_path) else db_path
            with h5py.File(db_path, 'a', libver='latest') as db:
                store_method(db, f"custom/{key}", data)
                db.flush()

        return self._store_entry('', store_override, data, key_override=(key, db))

    @staticmethod
    def build_hash(*args, **kwargs) -> str:
        return Cacher.generate_unique_key(*args, salt=Cacher.VERSION, **kwargs)

__cacher = Cacher()

def init_locks(new_lock):
    global __g_cacher_lock
    __g_cacher_lock = new_lock

    with __g_cacher_lock:
        __cacher.init()

# csv_path_key can be used to store the path to file
def cache_result_df():
    def decorator(func=None):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            args_hash = Cacher.build_hash(*args, func_cache=hashlib.md5(func.__code__.co_code).hexdigest(),**kwargs)

            with instance() as cacher:
                result = cacher.read_df(args_hash)
                if result is not None and not result.empty:
                    if cacher.with_stats:
                        LOGGER.info(f"cache hit for func='{func.__name__}'")
                    return result, args_hash

            result = func(*args, **kwargs)

            with instance() as cacher:
                assert isinstance(result, pd.DataFrame), "you don't return a dataframe. consider using other wrapper"

                cacher.store_df(args_hash, result)
                if cacher.with_stats:
                    LOGGER.info(f"cache miss for func='{func.__name__}'")

            return result, args_hash

        return inner

    return decorator


@contextmanager
def instance():
    __g_cacher_lock.acquire()
    try:
        yield __cacher
    finally:
        __g_cacher_lock.release()
