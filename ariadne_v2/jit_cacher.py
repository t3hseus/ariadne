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

from ariadne_v2.data_chunk import DFDataChunk

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
                 cache_path='_jit',
                 df_compression_level=5,
                 with_stats=True):
        self.cache_path_dir = cache_path
        self.cache_info_path = os.path.join(self.cache_path_dir, self.CACHE_INFO_FILE)
        os.makedirs(self.cache_path_dir, exist_ok=True)
        self.df_compression_level = df_compression_level
        self.cache_db_path = os.path.join(self.cache_path_dir, self.CACHE_DB_FILE)
        self.with_stats = with_stats
        self.cache = None

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
    def __save_as_datachunk(dc: DFDataChunk, db_path: str, hash):
        with h5py.File(db_path, 'a', libver='latest') as db:
            path = f'dc/{hash}'
            dc.to_hdf5(db, hash, path)
            db.flush()
            return path

    @staticmethod
    def __save_as_np_arr(df: pd.DataFrame, db_path: str, hash):
        with h5py.File(db_path, 'a', libver='latest') as db:
            path = f'df/{hash}'
            DFDataChunk.from_df(df).to_hdf5(db, hash, path)
            db.flush()
            return path

    @staticmethod
    def __load_as_np_arr(db_path: str, key) -> pd.DataFrame:
        with h5py.File(db_path, 'r', libver='latest') as db:
            if key in db:
                return DFDataChunk.from_hdf5(db, hash, key).as_df()
            return pd.DataFrame()

    @staticmethod
    def __load_as_datachunk(db_path: str, key) -> Union[DFDataChunk, None]:
        with h5py.File(db_path, 'r', libver='latest') as db:
            if key in db:
                return DFDataChunk.from_hdf5(db, hash, key)
            return None

    def _store_entry(self, args_hash: str, save_func: Callable, data: Any, db: Union[str, None]):
        db_path = self.cache_db_path if db is None else self.to_db_path(db)
        key = save_func(data, db_path, args_hash)
        self._append_cache({'hash': args_hash,
                            'date': str(datetime.datetime.now()),
                            'key': key,
                            'data_path': db_path,
                            'type': CACHE_DATA_TYPES.Dataframe,
                            'commit': self.rev})

    def to_db_path(self, db: str):
        new_path = os.path.join(self.cache_path_dir, db)
        return new_path + "/db.h5" if os.path.isdir(new_path) else db

    def _read_entry(self, args_hash, load_func: Callable, db: Union[str, None]):
        if not self.cache[self.cache.hash == args_hash].empty:
            path = self.cache[self.cache.hash == args_hash].data_path.values[0]
            key = self.cache[self.cache.hash == args_hash].key.values[0]
            try:
                db_path = self.cache_db_path if db is None else self.to_db_path(db)
                result = load_func(db_path, key)
            except Exception as ex:
                LOGGER.exception(f'Exception when trying to read cache with path {path}!:\n {ex}')

                return None
            return result
        return None

    def read_df(self, args_hash, db=None):
        df = self._read_entry(args_hash, self.__load_as_np_arr, db=db)
        if isinstance(df, DFDataChunk):
            return df.as_df()
        return df

    def store_df(self, args_hash, df: pd.DataFrame, db=None):
        return self._store_entry(args_hash, self.__save_as_np_arr, df, db=db)

    def read_datachunk(self, args_hash, db=None) -> Union[None, DFDataChunk]:
        return self._read_entry(args_hash, self.__load_as_datachunk, db=db)

    def store_datachunk(self, args_hash, dc: DFDataChunk, db=None):
        return self._store_entry(args_hash, self.__save_as_datachunk, dc, db=db)

    def update_custom(self, db: str, key,
                      update_method: Union[Callable[[h5py.File, str], Any], None],
                      update_attr: Union[Callable[[Any], Any], None]) -> Union[None, Any]:
        assert update_method or update_attr, "no methods provided for update"
        db_path = self.to_db_path(db)
        with h5py.File(db_path, 'a', libver='latest') as db:
            if update_method is None:
                val = db.attrs[key] if key in db.attrs else None
                val = update_attr(val)
                db.attrs[key] = val
                return val
            else:
                return update_method(db, f"custom/{key}")

    def read_custom(self, db: str, key,
                    load_method: Union[Callable[[h5py.File, str], Any], None]) -> Union[None, Any]:
        db_path = self.to_db_path(db)
        with h5py.File(db_path, 'r', libver='latest') as db:
            if load_method is None:
                if key in db.attrs:
                    return db.attrs[key]
                return None
            else:
                load_method(db, f"custom/{key}")

    def store_custom(self, db: str, key: str, data: Any,
                     store_method: Union[Callable[[h5py.File, str, Any], Any], None]):
        db_path = self.to_db_path(db)
        with h5py.File(db_path, 'a', libver='latest') as db:
            if store_method is None:
                db.attrs[key] = data
            else:
                store_method(db, f"custom/{key}", data)
            db.flush()

    def update_attr(self, db, key, update_attr):
        self.update_custom(db, key, update_method=None, update_attr=update_attr)

    def store_attr(self, db: str, key: str, data: Any):
        self.store_custom(db, key, data, store_method=None)

    def read_attr(self, db: str, key: str):
        return self.read_custom(db, key, load_method=None)

    @contextmanager
    def handle(self, db_path: str, mode: str = 'a') -> h5py.File:
        db_path = self.to_db_path(db_path)
        db = h5py.File(db_path, mode=mode, libver='latest')
        try:
            yield db
        finally:
            db.close()

    def raw_handle(self, db_path: str, mode: str = 'a') -> h5py.File:
        db_path = self.to_db_path(db_path)
        return h5py.File(db_path, mode=mode, libver='latest')

    @staticmethod
    def build_hash(*args, **kwargs) -> str:
        return Cacher.generate_unique_key(*args, salt=Cacher.VERSION, **kwargs)


__cacher = Cacher()


def init_locks(new_lock):
    global __g_cacher_lock
    __g_cacher_lock = new_lock

    with __g_cacher_lock:
        __cacher.init()

def fini_locks():
    del globals()['__cacher']
    del globals()['__g_cacher_lock']

# csv_path_key can be used to store the path to file
def cache_result_df():
    def decorator(func=None):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            args_hash = Cacher.build_hash(*args, func_cache=hashlib.md5(func.__code__.co_code).hexdigest(), **kwargs)

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
def instance(existing=None) -> Cacher:
    if existing is not None:
        try:
            yield existing
        finally:
            pass
    else:
        __g_cacher_lock.acquire()
        try:
            yield __cacher
        finally:
            __g_cacher_lock.release()
