import functools
import hashlib
import logging
import os
import datetime
import gin
import subprocess
import pandas as pd
import numpy as np
import collections

LOGGER = logging.getLogger('ariadne.cache')


class CACHE_DATA_TYPES:
    Object = 'Object'
    Dataframe = 'Dataframe'
    Numpy = 'Numpy'


# God bless python..
def is_hashable(obj):
    return isinstance(obj, collections.Hashable)


@gin.configurable
class Cacher():
    # bump this version if you change the code, otherwise cache might be wrong
    VERSION = 0
    CACHE_INFO_FILE = 'cache_info.bin'
    COLUMNS = ['hash', 'date', 'entry_path', 'data_path', 'type', 'commit']

    def __init__(self,
                 cache_path='.jit',
                 df_compression_level=5):
        self.cache_path_dir = cache_path
        self.cache_info_path = os.path.join(self.cache_path_dir, self.CACHE_INFO_FILE)
        os.makedirs(self.cache_path_dir, exist_ok=True)
        self.df_compression_level = df_compression_level
        self.cache: pd.DataFrame = pd.DataFrame(columns=self.COLUMNS)
        self.cached_data = {}
        try:
            self.rev = self.get_git_revision_hash()
        except Exception as ex:
            LOGGER.error('Exception when trying to get git hash... bad!')
            self.rev = 'no revision'
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
        self._save()

    def store_df(self, args_hash: str, data: pd.DataFrame, path: str = ''):
        assert not path or os.path.isabs(path), 'jit cache expects empty or absolute path to the csv dataframe'
        if not self.cache[self.cache.hash == args_hash].empty:
            # if we hit the same hash, it means it is already stored
            if os.path.exists(self.cache[self.cache.hash == args_hash].data_path.values[0]):
                return

        file_name = f'df_{args_hash}.bin'
        full_path = os.path.abspath(os.path.join(self.cache_path_dir, file_name))

        data.to_hdf(f"{full_path}",
                    key=file_name,
                    mode='w',
                    complevel=self.df_compression_level,
                    append=False,
                    format='fixed')

        if not self.cache[self.cache.hash == args_hash].empty:
            # delete old entry
            self.cache = self.cache[self.cache.hash != args_hash]

        self.cache = self.cache.append({'hash': args_hash,
                                            'date': str(datetime.datetime.now()),
                                            'entry_path': path,
                                            'data_path': full_path,
                                            'type': CACHE_DATA_TYPES.Dataframe,
                                            'commit': self.rev}, ignore_index=True)
        self.cache = self.cache.sort_values(by=['date'])

        self._save()

    def read_df(self, args_hash):
        if not self.cache[self.cache.hash == args_hash].empty:
            if args_hash in self.cached_data:
                return self.cached_data[args_hash]
            path = self.cache[self.cache.hash == args_hash].data_path.values[0]
            try:
                result: pd.DataFrame = pd.read_hdf(path)
                self.cached_data[args_hash] = result
            except Exception as ex:
                LOGGER.exception(f'Exception when trying to read cache with path {path}!:\n {ex}')
                return None
            return result
        return None

    def build_hash(self, *args, **kwargs) -> str:
        return self.generate_unique_key(*args, salt=self.VERSION, **kwargs)


_cacher = Cacher()


# csv_path_key can be used to store the path to file
def cache_result_df(csv_path_key=''):
    def decorator(func=None):
        cacher = _cacher

        @functools.wraps(func)
        def inner(*args, **kwargs):
            args_hash = cacher.build_hash(*args, func_cache=hashlib.md5(func.__code__.co_code).hexdigest(), **kwargs)
            result = cacher.read_df(args_hash)
            if result is not None and not result.empty:
                return result
            result = func(*args, **kwargs)
            assert isinstance(result, pd.DataFrame), "you don't return a dataframe. consider using other wrapper"

            csv_path = os.path.abspath(kwargs[csv_path_key]) if csv_path_key else ''
            cacher.store_df(args_hash, result, csv_path)
            return result

        return inner

    return decorator
