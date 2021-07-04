import multiprocessing
from abc import abstractmethod, ABCMeta
from contextlib import contextmanager
from typing import List, Union, Callable, Any, Dict

import numpy as np
import pandas as pd

class HDF5Serializable:
    @abstractmethod
    def to_hdf5(self, db, hash, path):
        pass

    @staticmethod
    def from_hdf5(db, hash, path):
        pass


class DataChunk(HDF5Serializable):
    def __init__(self, np_ndarr: np.ndarray, source=None):
        self.np_arr = np_ndarr
        self.__source = source

    def jit_hash(self):
        assert self.__source is not None
        return self.__source

    def cachable(self):
        return self.__source is not None


class DFDataChunk(DataChunk):
    def __init__(self,
                 np_index: np.ndarray,
                 np_chunk_data: np.ndarray,
                 columns: List[str],
                 dtypes: List,
                 source=None):
        super(DFDataChunk, self).__init__(np_ndarr=np_chunk_data,
                                          source=source)
        self.index = np_index
        self.columns = columns
        self.dtypes = dtypes

    @staticmethod
    def from_df(df: pd.DataFrame, hash_source=None):
        return DFDataChunk(np_chunk_data=df.values, np_index=df.index.values, columns=list(df.columns),
                           dtypes=[dt.str for dt in df.dtypes], source=hash_source)

    def as_df(self):
        return pd.DataFrame({
            column: self.np_arr[:, idx].astype(self.dtypes[idx])
            for idx, column in enumerate(self.columns)},
            index=self.index)

    def to_hdf5(self, db, hash, path):
        columns = list(self.columns)
        ndarr = self.np_arr
        idx = self.index
        db.create_dataset(f"{path}/val", shape=ndarr.shape, data=ndarr, compression="gzip")
        db[f"{path}/col"] = columns
        db[f"{path}/idx"] = idx
        db[f"{path}/dtype"] = self.dtypes

    @staticmethod
    def from_hdf5(db, hash, path):
        np_chunk_data = db[f"{path}/val"][()]
        columns = [column.decode('utf-8') for column in db[f"{path}/col"]]
        dtypes = [dtype.decode('utf-8') for dtype in db[f"{path}/dtype"]]
        index = db[f"{path}/idx"][()]
        return DFDataChunk(index, np_chunk_data, columns, dtypes, source=hash)
