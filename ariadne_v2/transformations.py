import logging
from typing import Optional, List
from copy import deepcopy

import gin
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer
)
import torch

from ariadne_v2 import jit_cacher
from ariadne_v2.data_chunk import DFDataChunk
from ariadne_v2.jit_cacher import Cacher

from tqdm import tqdm

LOGGER = logging.getLogger('ariadne.transforms')


class Compose:
    """Composes several transforms together. Mostly copied from torchvision.
    Args:
        transforms (list of ``Transform`` objects): list of
            transforms to compose.

    Example:
        >>> Compose([
        >>>     transforms.StandardScale(),
        >>>     transforms.ToCylindrical(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data: DFDataChunk, preserve_index=True, return_hash=False):
        hash = None
        if data.cachable():
            rep = {f'tr_{idx}': ("%r" % t) for idx, t in enumerate(self.transforms)}
            hash = Cacher.build_hash(preserve_index, **rep, SRC=data.jit_hash())
            with jit_cacher.instance() as cacher:
                dc = cacher.read_datachunk(hash)
                if dc:
                    if not return_hash:
                        return dc.as_df()
                    else:
                        return dc.as_df(), hash

        data = data.as_df()
        if preserve_index:
            data['index'] = data.index
        for t in self.transforms:
            data = t(data)
            if data.empty:
                LOGGER.warning(f'{t.__class__.__name__} returned empty data. '
                               'Skipping all further transforms')
                if not return_hash:
                    return data
                else:
                    return data, None

        if hash is not None:
            dc = DFDataChunk.from_df(data, hash)
            with jit_cacher.instance() as cacher:
                cacher.store_datachunk(hash, dc)

        if not return_hash:
            return data
        else:
            return data, hash

    def __repr__(self):
        """
        Returns:
            str: formatted strings with class_names,
                parameters and some statistics for each class
        """
        transforms_str = ',\n'.join(
            [f'  {t.__class__.__name__}' for t in self.transforms])
        fmt_str = f'{self.__name__}(\n{transforms_str}\n)'
        return fmt_str


class BaseTransformer(object):
    """Base class for transforms
    # Args:
         columns (list or tuple, ['x', 'y', 'z'] by default): Columns to transform
         drop_old (boolean, True by default): If True, original data is discarded,
                                            else preserved in columns with suffix '_old'
         keep_fakes (boolean, True by default): If True, hits with no track are preserved
    """

    def __init__(self, drop_old=False, columns=('x', 'y', 'z'), track_col='track', event_col='event',
                 station_col='station'):
        self.drop_old = drop_old
        self.columns = columns
        self.fakes = None
        self.event_column = event_col
        self.track_column = track_col
        self.station_column = station_col
        assert len(columns) == 3, "Columns must be list or tuple of length 3"

    def transform_data(self, data, normed):
        for i in range(3):
            if not self.drop_old:
                data.loc[:, self.columns[i] + '_old'] = 1
                data.loc[:, self.columns[i] + '_old'] = deepcopy(data.loc[:, self.columns[i]])

            data.loc[:, self.columns[i]] = normed[i]
        return data

    def drop_fakes(self, data):
        return data[data[self.track_column] != -1]

    def get_num_fakes(self):
        if self.fakes:
            return len(self.fakes)
        return 0

    def add_fakes(self, data, fakes):
        return pd.concat([data, fakes], axis=0, ignore_index=True)


class BaseScaler(BaseTransformer):
    """Base class for scalers.
     # Args:
         scaler (function or method pd.DataFrame -> iterable of pd.Series): scaler with fit_predict method
         columns (list or tuple, ['x', 'y', 'z'] by default): Columns to scale
         drop_old (boolean, True by default): If True, unscaled data is discarded,
                                            else preserved in columns with suffix '_old'
    """

    def __init__(self, scaler, drop_old=True, columns=('x', 'y', 'z')):
        super().__init__(drop_old=drop_old, columns=columns)
        self.scaler = scaler

    def __call__(self, data):
        """
        # Args:
            data (pd.DataFrame):  to clean up.
        # Returns:
            data (pd.DataFrame): transformed dataframe
        """
        norms = pd.DataFrame(self.scaler.fit_transform(data[[self.columns[0], self.columns[1], self.columns[2]]]))
        data = self.transform_data(data=data, normed=norms)
        return data

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with scaler: {self.scaler}'
                f'{"-" * 30}\n')


class BaseFilter(BaseTransformer):
    """Base class for all filtering transforms
     # Args:
         filter_rule (function or method pd.DataFrame -> iterable of pd.Series): Function, which
                                 convertes data, returned value must be iterable with pd.Series values (list etc)
         keep_fakes (boolean, True by default): If True, hits with no track are preserved
         station_col (string, 'station' by default): column with station identifiers
         event_col (string, 'event' by default): column with event identifiers
         track_col (string, 'track' by default): column with station identifiers
    """

    def __init__(self, filter_rule, num_stations=None, keep_filtered=True, station_col='station', track_col='track',
                 event_col='event'):
        super().__init__(station_col=station_col, track_col=track_col, event_col=event_col)
        self.num_stations = num_stations
        self._broken_tracks = None
        self._num_broken_tracks = None
        self.filter_rule = filter_rule
        self.keep_filtered = keep_filtered

    def __call__(self, data):
        """
        # Args:
            data (pd.DataFrame):  to clean up.
        # Returns:
            data (pd.DataFrame): transformed dataframe
        """
        data = data.copy()
        fakes = data.loc[data[self.track_column] == -1, :]
        data = self.drop_fakes(data)
        tracks = data.groupby([self.event_column, self.track_column])
        if self.num_stations is None:
            self.num_stations = tracks.size().max()
        good_tracks = tracks.filter(self.filter_rule)
        broken = list(data.loc[~data.index.isin(good_tracks.index)].index)
        self._broken_tracks = data.loc[broken, [self.event_column, self.track_column, self.station_column]]
        self._num_broken_tracks = len(self._broken_tracks[[self.event_column, self.track_column]].drop_duplicates())
        if self.keep_filtered and len(broken) > 0:
            data.loc[~data.index.isin(good_tracks.index), 'track'] = -1
        else:
            data = data.loc[data.index.isin(good_tracks.index), :]
        return self.add_fakes(data, fakes)

    def get_broken(self):
        return self._broken_tracks

    def get_num_broken(self):
        return self._num_broken_tracks

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with filter_rule: {self.filter_rule}\n'
                f'{"-" * 30}\n')


class BaseCoordConverter(BaseTransformer):
    """Base class for coordinate convertions

    # Args:
         convert_function(function or method pd.DataFrame -> iterable of pd.Series): Function, which
         convertes data, returned value must be iterable with pd.Series values (list etc)
         drop_old (boolean, False by default): If True, old columns are discarded from data
         from_columns (list or tuple of length 3, ['x', 'y', 'z'] by default): list of original features
         to_columns (list or tuple of length 3, ['r', 'phi', 'z'] by default): list of features to convert to
    """

    def __init__(self, convert_function, drop_old=False, from_columns=('x', 'y', 'z'),
                 to_columns=('r', 'phi', 'z'), postfix='general_convert'):
        assert len(from_columns) == 3, 'To convert coordinates, you need 3 old columns'
        assert len(to_columns) == 3, 'To convert coordinates, you need 3 new columns'
        super().__init__(drop_old=drop_old, columns=to_columns)
        self.convert_function = convert_function
        self.range_ = {}
        self.from_columns = from_columns
        self.to_columns = to_columns
        self.postfix = postfix

    def __call__(self, data):
        """
        # Args:
            data (pd.DataFrame):  to clean up.
        # Returns:
            data (pd.DataFrame): transformed dataframe
        """
        # assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        self.get_ranges(data, self.from_columns)
        converted = self.convert_function(data)
        if not self.drop_old:
            for col in self.from_columns:
                data[col + '_' + self.postfix] = data[col]
        if self.drop_old:
            for col in self.from_columns:
                del data[col]
        for i in range(len(self.to_columns)):
            data.loc[:, self.to_columns[i]] = converted[i]
        self.get_ranges(data, self.to_columns)
        return data

    def get_ranges(self, data, columns):
        for col in columns:
            try:
                self.range_[col] = (min(data[col]), max(data[col]))
            except:
                'This column is not used'

    def get_ranges_str(self):
        return '\n'.join([f'{i}: from {j[0]} to {j[1]}' for i, j in self.range_.items()])

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with convert_function: '
                f'{self.convert_function}\n'
                f'{"-" * 30}\n')


@gin.configurable
class PreserveOriginal:
    """Preserves original state of given columns.
    May be needed to use original state of column
    in future transforms or tests

    # Args:
         columns (list or tuple, None by default): Columns to keep state of.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def __call__(self, data):
        """
        # Args:
            data (pd.DataFrame):  to clean up.
        # Returns:
            data (pd.DataFrame): transformed dataframe
        """
        if not self.columns:
            self.columns = data.columns
        for col in self.columns:
            data.loc[:, col + '_original'] = data.loc[:, col]
        return data

    def __repr__(self):
        return (f"{'-' * 30}\n"
                f'{self.__class__.__name__} ceeping original state of columns: {self.columns} \n'
                f"{'-' * 30}\n")


@gin.configurable
class BakeStationValues:
    """Coverts z coordinate of hit to given value for each station station.
    Nessesary for BM@N, becaule strips have different depths.
     # Args:
         scaler (function or method pd.DataFrame -> iterable of pd.Series): scaler with fit_predict method
         columns (list or tuple, ['x', 'y', 'z'] by default): Columns to scale
         drop_old (boolean, True by default): If True, unscaled data is discarded,
                                            else preserved in columns with suffix '_old'
    """

    def __init__(self, values, col='z', station_col='station'):
        self.station_values = values
        self.station_column = station_col
        self.column = col

    def __call__(self, data):
        """
        # Args:
            data (pd.DataFrame):  to clean up.
        # Returns:
            data (pd.DataFrame): transformed dataframe
        """
        for i, value in self.station_values.items():
            data.loc[data['station'] == i, self.column] = value
        return data

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__}' #with scaler: {self.scaler}'
                f'{"-" * 30}\n')


@gin.configurable
class StandardScale(BaseScaler):
    """Standardizes coordinates by removing the mean and scaling to unit variance
    # Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        with_mean (boolean, True by default): If True, center the data before scaling
        with_std (boolean, True by default): If True, scale the data to unit variance (or equivalently, unit standard deviation).
        columns (list or tuple, ('x', 'y', 'z') by default): Columns to Standardize
    """

    def __init__(self, drop_old=True, with_mean=True, with_std=True, columns=('x', 'y', 'z')):
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = StandardScaler(with_mean, with_std)
        super().__init__(self.scaler, drop_old, columns)

    def __repr__(self):
        return (f"{'-' * 30} \n"
                f"{self.__class__.__name__} with scaling_parameters: drop_old={self.drop_old}, "
                f"with_mean={self.with_mean},with_std={self.with_std} \n"
                f"{'-' * 30} \n"
                f" Mean: {self.scaler.mean_} \n Var: {self.scaler.var_} \n Scale: {self.scaler.scale_} ")


@gin.configurable
class MinMaxScale(BaseScaler):
    """Transforms features by scaling each feature to a given range.
     # Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        feature_range (Tuple (min,max), default (0,1)): Desired range of transformed data.
        columns (list or tuple, ('x', 'y', 'z') by default): Columns to Standardize
    """

    def __init__(self, drop_old=True, feature_range=(0, 1), columns=('x', 'y', 'z')):
        assert feature_range[0] < feature_range[1], 'minimum is not smaller value then maximum'
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=feature_range)
        super().__init__(self.scaler, drop_old, columns=columns)

    def __repr__(self):
        return (f"{'-' * 30}\n"
                f"{self.__class__.__name__} with parameters: "
                f"drop_old={self.drop_old}, feature_range={self.feature_range} \n"
                f"{'-' * 30}\n"
                f" Data min: {self.scaler.data_min_} \n Data max: {self.scaler.data_max_} \n Scale: {self.scaler.scale_}")


@gin.configurable
class Normalize(BaseScaler):
    """Normalizes samples individually to unit norm.
    Each sample (i.e. each row of the data matrix) with at least one non zero component is rescaled independently of
    other samples so that its norm (l1, l2 or inf) equals one.

      # Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        norm (‘l1’, ‘l2’, or ‘max’ (‘l2’ by default)): The norm to use to normalize each non zero sample.
                              If norm=’max’ is used, values will be rescaled by the maximum of the absolute values.
        columns (list or tuple, ('x', 'y', 'z') by default): Columns to Standardize
    """

    def __init__(self, drop_old=True, norm='l2', columns=('x', 'y', 'z')):
        self.norm = norm
        self.scaler = Normalizer(norm=norm)
        super().__init__(self.scaler, drop_old, columns)

    def __repr__(self):
        return (f"{'-' * 30}\n"
                f"{self.__class__.__name__} with parameters: drop_old={self.drop_old}, norm={self.norm} \n"
                f"{'-' * 30}\n")


@gin.configurable
class ConstraintsNormalize(BaseTransformer):
    """Normalizes samples using station given characteristics  or computes them by call.
    If you need to compute characteristics, you can use MinMaxScale too (maybe better)
    Each station can have its own constraints or global constrains.
      Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        columns (list or tuple, ('x', 'y', 'z') by default): Columns to scale
        margin (number, positive): margin applied to stations (min = min-margin, max=max+margin)
        constraints (dict, None by deault) If None, constraints are computed using dataset statistics.
        use_global_constraints (boolean, True by default) If True, all data is scaled using given global constraints.

    If use_global_constraints is True and constraints is not None, constraints must be {column:(min,max)},
    else it must be {station: {column:(min,max)}}.

    Station keys for constraints must be in dataset. Number of constraints for each column must be 2.
    Number of constraints must be the same as number of columns
    """

    def __init__(self, drop_old=True, columns=('x', 'y', 'z'), margin=1e-3, use_global_constraints=True,
                 constraints=None):
        super().__init__(drop_old=drop_old, columns=columns)
        assert margin > 0, 'Margin is not positive'
        self.margin = margin
        self.use_global_constraints = use_global_constraints
        self.constraints = constraints
        if constraints is not None:
            if use_global_constraints:
                for col in columns:
                    assert col in constraints.keys(), f'{col} is not in constraint keys {constraints.keys()}'
                    assert len(constraints[col]) == 2, f'Not applicable number of constraints for column {col}'
                    assert constraints[col][0] < constraints[col][
                        1], f'Minimum is not smaller than maximum for column {col}'
            else:
                for key, constraint in constraints.items():
                    for col in columns:
                        assert col in constraint.keys(), f'{col} is not in constraint keys for station {key}'
                        assert len(constraint[
                                       col]) == 2, f'Not applicable number of constraints for column {col} and station {key}'
                        assert constraint[col][0] < constraint[col][
                            1], f'Minimum is not smaller than maximum for column {col} and station {key}'

    def __call__(self, data):
        """
        # Args:
            data (pd.DataFrame):  to clean up.
        # Returns:
            data (pd.DataFrame): transformed dataframe
        """
        if self.constraints is None:
            self.constraints = self.get_stations_constraints(data)
        if self.use_global_constraints:
            global_constrains = {}
            for col in self.columns:
                global_min = self.constraints[col][0]
                global_max = self.constraints[col][1]
                assert global_min < global_max, f"global_min should be < global_max {global_min} < {global_max}"
                global_constrains[col] = (global_min, global_max)
            x_norm, y_norm, z_norm = self.normalize(data, global_constrains)
            data = super().transform_data(data, [x_norm, y_norm, z_norm])
        else:
            ##assert all([station in data['station'].unique() for station in
            #           self.constraints.keys()]), "Some station keys in constraints are not presented in data. Keys: " \
            #                                       f"{data['station'].unique()}; data keys: {self.constraints.keys()}"

            for station in list(data['station'].unique()):
                # print(data['station'].unique())
                group = data.loc[data['station'] == station,]
                x_norm, y_norm, z_norm = self.normalize(group, self.constraints[station])
                data.loc[data['station'] == station, :] = \
                    self.transform_data_by_group(data['station'] == station, data,
                                                 [x_norm, y_norm, z_norm])
        #data['event'] = data['event'] % 200
        return data

    def transform_data_by_group(self, grouping, data, normed):
        for i in range(3):
            assert self.drop_old, "Saving old data is not supported for now"
            data.loc[grouping, self.columns[i]] = normed[i]
        return data

    def get_stations_constraints(self, df):
        groups = df['station'].unique()

        station_constraints = {}
        for station_num in groups:
            group = df.loc[df['station'] == station_num,]
            min_x, max_x = min(group[self.columns[0]]) - self.margin, max(group[self.columns[0]]) + self.margin
            min_y, max_y = min(group[self.columns[1]]) - self.margin, max(group[self.columns[1]]) + self.margin
            min_z, max_z = min(group[self.columns[2]]) - self.margin, max(group[self.columns[2]]) + self.margin
            station_constraints[station_num] = {self.columns[0]: (min_x, max_x),
                                                self.columns[1]: (min_y, max_y),
                                                self.columns[2]: (min_z, max_z)}
        return station_constraints

    def normalize(self, df, constraints):
        x_min, x_max = constraints[self.columns[0]]
        y_min, y_max = constraints[self.columns[1]]
        z_min, z_max = constraints[self.columns[2]]
        #assert all(df[self.columns[0]].between(x_min, x_max)), \
        #    f'Some values in column {self.columns[0]} are not in {constraints[self.columns[0]]}'
        x_norm = 2 * (df[self.columns[0]] - x_min) / (x_max - x_min) - 1
        #assert all(df[self.columns[1]].between(y_min, y_max)), \
        #    f'Some values in column {self.columns[1]} are not in {constraints[self.columns[1]]}'
        y_norm = 2 * (df[self.columns[1]] - y_min) / (y_max - y_min) - 1
        #assert all(df[self.columns[2]].between(z_min, z_max)), \
        #    f'Some values in column {self.columns[2]} are not in {constraints[self.columns[2]]}'
        z_norm = 2 * (df[self.columns[2]] - z_min) / (z_max - z_min) - 1
        return x_norm, y_norm, z_norm

    def __repr__(self):
        return (f"{'-' * 30}\n"
                f"{self.__class__.__name__} with parameters: "
                f"drop_old={self.drop_old}, "
                f"use_global_constraints={self.use_global_constraints}\n"
                f"{' ' * 20}margin={self.margin}, columns={self.columns}\n"
                f"{'-' * 30}\nconstraints are: {self.constraints}")


@gin.configurable
class DropShort(BaseFilter):
    """Drops tracks with num of points less then given from data.
      # Args:
        num_stations (int, default None): Desired number of stations (points). If None, maximum stations number for one track is taken from data.
        keep_fakes (bool, default True): If True, points with no tracks are preserved, else they are deleted from data.
        station_column (str, 'station' by default): Event column in data
        track_column (str, 'track' by default): Track column in data
        event_column (str, 'event' by default): Station column in data
    """

    def __init__(self, num_stations=None, keep_filtered=True, station_col='station', track_col='track',
                 event_col='event'):
        self.num_stations = num_stations
        self.broken_tracks_ = None
        self.num_broken_tracks_ = None
        self.filter = lambda x: len(x) >= self.num_stations
        super().__init__(self.filter, num_stations=num_stations, station_col=station_col, track_col=track_col,
                         event_col=event_col, keep_filtered=keep_filtered)

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with parameters: num_stations={self.num_stations}, '
                f'    track_column={self.track_column}, station_column={self.station_column}, '
                f'event_column={self.event_column}\n'
                f'{"-" * 30}\n'
                f'Number of broken tracks: {self.get_num_broken()} \n')


@gin.configurable
class DropEmpty(object):
    """Drops events without any tracks
    """

    def __init__(self, track_col='track', event_col='event'):
        self.track_column = track_col
        self.event_column = event_col

    def __call__(self, data):
        group = data.groupby(self.event_column)
        result = group.filter(lambda e: not e[e[self.track_column] != -1].empty)
        return result

    def __repr__(self):
        return self.__class__.__name__

@gin.configurable
class DropSpinningTracks(BaseFilter):
    """Drops tracks with points on same stations (e.g. (2,2,2) or (1,2,1)).
      # Args:
        keep_fakes (bool, True by default ): If True, points with no tracks are preserved, else they are deleted from data.
        station_col (str, 'station' by default): Event column in data
        track_col(str, 'track' by default): Track column in data
        event_col (str, 'event' by default): Station column in data
    """

    def __init__(self, keep_filtered=True, station_col='station', track_col='track', event_col='event'):
        self.filter = lambda x: x[self.station_column].unique().shape[0] == x[self.station_column].shape[0]
        super().__init__(self.filter, station_col=station_col, track_col=track_col, event_col=event_col,
                         keep_filtered=keep_filtered)

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with parameters:'
                f'    track_column={self.track_column}, station_column={self.station_column}, event_column={self.event_column}\n'
                f'{"-" * 30}\n'
                f'Number of broken tracks: {self.get_num_broken()} \n')


@gin.configurable
class UnspinSpinningTracks(BaseFilter):
    """Drops tracks with points on same stations (e.g. (2,2,2) or (1,2,1)). If z values on station are
      # Args:
        keep_fakes (bool, True by default ): If True, points with no tracks are preserved, else they are deleted from data.
        station_col (str, 'station' by default): Event column in data
        track_col(str, 'track' by default): Track column in data
        event_col (str, 'event' by default): Station column in data
    """

    def check_spinning(self, x):
        if x[self.station_column].unique().shape[0] != x[self.station_column].shape[0]:
            for station in x[self.station_column].unique():
                z_values = x[x[self.station_column] == station][self.z_column].values
                if len(z_values) > 1:
                    if len(z_values.unique()) > 1:
                        for col in self.columns:
                            x.loc[x[self.station_column] == station, col] = np.mean(
                                x.loc[x[self.station_column] == station, col])
                        print(x.loc[x[self.station_column] == station].index)
                        x.drop(x.loc[x[self.station_column] == station].index[:-1], inplace=True)
                    else:
                        return False
        return True

    def __init__(self, keep_filtered=True, columns=('x', 'y', 'z'), z_column='z', station_col='station',
                 track_col='track', event_col='event'):

        self.z_column = z_column
        self.columns = columns
        self.filter = self.check_spinning
        super().__init__(self.filter, station_col=station_col, track_col=track_col, event_col=event_col,
                         keep_filtered=keep_filtered)

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with parameters:'
                f'    track_column={self.track_column}, station_column={self.station_column}, event_column={self.event_column}\n'
                f'{"-" * 30}\n'
                f'Number of broken tracks: {self.get_num_broken()} \n')


@gin.configurable
class DropTracksWithHoles(BaseFilter):
    """Drops tracks with points on same stations (e.g. (2,2,2) or (1,2,1)).
      # Args:
        keep_fakes (bool, True by default ): If True, points with no tracks are preserved, else they are deleted from data.
        station_col (str, 'station' by default): Event column in data
        track_col(str, 'track' by default): Track column in data
        event_col (str, 'event' by default): Station column in data
    """

    def __init__(self,
                 keep_filtered=True,
                 station_col='station',
                 track_col='track',
                 event_col='event',
                 min_station_num=0):
        self.filter = lambda x: x[self.station_column].values.shape[0] == \
                                len(np.arange(min_station_num, int(x[self.station_column].max()))) + 1
        super().__init__(self.filter, station_col=station_col, track_col=track_col, event_col=event_col,
                         keep_filtered=keep_filtered)

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with parameters:'
                f'    track_column={self.track_column}, station_column={self.station_column}, event_column={self.event_column}\n'
                f'{"-" * 30}\n'
                f'Number of broken tracks: {self.get_num_broken()} \n')

    
@gin.configurable
class DropEmptyFirstStation(BaseFilter):
    """Drops tracks with no points on first station.
      # Args:
        keep_fakes (bool, True by default ): If True, points with no tracks are preserved, else they are deleted from data.
        station_col (str, 'station' by default): Event column in data
        track_col(str, 'track' by default): Track column in data
        event_col (str, 'event' by default): Station column in data
    """

    def __init__(self,
                 keep_filtered=True,
                 station_col='station',
                 track_col='track',
                 event_col='event',
                 min_station_num=0):

        self.filter = lambda x: min_station_num in x[self.station_column].values
        super().__init__(self.filter, station_col=station_col, track_col=track_col, event_col=event_col,
                         keep_filtered=keep_filtered)

    def __repr__(self):
        return(f'{"-" * 30}\n'
               f'{self.__class__.__name__} with parameters:'
               f'    track_column={self.track_column}, station_column={self.station_column}, event_column={self.event_column}\n'
               f'{"-" * 30}\n'
               f'Number of broken tracks: {self.get_num_broken()} \n')


@gin.configurable
class DropFakes(BaseTransformer):
    """Drops points without tracks (marked as -1).
    Args:
        track_col (str, 'track' by default): Track column in data
    """

    def __init__(self, track_col='track'):
        super().__init__(track_col=track_col)
        self._num_fakes = None
        self.track_col = track_col

    def __call__(self, data):
        """"
        # Args:
            data (pd.DataFrame):  to clean up.
        # Returns:
            data (pd.DataFrame): transformed dataframe
        """
        #data = data[data.station < 9]
        data = self.drop_fakes(data)
        return data

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with parameters: track_col={self.track_col}'
                f'{"-" * 30}\n'
                f'Number of misses: {self.get_num_fakes()} \n')


@gin.configurable
class ToCylindrical(BaseCoordConverter):
    """Convertes data to polar coordinates. Note that cartesian coordinates are used in reversed order!
       Formula used: r = sqrt(x^2 + y^2), phi = atan2(x,y)

       # Args:
           drop_old (boolean, False by default): If True, old coordinate features are deleted from data
           cart_columns (list or tuple of length 3,  ['x', 'y', 'z'] by default ): columns of x, y and z in cartesian coordiates
           polar_columns = (list or tuple of length 3, ['r','phi', 'z'] by default):  columns of r and phi (and redundant z) in cylindrical coordinates
       New "z" column (same value for each station) will be r for cylindrical chamber.
    """

    def __init__(self, drop_old=False, cart_columns=('x', 'y', 'z'), polar_columns=('r', 'phi', 'z'),
                 postfix='before_cyl'):
        super().__init__(self.convert, drop_old=drop_old, from_columns=cart_columns, to_columns=polar_columns,
                         postfix=postfix)

    def convert(self, data):
        r = np.sqrt(data[self.from_columns[0]] ** 2 + data[self.from_columns[1]] ** 2)
        phi = np.arctan2(data[self.from_columns[0]], data[self.from_columns[1]])
        z = data[self.from_columns[2]]
        return (r, phi, z)

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with parameters: drop_old={self.drop_old}, '
                f'from_columns={self.from_columns}, to_columns={self.to_columns}\n'
                f'{"-" * 30}\n'
                f' Ranges: {self.get_ranges_str()} ')


@gin.configurable
class ToCartesian(BaseCoordConverter):
    """Converts coordinates to cartesian. Formula is: y = r * cos(phi), x = r * sin(phi).
    Note that always resulting columns are x,y,z. z column after convertion has same values as before.
      # Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        cart_columns (list or tuple of length 3,  ['x', 'y', 'z'] by default ): columns of x and y in cartesian coordiates
        polar_columns = (list or tuple of length 3, ['r','phi','z'] by default):  columns of r and phi in cylindrical coordiates

    """

    def __init__(self, drop_old=True, cart_columns=('x', 'y', 'z'), polar_columns=('r', 'phi', 'z')):
        self.from_columns = polar_columns
        self.to_columns = cart_columns
        super().__init__(self.convert, drop_old=drop_old, from_columns=self.from_columns, to_columns=self.to_columns)

    def convert(self, data):
        y_new = data[self.from_columns[0]] * np.cos(data[self.from_columns[1]])
        x_new = data[self.from_columns[0]] * np.sin(data[self.from_columns[1]])
        z_new = data[self.from_columns[2]]
        return (x_new, y_new, z_new)

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with parameters: '
                f'drop_old={self.drop_old}, phi_col={self.to_columns[1]}, r_col={self.to_columns[0]}\n'
                f'{"-" * 30}\n'
                f'Ranges: {self.get_ranges_str()}')


@gin.configurable
class ToBuckets(BaseTransformer):
    """Data may contains from tracks with varying lengths.
    To prepare-hydra-wombat a train dataset in a proper way, we have to
    split data on so-called buckets. Each bucket includes
    tracks based on their length, as we can't predict the
    6'th point of the track with length 4, but we can predict
    3-d point

        # Args:
            flat (boolean, True by default): If True, converted data is single dataframe
                            with additional column, else it is dict of dataframes
            random_state (int, 42 by default): seed for the RandomState
            shuffle (boolean, False by default): whether or not shuffle output dataset.
            keep_fakes (boolean, True by default):  . If True,
                     points without tracks are preserved.
            event_col (string, 'event' by default): Column with event data.
            track_col (string, 'event' by default): Column with track numbers.

    """

    def __init__(self, flat=True, shuffle=False, max_stations=None, random_state=42,
                 event_col='event', track_col='track', keep_fakes=False, max_bucket_size=None):
        super().__init__(event_col=event_col, track_col=track_col)
        self.flat = flat
        self.shuffle = shuffle
        self.random_state = random_state
        self.max_num_stations = max_stations
        self.keep_fakes = keep_fakes
        self.max_bucket_size = max_bucket_size

    def __call__(self, df):
        """
        # Args:
            data (pd.DataFrame): data to clean up.
        # Returns:
            data (pd.DataFrame or dict(len:pd.DataFrame): transformed dataframe,
            if flat is True, returns dataframe with specific column, else dict with bucket dataframes
        """
        # assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        df['index'] = df.index
        rs = np.random.RandomState(self.random_state)
        groupby = df.groupby([self.event_column, self.track_column])
        maxlen = groupby.size().max()
        if self.max_num_stations is None:
            self.max_num_stations = maxlen
        minlen = max(groupby.size().min(), 3)
        subbuckets = {}
        res = {}
        val_cnt = groupby.size().unique()  # get all unique track lens (in BES3 all are 3)
        val_cnt = range(minlen, max(maxlen + 1, val_cnt.max()))
        print(val_cnt)
        for length in val_cnt:
            this_len = groupby.filter(lambda x: x.shape[0] == length)
            if len(this_len) > 0:
                this_len_groups = this_len.groupby([self.event_column, self.track_column])
                bucket_index = np.stack(list(this_len_groups['index'].agg(lambda x: list(x.values))), axis=0)
            else:
                bucket_index = []
            subbuckets[length] = bucket_index
        print(subbuckets)
        # approximate size of the each bucket
        bsize = len(df) // (self.max_num_stations - 2)
        print(bsize)
        if self.max_bucket_size is not None:
            bsize = min(bsize, self.max_bucket_size)
        buckets = {i: [] for i in range(3, self.max_num_stations + 1)}
        print(buckets)
        # reverse loop until two points
        for n_points in range(self.max_num_stations, minlen - 1, -1):
            print(n_points, maxlen)
            # while bucket is not full
            k = n_points
            if n_points not in buckets.keys():
                continue
            while len(buckets[n_points]) < bsize:
                if k < n_points or k > maxlen:
                    break
                if self.shuffle:
                    rs.shuffle(subbuckets[k])
                # if we can't extract all data from subbucket
                # without bsize overflow
                if len(buckets[n_points]) + len(subbuckets[k]) > bsize:
                    n_extract = bsize - len(buckets[n_points])
                    # extract n_extract samples
                    buckets[n_points].extend(subbuckets[k][:n_extract, :n_points])
                    print(buckets[n_points])
                    # remove them from original subbucket
                    subbuckets[k] = subbuckets[k][n_extract:]
                    print(subbuckets)
                else:
                    if len(subbuckets[k]) == 0:
                        k += 1
                        continue
                    buckets[n_points].extend(subbuckets[k][:, :n_points])
                    # remove all data from the original list
                    subbuckets[k] = []
                    # increment index
                k += 1
                print(buckets)
                if all([len(subbuckets[i]) == 0 for i in range(n_points, maxlen + 1) if i in subbuckets.keys()]):
                    break

            if all([len(subbuckets[i]) == 0 for i in subbuckets.keys()]):
                break
        # append unappended items
        for i, k in subbuckets.items():
            if len(k) > 0:
                try:
                    append_len = int(len(k) / (i - 2))
                    begin = 0
                    for j in range(i - 1, min(list(buckets.keys())) - 1, -1):
                        print(k[begin:begin + append_len])
                        buckets[j].extend(k[begin:begin + append_len, :j])
                        begin = begin + append_len
                except:
                    print('alert!')
        buckets = {k: np.concatenate(i) for k, i in buckets.items() if len(i) > 0}
        self.buckets_ = buckets
        if self.flat is True:
            res = df.copy()
            res['bucket'] = 0
            for i, bucket in buckets.items():
                res.loc[bucket, 'bucket'] = i
            if self.keep_fakes:
                self.fakes.loc[:, 'bucket'] = -1
                res = super().add_fakes(data)
        else:
            res = {i: df.loc[bucket] for i, bucket in buckets.items()}
            if self.keep_fakes:
                res[-1] = self.fakes
        return res

    def get_bucket_index(self):
        """
        # Returns: dict(len: indexes) - dict with lens and list of indexes in bucket
        """
        return self.buckets_

    def get_buckets_sizes(self):
        """

        # Returns:
            {bucket:len} dict with length of data in bucket
        """
        return {i: len(j) for i, j in self.buckets_.items()}

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with parameters: flat={self.flat}, '
                f'random_state={self.random_state}, shuffle={self.shuffle}, keep_fakes={self.keep_fakes}\n'
                f'{"-" * 30}\n')


@gin.configurable
class FixStationsBMN(BaseTransformer):
    """Renumbers stations from local number by detector to global, use before other transforms.
    Args:
        det_col (str, 'det' by default): Detector column in data
        station_col (str, 'station' by default): Station column in data
    """
    
    def __init__(self, det_col='det', station_col='station'):
        super().__init__(station_col=station_col)
        self.det_col = det_col
        self.station_col = station_col
        
    def __call__(self, data):
        """
        Args:
            data (pd.DataFrame): to transform.
        Returns:
            data (pd.DataFrame): transformed dataframe.
        """
        data.loc[data[self.det_col] == 1, self.station_col] = data.loc[data[self.det_col] == 1, self.station_col].values + 3
        return data
    
    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with parameters: det_col={self.det_col}, station_col={self.station_col}'
                f'{"-" * 30}\n')


@gin.configurable
class AddVirtualPoints:
    """Adds virtual points for tracks with holes using given TrackNET model.
    Args:
        model (TrackNETv2): TrackNET instance
        z_values: values for stations
    """

    def __init__(self, model_loader, z_values, device, columns=['x', 'y', 'z']):
        self.model = model_loader()[1][0]
        self.z_values = z_values
        self.columns = columns
        self.device = device

    def __call__(self, df):
        grouped_df = df[df['track'] != -1].groupby(['track', 'event'])
        #print(df)
        print(len(df))
        new_hits = []
        max_index = df.index.max()
        for i, data in tqdm(grouped_df):
            if len(data) != data['station'].max() + 1:
                new_hits.append(self.get_new_hits(data, max_index))
                max_index += len(new_hits[-1])
        df = pd.concat([df, *new_hits], ignore_index=True)
        print(len(df))
        return df

    def get_new_hits(self, data, max_index):
        track_len = data['station'].max() + 1
        track = np.column_stack([self.z_values]*3)
        track[data['station']] = data[list(self.columns)].values
        track = track[:track_len]
        track_torch = torch.unsqueeze(torch.from_numpy(track), 0).to(self.device)
        mask = torch.from_numpy((track[:, 0] != track[:, 1])[:track_len].reshape(1, -1)).to(self.device)
        new_track = self.model(track_torch, torch.tensor([track_len], dtype=torch.int64), mask=mask, return_x=True)
        new_hits = new_track[~mask].cpu().numpy()

        new_df = pd.DataFrame({
            'event': [data.event.iloc[0]] * len(new_hits),
            'x': new_hits[:, 0],
            'y': new_hits[:, 1],
            'z': new_hits[:, 2],
            'station': np.where(~(track[:, 0] != track[:, 1])[:track_len])[0],
            'track': [data.track.iloc[0]] * len(new_hits),
            'index': list(range(max_index + 1, max_index + 1 + len(new_hits)))
        })
        return new_df

    def __repr__(self):
        return (f'{"-" * 30}\n'
                f'{self.__class__.__name__} with parameters: model={self.model}, z_values={self.z_values}, device={self.device}'
                f'{"-" * 30}\n')

    
    
@gin.configurable
class DropOverPhi(BaseFilter):
    """Drops tracks with no points on first station.
      # Args:
        keep_fakes (bool, True by default ): If True, points with no tracks are preserved, else they are deleted from data.
        station_col (str, 'station' by default): Event column in data
        track_col(str, 'track' by default): Track column in data
        event_col (str, 'event' by default): Station column in data
    """

    def __init__(self,
                 keep_filtered=True,
                 station_col='station',
                 track_col='track',
                 event_col='event',
                 min_station_num=0):

        self.filter = self.filter_overphi
        super().__init__(self.filter, station_col=station_col, track_col=track_col, event_col=event_col,
                         keep_filtered=keep_filtered)
        
    def filter_overphi(self, x):
        differ = x.phi.values[1:] - x.phi.values[:-1]
        return (np.abs(differ) < 1).all()

    def __repr__(self):
        return(f'{"-" * 30}\n'
               f'{self.__class__.__name__} with parameters:'
               f'    track_column={self.track_column}, station_column={self.station_column}, event_column={self.event_column}\n'
               f'{"-" * 30}\n'
               f'Number of broken tracks: {self.get_num_broken()} \n')
    
    
@gin.configurable
class CombineEvents(object):
    """Combine N events to one.
    """

    def __init__(self, combine_n=1, track_col='track', event_col='event'):
        self.combine_n = combine_n
        self.event_column = event_col
        self.track_column = track_col

    def __call__(self, data):
        data[self.track_column][data[self.track_column] >= 0] = data[self.track_column][data[self.track_column] >= 0] + 10 * data[self.event_column][data[self.track_column] >= 0]
        n_events = len(data[self.event_column].unique()) // self.combine_n
        data[self.event_column] = data[self.event_column] % n_events
        return data

    def __repr__(self):
        return(f'{"-" * 30}\n'
               f'{self.__class__.__name__} with parameters:'
               f'    track_column={self.track_column}, event_column={self.event_column}\n'
               f'{"-" * 30}\n'
               f'Number of combined events: {self.combine_n} \n')