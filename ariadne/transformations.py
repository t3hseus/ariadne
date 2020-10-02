import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from copy import deepcopy


class Compose(object):
    """Composes several transforms together.
    # Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    # Example:
        >>> Compose([
        >>>     transforms.StandardScale(),
        >>>     transforms.ToCylindrical(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        """
        # Returns: formatted strings with class_names, parameters and some statistics for each class
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '{0} \n'.format(t)
        format_string += '\n)'
        return format_string


class BaseTransformer(object):
    """Base class for transforms
    # Args:
         columns (list or tuple, ['x', 'y', 'z'] by default): Columns to transform
         drop_old (boolean, True by default): If True, original data is discarded,
                                            else preserved in columns with suffix '_old'
         keep_fakes (boolean, True by default): If True, hits with no track are preserved
    """

    def __init__(self, drop_old=False, keep_fakes=True, columns=('x', 'y', 'z'), track_col='track', event_col='event', station_col='station'):
        self.drop_old = drop_old
        self.columns = columns
        self.keep_fakes = keep_fakes
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
        if self.keep_fakes:
            self.fakes = data.loc[data[self.track_column] == -1, :]
        return data.loc[data[self.track_column] != -1, :]

    def get_num_fakes(self):
        return len(self.fakes)

    def add_fakes(self, data):
        return pd.concat([data, self.fakes], axis=0).reset_index()


class BaseScaler(BaseTransformer):
    """Base class for scalers.
     # Args:
         scaler (function or method pd.DataFrame -> iterable of pd.Series): scaler with fit_predict method
         columns (list or tuple, ['x', 'y', 'z'] by default): Columns to scale
         drop_old (boolean, True by default): If True, unscaled data is discarded,
                                            else preserved in columns with suffix '_old'
    """

    def __init__(self, scaler, drop_old=True, columns=('x', 'y', 'z')):
        self.columns = columns
        self.scaler = scaler
        super().__init__(drop_old=drop_old, columns=columns)

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
        return '-' * 30 + '\n' + f'{self.__class__.__name__} with scaler: {self.scaler} \n' + '-' * 30 + '\n'


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

    def __init__(self, filter_rule, num_stations=None, keep_fakes=True, station_col='station', track_col='track',
                 event_col='event'):
        self.num_stations = num_stations
        self._broken_tracks = None
        self._num_broken_tracks = None
        self.keep_fakes = keep_fakes
        self.filter_rule = filter_rule
        super().__init__(keep_fakes=keep_fakes, station_col=station_col, track_col=track_col, event_col=event_col)

    def __call__(self, data):
        """
        # Args:
            data (pd.DataFrame):  to clean up.
        # Returns:
            data (pd.DataFrame): transformed dataframe
        """
        # assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        data = super().drop_fakes(data)
        tracks = data.groupby([self.event_column, self.track_column])
        if self.num_stations is None:
            self.num_stations = tracks.size().max()
        good_tracks = tracks.filter(self.filter_rule)
        broken = list(data.loc[~data.index.isin(good_tracks.index)].index)
        self._broken_tracks = data.loc[broken, [self.event_column, self.track_column, self.station_column]]
        self._num_broken_tracks = len(self._broken_tracks[[self.event_column, self.track_column]].drop_duplicates())
        good_tracks = super().add_fakes(good_tracks)
        return good_tracks

    def get_broken(self):
        return self._broken_tracks

    def get_num_broken(self):
        return self._num_broken_tracks

    def __repr__(self):
        return '-' * 30 + '\n' + f'{self.__class__.__name__} with filter_rule: {self.filter_rule}\n' + '-' * 30 + '\n'


class BaseCoordConverter(BaseTransformer):
    """Base class for coordinate convertions

    # Args:
         convert_function(function or method pd.DataFrame -> iterable of pd.Series): Function, which
         convertes data, returned value must be iterable with pd.Series values (list etc)
         drop_old (boolean, False by default): If True, old columns are discarded from data
         from_columns (list or tuple, ['x', 'y', 'z'] by default): list of original features
         to_columns (list or tuple, ['r', 'phi'] by default): list of features to convert to
    """

    def __init__(self, convert_function, drop_old=False, from_columns=('x', 'y'),
                 to_columns=('r', 'phi')):
        self.drop_old = drop_old
        self.cart_columns = from_columns
        self.polar_columns = to_columns
        self.convert_function = convert_function
        self.range_ = {}
        self.from_columns = from_columns
        self.to_columns = to_columns
        super().__init__(drop_old=drop_old, columns=list(to_columns) + ['z'])

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
        for i in range(len(self.to_columns)):
            data.loc[:, self.to_columns[i]] = converted[i]
        self.get_ranges(data, self.to_columns)
        if self.drop_old is True:
            for col in self.from_columns:
                del data[col]
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
        return '-' * 30 + '\n' + f'{self.__class__.__name__} with convert_function: {self.convert_function}\n' + '-' * 30 + '\n'


class StandardScale(BaseScaler):
    """Standardizes coordinates by removing the mean and scaling to unit variance
    # Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        with_mean (boolean, True by default): If True, center the data before scaling
        with_std (boolean, True by default): If True, scale the data to unit variance (or equivalently, unit standard deviation).
        columns (list or tuple of length 3): Columns to Standardize
    """

    def __init__(self, drop_old=True, with_mean=True, with_std=True, columns=('x', 'y', 'z')):
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = StandardScaler(with_mean, with_std)
        super().__init__(self.scaler, drop_old, columns)

    def __repr__(self):
        return '-' * 30 + '\n' + \
               f'{self.__class__.__name__} with scaling_parameters: drop_old={self.drop_old}, with_mean={self.with_mean},with_std={self.with_std} \n' + \
               '-' * 30 + '\n' + \
               f' Mean: {self.scaler.mean_} \n Var: {self.scaler.var_} \n Scale: {self.scaler.scale_} '


class MinMaxScale(BaseScaler):
    """Transforms features by scaling each feature to a given range.
     # Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        feature_range (Tuple (min,max), default (0,1)): Desired range of transformed data.
        columns (list or tuple of length 3): Columns to Standardize
    """

    def __init__(self, drop_old=True, feature_range=(0, 1), columns=('x', 'y', 'z')):
        assert feature_range[0] < feature_range[1], 'minimum is not smaller value then maximum'
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.columns = columns
        self.drop_old = drop_old
        super().__init__(self.scaler, drop_old, columns=columns)

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: drop_old={self.drop_old}, feature_range={self.feature_range} \n' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f' Data min: {self.scaler.data_min_} \n Data max: {self.scaler.data_max_} \n Scale: {self.scaler.scale_} '


class Normalize(BaseScaler):
    """Normalizes samples individually to unit norm.
    Each sample (i.e. each row of the data matrix) with at least one non zero component is rescaled independently of
    other samples so that its norm (l1, l2 or inf) equals one.

      # Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        norm (‘l1’, ‘l2’, or ‘max’ (‘l2’ by default)): The norm to use to normalize each non zero sample.
                              If norm=’max’ is used, values will be rescaled by the maximum of the absolute values.
        columns (list or tuple of length 3): Columns to Standardize
    """

    def __init__(self, drop_old=True, norm='l2', columns=('x', 'y', 'z')):
        self.norm = norm
        self.scaler = Normalizer(norm=norm)
        self.columns = columns
        self.drop_old = drop_old
        super().__init__(self.scaler, drop_old, columns)

    def __repr__(self):
        return '-' * 30 + '\n' + f'{self.__class__.__name__} with parameters: drop_old={self.drop_old}, norm={self.norm} \n' + \
               '-' * 30 + '\n'


class ConstraintsNormalize(BaseTransformer):
    """Normalizes samples using station given characteristics  or computes them by call.
    If you need to compute characteristics, you can use MinMaxScale too (maybe better)
    Each station can have its own constraints or global constrains.
      Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        columns (list or tuple of length 3): Columns to scale
        margin (number, positive): margin applied to stations (min = min-margin, max=max+margin)
        constraints (dict, None by deault) If None, constraints are computed using dataset statistics.
        use_global_constraints (boolean, True by default) If True, all data is scaled using given global constraints.

    If use_global_constraints is True and constraints is not None, constraints must be {column:(min,max)},
    else it must be {station: {column:(min,max)}}.

    Station keys must be in dataset.
    """

    def __init__(self, drop_old=True, columns=('x', 'y', 'z'), margin=1e-3, use_global_constraints=True,
                 constraints=None):
        assert margin > 0, 'Margin is not positive'
        self.columns = columns
        self.drop_old = drop_old
        self.margin = margin
        self.use_global_constraints = use_global_constraints
        self.constraints = constraints
        if constraints is not None:
            if use_global_constraints:
                for col in columns:
                    assert col in constraints.keys(), f'{col} is not in constraint keys'
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
        super().__init__(drop_old=drop_old, columns=columns)

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
                global_min = min([x[col][0] for x in self.constraints.values()])
                global_max = max([x[col][1] for x in self.constraints.values()])
                global_constrains[col] = (global_min, global_max)
            x_norm, y_norm, z_norm = self.normalize(data, global_constrains)
            data = super().transform_data(data, [x_norm, y_norm, z_norm])
        else:
            assert all([station in data['station'].unique() for station in
                        self.constraints.keys()]), 'Some station keys in constraints are not presented in data'
            for station in self.constraints.keys():
                group = data.loc[data['station'] == station,]
                x_norm, y_norm, z_norm = self.normalize(group, self.constraints[station])
                data.loc[data['station'] == station, :] = \
                    self.transform_data_by_group(data['station'] == station, data,
                                                    [x_norm, y_norm, z_norm])
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
        print(x_min, x_max)
        y_min, y_max = constraints[self.columns[1]]
        print(y_min, y_max)
        z_min, z_max = constraints[self.columns[2]]
        print(z_min, z_max)
        assert all(df[self.columns[0]].between(x_min, x_max)), \
            f'Some values in column {self.columns[0]} are not in {constraints[self.columns[0]]}'
        x_norm = 2 * (df[self.columns[0]] - x_min) / (x_max - x_min) - 1
        assert all(df[self.columns[1]].between(y_min, y_max)), \
            f'Some values in column {self.columns[1]} are not in {constraints[self.columns[1]]}'
        y_norm = 2 * (df[self.columns[1]] - y_min) / (y_max - y_min) - 1
        assert all(df[self.columns[2]].between(z_min, z_max)), \
            f'Some values in column {self.columns[2]} are not in {constraints[self.columns[2]]}'
        z_norm = 2 * (df[self.columns[2]] - z_min) / (z_max - z_min) - 1
        return x_norm, y_norm, z_norm

    def __repr__(self):
        return '-' * 30 + '\n' + \
               f'{self.__class__.__name__} with parameters: drop_old={self.drop_old}, use_global_constraints={self.use_global_constraints} \n' + \
               f'                                             margin={self.margin}, columns={self.columns}\n ' + \
               '-' * 30 + '\n' + \
               f'constraints are: {self.constraints}'


class DropShort(BaseFilter):
    """Drops tracks with num of points less then given from data.
      # Args:
        num_stations (int, default None): Desired number of stations (points). If None, maximum stations number for one track is taken from data.
        keep_fakes (bool, default True): If True, points with no tracks are preserved, else they are deleted from data.
        station_column (str, 'station' by default): Event column in data
        track_column (str, 'track' by default): Track column in data
        event_column (str, 'event' by default): Station column in data
    """

    def __init__(self, num_stations=0, keep_fakes=True, station_col='station', track_col='track', event_col='event'):
        self.num_stations = num_stations
        self.broken_tracks_ = None
        self.num_broken_tracks_ = None
        self.filter = lambda x: x.shape[0] >= self.num_stations
        super().__init__(self.filter, station_col=station_col, track_col=track_col,
                         event_col=event_col, keep_fakes=keep_fakes)

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: num_stations={self.num_stations}, ' \
               f'keep_fakes={super().keep_fakes}, \n' + \
               f'    track_column={super().track_column}, station_column={super().station_column}, ' \
               f'event_column={super().event_column}\n' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f'Number of broken tracks: {super().get_num_broken()} \n'


class DropSpinningTracks(BaseFilter):
    """Drops tracks with points on same stations (e.g. (2,2,2) or (1,2,1)).
      # Args:
        keep_fakes (bool, True by default ): If True, points with no tracks are preserved, else they are deleted from data.
        station_col (str, 'station' by default): Event column in data
        track_col(str, 'track' by default): Track column in data
        event_col (str, 'event' by default): Station column in data
    """

    def __init__(self, keep_fakes=True, station_col='station', track_col='track', event_col='event'):
        self.filter = lambda x: x[self.station_column].unique().shape[0] == x[self.station_column].shape[0]
        super().__init__(self.filter, station_col=station_col, track_col=track_col, event_col=event_col,
                         keep_fakes=keep_fakes)

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters:' + \
               f'    track_column={super().track_column}, station_column={super().station_column}, event_column={super().event_column}\n' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f'Number of broken tracks: {super().get_num_broken()} \n'


class DropFakes(BaseTransformer):
    """Drops points without tracks.
    Args:
        track_col (str, 'track' by default): Track column in data
    """

    def __init__(self, track_col='track'):
        self._num_fakes = None
        self.track_col = track_col
        super().__init__(track_col=track_col, keep_fakes=False)

    def __call__(self, data):
        """"
        # Args:
            data (pd.DataFrame):  to clean up.
        # Returns:
            data (pd.DataFrame): transformed dataframe
        """
        # assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        data = self.drop_fakes(data)
        return data

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: track_col={self.track_col}' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f'Number of misses: {self.get_num_fakes()} \n'


class ToCylindrical(BaseCoordConverter):
    """Convertes data to polar coordinates. Note that cartesian coordinates are used in reversed order!
       Formula used: r = sqrt(x^2 + y^2), phi = atan2(x,y)

       # Args:
           drop_old (boolean, False by default): If True, old coordinate features are deleted from data
           cart_columns (list or tuple,  ['x', 'y'] by default ): columns of x and y in cartesian coordiates
           polar_columns = (list or tuple, ['r','phi'] by default):  columns of r and phi in cylindrical coordiates
    """

    def __init__(self, drop_old=False, cart_columns=('x', 'y'), polar_columns=('r', 'phi')):
        self.drop_old = drop_old
        self.from_columns = cart_columns
        self.to_columns = polar_columns
        super().__init__(self.convert, drop_old=drop_old, from_columns=cart_columns, to_columns=polar_columns)

    def convert(self, data):
        r = np.sqrt(data[self.from_columns[0]] ** 2 + data[self.from_columns[1]] ** 2)
        phi = np.arctan2(data[self.from_columns[0]], data[self.from_columns[1]])
        return (r, phi)

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: drop_old={self.drop_old}, ' + \
               f'from_columns={self.from_columns}, to_columns={self.to_columns}\n' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f' Ranges: {super().get_ranges_str()} '


class ToCartesian(BaseCoordConverter):
    """Converts coordinates to cartesian. Formula is: y = r * cos(phi), x = r * sin(phi).
    Note that always resulting columns are x,y,z.
      # Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        cart_columns (list or tuple,  ['x', 'y'] by default ): columns of x and y in cartesian coordiates
        polar_columns = (list or tuple, ['r','phi'] by default):  columns of r and phi in cylindrical coordiates
    """

    def __init__(self, drop_old=True, cart_columns=('x', 'y'), polar_columns=('r', 'phi')):
        self.from_columns = polar_columns
        self.to_columns = cart_columns
        super().__init__(self.convert, drop_old=drop_old, from_columns=self.from_columns, to_columns=self.to_columns)

    def convert(self, data):
        y_new = data[self.from_columns[0]] * np.cos(data[self.from_columns[1]])
        x_new = data[self.from_columns[0]] * np.sin(data[self.from_columns[1]])
        return (x_new, y_new)

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: ' \
               f'drop_old={self.drop_old}, phi_col={self.to_columns[1]}, r_col={self.to_columns[0]}\n' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f'Ranges:' + super().get_ranges_str()


class ToBuckets(BaseTransformer):
    """Data may contains from tracks with varying lengths.
    To prepare a train dataset in a proper way, we have to
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

    def __init__(self, flat=True, shuffle=False, random_state=42,
                 keep_fakes=False, event_col='event', track_col='track'):
        self.flat = flat
        self.shuffle = shuffle
        self.random_state = random_state
        self.track_col = track_col
        self.event_col = event_col
        self.keep_fakes = keep_fakes
        super().__init__(keep_fakes=keep_fakes, event_col=event_col, track_col=track_col)

    def __call__(self, df):
        """
        # Args:
            data (pd.DataFrame): data to clean up.
        # Returns:
            data (pd.DataFrame or dict(len:pd.DataFrame): transformed dataframe,
            if flat is True, returns dataframe with specific column, else dict with bucket dataframes
        """
        # assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        data = self.drop_fakes(df)
        rs = np.random.RandomState(self.random_state)
        groupby = df.groupby([self.event_col, self.track_col])
        maxlen = groupby.size().max()
        n_stations_min = groupby.size().min()
        subbuckets = {}
        res = {}
        val_cnt = groupby.size().unique()  # get all unique track lens (in BES3 all are 3)
        for length in val_cnt:
            this_len = groupby.filter(lambda x: x.shape[0] == length)
            bucket_index = list(df.loc[df.index.isin(this_len.index)].index)
            subbuckets[length] = bucket_index
        # approximate size of the each bucket
        bsize = len(df) // (maxlen - 2)
        # set index
        k = maxlen
        buckets = {i: [] for i in subbuckets.keys()}
        # reverse loop until two points
        for n_points in range(maxlen, 2, -1):
            # while bucket is not full
            while len(buckets[n_points]) < bsize:
                if self.shuffle:
                    rs.shuffle(subbuckets[k])
                # if we can't extract all data from subbucket
                # without bsize overflow
                if len(buckets[n_points]) + len(subbuckets[k]) > bsize:
                    n_extract = bsize - len(buckets[n_points])
                    # extract n_extract samples
                    buckets[n_points].extend(subbuckets[k][:n_extract])
                    # remove them from original subbucket
                    subbuckets[k] = subbuckets[k][n_extract:]
                else:
                    buckets[n_points].extend(subbuckets[k])
                    # remove all data from the original list
                    subbuckets[k] = []
                    # decrement index
                    k -= 1
        self.buckets_ = buckets
        if self.flat is True:
            res = copy(df)
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
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: flat={self.flat}, ' \
               f'random_state={self.random_state}, shuffle={self.shuffle}, keep_fakes={self.keep_fakes}\n' + \
               '------------------------------------------------------------------------------------------------\n'
