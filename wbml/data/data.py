# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import csv
import os
import pickle
import os

import numpy as np
from lab import B
from plum import Dispatcher, Referentiable, Self
from dateutil.parser import parse

__all__ = ['Data', 'CSVReader', 'normalise_norm', 'normalise_01']

dispatch = Dispatcher()


@dispatch(object)
def ensure_numeric(x):
    """Ensure that an object is numeric.

    Args:
        x (object): Object to make numeric.

    Returns:
        tensor: Input as tensor.
    """
    return np.array(x)


@dispatch(B.Numeric)
def ensure_numeric(x):
    return x


class Data(Referentiable):
    """Data set.

    Args:
        x (tensor): Inputs.
        y (tensor): Outputs.
        names: Names of the outputs.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, x, y, names):
        x = ensure_numeric(x)
        y = ensure_numeric(y)

        # Ensure that `x` and `y` are at least two dimensional.
        x = x[:, None] if B.rank(x) == 1 else x
        y = y[:, None] if B.rank(y) == 1 else y

        self.x = x
        self.y = y
        self.names = names

    @property
    def n(self):
        """Number of data points."""
        return np.shape(self.x)[0]

    @property
    def m(self):
        """Number of features."""
        return np.shape(self.x)[1]

    @property
    def p(self):
        """Number of outputs."""
        return np.shape(self.y)[1]

    def __getitem__(self, item):
        return Data(self.x[item], self.y[item], self.names)

    def select(self, *names):
        """Select particular outputs.

        Args:
            *names: Outputs to select.

        Returns:
            tuple[:class:`.data.Data`]: Selected outputs and the remainder.
        """
        sel_inds = [self.names.index(name) for name in names]
        rem_inds = sorted(set(range(len(self.names))) - set(sel_inds))
        return Data(self.x, self.y[:, sel_inds],
                    list(np.take(self.names, sel_inds))), \
               Data(self.x, self.y[:, rem_inds],
                    list(np.take(self.names, rem_inds)))

    def shuffle(self):
        """Shuffle the data.

        Returns:
            :class:`.data.Data`: Shuffled data.
        """
        return self[np.random.permutation(self.n)]

    @_dispatch(float)
    def split(self, f):
        """Split the data.

        Args:
            point (int or float): Point to split the data at. Can either be an
                int, representing an index, or a float, representing a
                fractional point.

        Returns:
            tuple[:class:`.data.Data`]: First and second part of the data.
        """
        return self.split(int(np.round(self.n * f)))

    @_dispatch(int)
    def split(self, i):
        return Data(self.x[:i], self.y[:i], self.names), \
               Data(self.x[i:], self.y[i:], self.names)

    def truncate(self, x):
        """Truncate the data. Takes in arguments like :meth:`.data.Data.split`.

        Returns:
            :class:`.data.Data`: Truncated data.
        """
        return self.split(x)[0]

    def interval(self,
                 x_start,
                 x_end,
                 start_inclusive=True,
                 end_inclusive=False):
        """Select a interval of the data.

        Args:
            x_start: Start of the interval.
            x_end: End of the interval.
            start_inclusive (bool, optional): The start value is inclusive.
                Defaults to `True`.
            end_inclusive (bool, optional): The end value is inclusive.
                Defaults to `False`.

        Returns:
            tuple[:class:`.data.Data`]: Selected interval and the entire data
                with the interval set to missing.
        """
        mask = (x_start <= self.x if start_inclusive else x_start < self.x) & \
               (self.x <= x_end if end_inclusive else self.x < x_end)
        mask = np.all(mask, axis=1)
        y_no_interval = self.y.copy()
        y_no_interval[mask] = np.nan
        return self[mask], Data(self.x, y_no_interval, self.names)

    def concat_y(self, other):
        """Concatenate outputs from another data set at the current inputs.

        Args:
            other (:class:`.data.Data`): Other data set.

        Returns:
            :class:`.data.Data`: Concatenation.
        """
        conflicting_names = list(set(self.names) & set(other.names))
        if len(conflicting_names) > 0:
            raise RuntimeError('Conflicting names: {}.'
                               ''.format(', '.join(conflicting_names)))
        return Data(self.x,
                    np.concatenate((self.y, other.y), axis=1),
                    self.names + other.names)

    def concat_xy(self, other):
        """Concatenate outputs from another data set at new inputs.

        Args:
            other (:class:`.data.Data`): Other data set.

        Returns:
            :class:`.data.Data`: Concatenation.
        """
        nans = np.zeros((other.n, self.p))
        nans.fill(np.nan)
        self_ext = Data(np.concatenate((self.x, other.x), axis=0),
                        np.concatenate((self.y, nans), axis=0),
                        self.names)

        nans = np.zeros((self.n, other.p))
        nans.fill(np.nan)
        other_ext = Data(np.concatenate((self.x, other.x), axis=0),
                         np.concatenate((nans, other.y), axis=0),
                         other.names)

        return self_ext.concat_y(other_ext)

    def merge_x(self):
        """Safely merge outputs for duplicate inputs.

        Returns:
            :class:`.data.Data`: Data with duplicate inputs merged.
        """

        def merge(x, y):
            z = []
            for x_i, y_i in zip(x, y):
                if np.isnan(x_i) and np.isnan(y_i):
                    z.append(np.nan)
                elif np.isnan(x_i) and not np.isnan(y_i):
                    z.append(y_i)
                elif not np.isnan(x_i) and np.isnan(y_i):
                    z.append(x_i)
                else:
                    raise RuntimeError('Merge conflict.')
            return np.array(z)

        # Skip these indices because they already have been merged.
        skip_is = []
        merged_x, merged_y = [], []

        for i in range(self.n):
            # Skip already merged points.
            if i in skip_is:
                continue

            # Merge duplicate further values and append to merged result.
            x_i, y_i = self.x[i], self.y[i]
            for j in range(i + 1, self.n):
                if np.allclose(x_i, self.x[j]):
                    y_i = merge(y_i, self.y[j])
                    skip_is.append(j)
            merged_x.append(x_i)
            merged_y.append(y_i)

        # Construct result and return.
        return Data(np.stack(merged_x, axis=0),
                    np.stack(merged_y, axis=0),
                    self.names)

    @_dispatch({list, tuple}, [object])
    def interval_y(self, names, *args, **kw_args):
        """Select an interval from a particular output. Further takes in
        arguments for :meth:`.data.Data.interval`.

        Args:
            *names (str): Outputs to select the interval from.

        Returns:
            tuple[:class:`.data.Data`]: The selected interval and the remainder.
        """
        output, other_outputs = self.select(*names)
        interval, remainder = output.interval(*args, **kw_args)
        merged = other_outputs.concat_y(remainder).select(*self.names)[0]
        return interval, merged

    @_dispatch([object])
    def interval_y(self, name, *args, **kw_args):
        return self.interval_y([name], *args, **kw_args)

    def normalise(self, ref=None, unnormaliser=False):
        """Normalise data.

        Args:
            ref (:class:`.data.Data`, optional): Reference to normalise with
                respect to. Defaults to itself.
            unnormaliser (bool, optional): Also return a function to
                unnormalise data. Defaults to `False`.

        Returns:
            :class:`.data.Data`: Normalised data. Can further return a
                function to unnormalise data.
        """
        ref = self if ref is None else ref.select(*self.names)[0]
        y_mean = np.nanmean(ref.y, axis=0, keepdims=True)
        y_std = np.nanstd(ref.y, axis=0, keepdims=True)
        data = Data(self.x, (self.y - y_mean) / y_std, self.names)

        # Also return function to unnormalise data if asked for.
        if unnormaliser:
            dispatch = Dispatcher()

            @dispatch(Data)
            def unnormalise(d):
                # Get the right scales and offsets.
                indices = [self.names.index(name) for name in d.names]
                offsets = np.array([y_mean[0, i] for i in indices])
                scales = np.array([y_std[0, i] for i in indices])
                return Data(d.x, d.y * scales + offsets, d.names)

            @dispatch(object)
            def unnormalise(x):
                return x * y_std + y_mean

            @dispatch(object, object)
            def unnormalise(x, name):
                i = self.names.index(name)
                return x * y_std[0, i] + y_mean[0, i]

            return data, unnormalise
        else:
            return data

    def cast(self, dtype):
        """Cast the data to a particular data type.

        Args:
            dtype (dtype): Date type to cast to.

        Returns:
            :class:`.data.Data`: Data casted to `dtype`.
        """
        return Data(B.cast(self.x, dtype), B.cast(self.y, dtype), self.names)

    @staticmethod
    def merge(*ds):
        """Merge a number of data sets.

        Args:
            *ds (:class:`.data.Data`): Data sets to merge.

        Returns:
            :class:`.data.Data`: Merged data set.
        """
        merged = ds[0]
        for d in ds[1:]:
            merged = merged.concat_xy(d)
        return merged.merge_x()


class CSVReader:
    """Read and parse a CSV."""

    def __init__(self):
        self._fields = []
        self._groups = []
        self._current_group = 0

    def set_field_group(self, name, matrix=True):
        """Create a group of fields.

        Args:
            name (str): Name of group.
            matrix (bool, optional): Output group as a matrix instead of a
                dictionary. Defaults to `True`.
        """
        if name not in [x['name'] for x in self._groups]:
            self._groups.append({'name': name,
                                 'matrix': matrix})
        self._current_group = name

    def add_field(self, column_name, type, output_name=None):
        """Add a field.

        Args:
            column_name (str): Name of column.
            type (type or function): Type of column. Can also be a function
                that convert the value into the right type.
            output_name (str, optional): Change the name of the column.
        """
        # If there is no field group yet, make one called zero.
        if len(self._groups) == 0:
            self.set_field_group(0, matrix=True)

        # Default output name to column name.
        output_name = column_name if output_name is None else output_name

        # A list for `type` represents possible discrete values.
        if callable(type):
            converter = type
        elif isinstance(type, list):
            list_of_values = [str(x) for x in type]
            converter = lambda x: list_of_values.index(x)
        else:
            raise ValueError('unknown field type "{}"'.format(type))

        self._fields.append({'column_name': column_name,
                             'output_name': output_name,
                             'converter': converter,
                             'group': self._current_group})

    def read(self, file_name, delimiter=',', cache=False):
        """Read and parse a CSV.

        Args:
            file_name (str): File name.
            delimiter (str, optional): Delimiter. Defaults to a comma.
            cache (bool, optional): Use a cache if available. Defaults to
                `False`.
        """
        cache_name = '{}.cache'.format(file_name)

        # Check for cached data.
        if cache and os.path.exists(cache_name):
            with open(cache_name, 'rb') as f:
                out = pickle.load(f)
        else:
            # Delete cache if it exists.
            if os.path.exists(cache_name):
                os.unlink(cache_name)

            with open(file_name, 'r') as f:
                reader = csv.DictReader(f, delimiter=delimiter)

                # Read data.
                out = {x['output_name']: [] for x in self._fields}
                for row in reader:
                    for field in self._fields:
                        out[field['output_name']].append(
                            field['converter'](row[field['column_name']])
                        )

                # Save data in cache.
                if cache:
                    with open(cache_name, 'wb') as cache_f:
                        pickle.dump(out, cache_f)

        return self._organise_output(out)

    def _organise_output(self, out):
        outs = []

        for group in self._groups:
            # Collect the group's fields.
            group_field_names = []
            for field in self._fields:
                if field['group'] == group['name']:
                    group_field_names.append(field['output_name'])

            # Turn into a matrix or a dictionary.
            if group['matrix']:
                outs.append(np.stack([out[name] for name in
                                      group_field_names], axis=1))
            else:
                outs.append({name: out[name] for name in group_field_names})

        return outs[0] if len(outs) == 1 else outs


@dispatch([object])
def normalise_01(*xs):
    """Normalise objects to [0, 1].

    Args:
        *xs: Objects to normalise.

    Returns:
        list: Normalised objects.
    """
    return [normalise_01(x) for x in xs]


@dispatch(object)
def normalise_01(x):
    def apply(f, x, axes):
        for axis in axes:
            x = f(x, axis=axis, keepdims=True)
        return x

    axes = (0, 2) if B.rank(x) == 3 else (1,)
    min = apply(np.min, x, axes)
    max = apply(np.max, x, axes)
    return (x - min) / (max - min)


@dispatch([object])
def normalise_norm(*xs):
    """Normalise objects to N(0, 1).

    Args:
        *xs: Objects to normalise.

    Returns:
        list: Normalised objects.
    """
    return [normalise_norm(x) for x in xs]


@dispatch(object)
def normalise_norm(x):
    axes = (0, 2) if B.rank(x) == 3 else (1,)
    mean = np.mean(x, axis=axes)
    std = np.std(x, axis=axes)
    return (x - mean) / (std + B.epsilon)


def time_converter(x):
    """Convert a time to number of minutes since the start of the day.

    Args:
        x (str): Time.

    Returns:
        int: Minutes since the start of the day.
    """
    d = parse(x)
    if d.second != 0:
        raise ValueError('Input must not specify nonzero seconds.')
    return 60 * d.hour + d.minute


def float_converter(x):
    """Convert a value to a float.

    Args:
        x (str): Value in string format.

    Returns:
        float: Value.
    """
    x = x.lower()
    if x in ['', ' ', 'n/a']:
        return np.nan
    else:
        return float(x)


def data_path(*xs):
    """Get the path of a data file.

    Args:
        *xs (str): Parts of the path.

    Returns:
        str: Absolute path.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir,
                                        os.pardir,
                                        'data',
                                        *xs))
