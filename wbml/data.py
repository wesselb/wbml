# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import csv
import os
import pickle

import numpy as np
from lab import B
from plum import Dispatcher, Referentiable, Self

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
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, x, y):
        x = ensure_numeric(x)
        y = ensure_numeric(y)

        # Ensure that `x` and `y` are at least two-dimensional.
        x = x[:, None] if B.rank(x) == 1 else x
        y = y[:, None] if B.rank(y) == 1 else y

        self.x = x
        self.y = y

    @property
    def n(self):
        """Number of data points."""
        return B.shape(self.x)[0]

    @property
    def m(self):
        """Number of features."""
        return B.shape(self.x)[1]

    def __getitem__(self, item):
        return Data(self.x[item], self.y[item])

    def shuffle(self):
        """Shuffle the data.

        Returns:
            :class:`.data.Data`: Shuffled data.
        """
        inds = np.random.permutation(self.n)
        return Data(self.x[inds], self.y[inds])

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
        return Data(self.x[:i], self.y[:i]), \
               Data(self.x[i:], self.y[i:])

    def truncate(self, x):
        """Truncate the data.

        Args:
            point (int or float): Point to truncate the data at. Can either
                be an int, representing an index, or a float, representing a
                fractional point.

        Returns:
            :class:`.data.Data`: Truncated data.
        """
        return self.split(x)[0]

    def normalise(self, ref=None, unnormaliser=False):
        """Normalise data.

        Args:
            ref (:class:`.data.Data`, optional): Reference to normalise with
                respect to. Defaults to itself.
            unnormaliser (bool, optional): Also return a function to
                unnormalise data. Defaults to `False`.

        Returns:
            :class:`.data.Data`: Normalised data. Can further return a
                function to unnormalise the data.
        """
        ref = self if ref is None else ref
        y_mean = np.nanmean(ref.y, axis=0, keepdims=True)
        y_std = np.nanstd(ref.y, axis=0, keepdims=True)
        data = Data(self.x, (self.y - y_mean) / y_std)

        # Also return function to unnormalise data if asked for.
        if unnormaliser:
            def unnormalise(d):
                return Data(d.x, d.y * y_std + y_mean)

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
        return Data(B.cast(self.x, dtype), B.cast(self.y, dtype))


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
