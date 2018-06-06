# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import csv
import os
import pickle

import numpy as np
from lab.tf import B
from plum import Dispatcher, Referentiable, Self
from sklearn.decomposition import PCA

__all__ = ['Data', 'CSVReader', 'normalise_norm', 'normalise_01']

dispatch = Dispatcher()


@dispatch(object)
def ensure_numeric(x):
    return np.array(x)


@dispatch(B.Numeric)
def ensure_numeric(x):
    return x


class Data(Referentiable):
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, x, y):
        self.x = ensure_numeric(x)
        self.y = ensure_numeric(y)

        # Ensure that `y` is a two-dimensional object.
        if B.rank(y) == 1:
            self.y = ensure_numeric(self.y)[:, None]

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
        inds = np.random.permutation(self.n)
        return Data(self.x[inds], self.y[inds])

    @dispatch(float)
    def split(self, f):
        return self.split(int(np.round(self.n * f)))

    @dispatch(int)
    def split(self, i):
        return Data(self.x[:i], self.y[:i]), \
               Data(self.x[i:], self.y[i:])

    def truncate(self, x):
        return self.split(x)[0]

    def normalise(self):
        mean_x = np.mean(self.x, axis=0)[None, :]
        std_x = np.std(self.x, axis=0)[None, :]
        return Data((self.x - mean_x) / std_x, self.y)

    def pca(self, n_components, get_transform=False):
        pca = PCA(n_components=n_components)
        pca.fit(self.x)

        def transform(data):
            return Data(pca.transform(data.x), data.y)

        if get_transform:
            return transform(self), transform
        else:
            return transform(self)

    def cast(self, dtype):
        return Data(B.cast(self.x, dtype), B.cast(self.y, dtype))


class CSVReader:
    def __init__(self):
        self._fields = []
        self._groups = []
        self._current_group = 0

    def set_field_group(self, name, matrix=True):
        if name not in [x['name'] for x in self._groups]:
            self._groups.append({'name': name,
                                 'matrix': matrix})
        self._current_group = name

    def add_field(self, column_name, type, output_name=None):
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
    return [normalise_norm(x) for x in xs]


@dispatch(object)
def normalise_norm(x):
    axes = (0, 2) if B.rank(x) == 3 else (1,)
    mean = np.mean(x, axis=axes)
    std = np.std(x, axis=axes)
    return (x - mean) / (std + B.epsilon)
