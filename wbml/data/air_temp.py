# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import warnings

import numpy as np

from .data import CSVReader, Data, float_converter, time_converter, data_path

__all__ = ['load']


def load():
    sites = ['Bra', 'Cam', 'Chi', 'Sot']
    features = [('DEPTH', 'height'), ('WSPD', 'speed'), ('ATMP', 'temp')]

    days = list(range(1, 1 + 31))
    fn = '{site:s}{day:02d}Jul2013.csv'
    r = CSVReader()
    r.set_field_group('input')
    r.add_field('Time', time_converter, 'time')
    r.set_field_group('output')
    for name, output_name in features:
        r.add_field(name, float_converter, output_name)

    x_all, ys_all = None, []
    for site in sites:
        xs_site, ys_site = [], []
        for day in days:
            mins, y_incomplete = \
                r.read(data_path('air_temp', fn.format(site=site, day=day)))

            # Fill in missing data.
            x = day + np.arange(0, 24 * 60, 5)[:, None] / float(24 * 60)
            y = np.zeros((12 * 24, 3))
            y.fill(np.nan)
            # Input is at multiples of five minutes.
            y[(mins[:, 0] / 5).astype(int), :] = y_incomplete

            xs_site.append(x)
            ys_site.append(y)

        x_all = np.concatenate(xs_site, axis=0)
        ys_all.append(np.concatenate(ys_site, axis=0))

    x = x_all
    y = np.concatenate(ys_all, axis=1)

    # Clear invalid sensor readings.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y[y < -1000] = np.nan

    # Construct data and reorder outputs.
    d = Data(x, y,
             ['{}/{}'.format(site, feature)
              for site in sites
              for _, feature in features])
    d = d.select('Bra/height', 'Sot/height', 'Cam/height', 'Chi/height',
                 'Bra/speed', 'Sot/speed', 'Cam/speed', 'Chi/speed',
                 'Bra/temp', 'Sot/temp', 'Cam/temp', 'Chi/temp')[0]

    # Construct data sets of various sizes.
    ds = []
    for all_interval in [(10, 20), (8, 23), None]:
        d_all = d if all_interval is None else d.interval(*all_interval)[0]

        # Select test sets.
        test1, d_train = d_all.interval_y('Cam/temp', 12, 16)
        test2, d_train = d_train.interval_y('Chi/temp', 14, 18)

        # Surround the test sets a bit for nice predictions.
        test1_ext = d.interval_y('Cam/temp', 11.5, 16.5)[0]
        test2_ext = d.interval_y('Chi/temp', 13.5, 18.5)[0]

        ds.append((d_all, d_train, [test1, test2, test1_ext, test2_ext]))

    return ds
