# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os

import netCDF4 as nc

from .data import data_path

__all__ = ['load']


def load():
    # Load observations.
    obs = nc.Dataset(data_path('stratis', 'erai_T2_1979-2016_daily.nc'))['T2'][:, :]

    # Load simulators.
    sims = os.listdir(data_path('stratis'))
    sims = [f for f in sims
            if os.path.splitext(f)[1].lower() == '.nc'
            if f != 'erai_T2_1979-2016_daily.nc']

    return obs, [nc.Dataset(data_path('stratis', f))['tas'][:, :, :]
                 for f in sims]
