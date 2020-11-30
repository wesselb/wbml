import os
import re
from collections import OrderedDict

import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .data import data_path, asserted_dependency

__all__ = ["load"]


def load():
    _fetch()

    # Load observations.
    data = nc.Dataset(data_path("cmip5", "erai_T2_1979-2016_daily.nc"))
    matrix = data["T2"][:].reshape(data["T2"].shape[0], -1)
    # Convert index to `datetime`s. Not sure what the time values represent...
    times = [
        datetime(year=1979, month=1, day=1) + i * timedelta(days=1)
        for i in range(len(data["time"][:]))
    ]
    obs = pd.DataFrame(matrix, index=pd.Index(times, name="time"))

    # Load locations.
    lat = data["latitude"][:]
    lon = data["longitude"][:]
    lat, lon = np.broadcast_arrays(lat[:, None], lon[None, :])
    assert lat.shape == data["T2"].shape[1:3]
    loc = pd.DataFrame({"latitude": lat.flatten(), "longitude": lon.flatten()})

    # Find simulator files.
    sim_files = os.listdir(data_path("cmip5"))
    sim_files = [
        f
        for f in sim_files
        if os.path.splitext(f)[1].lower() == ".nc"
        if f != "erai_T2_1979-2016_daily.nc"
    ]

    # Load sims.
    sims = OrderedDict()
    for sim_file in sim_files:
        data = nc.Dataset(data_path("cmip5", sim_file))
        matrix = data["tas"][:].reshape(data["tas"].shape[0], -1)
        sim_name = re.match(
            r"^cmip5_tas_amip_(.*)_r1i1p1_1979-2008.nc$", sim_file
        ).group(1)
        sims[sim_name] = pd.DataFrame(
            matrix, index=pd.Index(data["time"][:], name="time")
        )

    return loc, obs, sims


def _fetch():
    files = [
        "cmip5_tas_amip_ACCESS1-0_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_ACCESS1-3_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_BNU-ESM_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_CCSM4_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_CMCC-CM_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_CNRM-CM5_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_CSIRO-Mk3-6-0_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_CanAM4_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_EC-EARTH_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_FGOALS-g2_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_FGOALS-s2_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_GFDL-CM3_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_GFDL-HIRAM-C180_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_GFDL-HIRAM-C360_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_HadGEM2-A_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_IPSL-CM5A-LR_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_IPSL-CM5A-MR_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_IPSL-CM5B-LR_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_MIROC5_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_MPI-ESM-LR_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_MPI-ESM-MR_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_MRI-AGCM3-2H_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_MRI-AGCM3-2S_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_MRI-CGCM3_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_NorESM1-M_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_bcc-csm1-1-m_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_bcc-csm1-1_r1i1p1_1979-2008.nc",
        "cmip5_tas_amip_inmcm4_r1i1p1_1979-2008.nc",
        "erai_T2_1979-2016_daily.nc",
    ]
    for file in files:
        asserted_dependency(target=data_path("cmip5", file))
