import xarray as xr
import pandas as pd
import numpy as np

from .data import resource, data_path

__all__ = ["load"]


def load():
    _fetch()

    return {
        "hf": {
            # Temperature two meters above ground from a Regional Climate Model
            # (RMC):
            "temperature": _load_nc(
                data_path("mf_temp", "T2_monthly.nc"),
                ("Time", "time"),
                ("T2", "temperature"),
                ("lat", "latitude"),
                ("lon", "longitude"),
                expand_coords=False,
            )
        },
        "lf": {  # The below is all re-analysis data.
            # Temperature two meters above ground:
            "temperature": _load_nc(
                data_path("mf_temp", "era5_t2m_monthly.nc"),
                ("time", "time"),
                ("t2m", "temperature"),
                ("latitude", "latitude"),
                ("longitude", "longitude"),
                expand_coords=True,
            ),
            # Northbound wind speed:
            "wind_north": _load_nc(
                data_path("mf_temp", "era5_v10_monthly.nc"),
                ("time", "time"),
                ("v10", "wind_north"),
                ("latitude", "latitude"),
                ("longitude", "longitude"),
                expand_coords=True,
            ),
            # Eastbound wind speed:
            "wind_east": _load_nc(
                data_path("mf_temp", "era5_u10_monthly.nc"),
                ("time", "time"),
                ("u10", "wind_east"),
                ("latitude", "latitude"),
                ("longitude", "longitude"),
                expand_coords=True,
            ),
        },
    }


def _load_nc(path, x, y, coord1, coord2, expand_coords=True):
    data = xr.open_dataset(path)

    # Load the variables.
    df = pd.DataFrame(
        _get_reshape(data, y[0]), index=pd.Index(_get_reshape(data, x[0]), name=x[1])
    )

    # Load the coordinates. These can be stored compressedly or expandedly.
    if expand_coords:
        coord1_values = getattr(data, coord1[0]).values
        coord2_values = getattr(data, coord2[0]).values
        # Expand the coordinates through broadcasting.
        coord1_values, coord2_values = np.broadcast_arrays(
            coord1_values[:, None], coord2_values[None, :]
        )
        coords = pd.DataFrame(
            {coord1[1]: coord1_values.reshape(-1), coord2[1]: coord2_values.reshape(-1)}
        )
    else:
        # No expansion is needed.
        coords = pd.DataFrame(
            {
                coord1[1]: _get_reshape(data, coord1[0]),
                coord2[1]: _get_reshape(data, coord2[0]),
            }
        )

    return coords, df


def _get_reshape(data, name):
    values = getattr(data, name).values
    return values.reshape(*values.shape[0:-2], -1)


def _fetch():
    for remote, local in [
        ("HiFid/T2_monthly.nc", "T2_monthly.nc"),
        ("LoFid/era5_t2m_monthly.nc", "era5_t2m_monthly.nc"),
        ("LoFid/era5_v10_monthly.nc", "era5_v10_monthly.nc"),
        ("LoFid/era5_u10_monthly.nc", "era5_u10_monthly.nc"),
    ]:
        resource(
            target=data_path("mf_temp", local),
            url=f"http://gws-access.ceda.ac.uk/public/bas_climate/files/"
            f"scott/wip/multifidelity_modelling/Peru_Simulations/"
            f"monthly_data/{remote}",
        )
