import numpy as np
import pandas as pd
import scipy.io

from .data import data_path, resource, dependency

__all__ = ["load"]


def load():
    _fetch()

    # Compute angles.
    azimuths = np.concatenate(
        (
            np.array([-80, -65, -55]),
            np.arange(-45, 45 + 1, 5),
            np.array([55, 65, 80]),
        )
    )[:, None]
    elevations = (-45 + 5.625 * np.arange(50))[None, :]
    azimuths, elevations = np.broadcast_arrays(azimuths, elevations)

    # Load data.
    data = scipy.io.loadmat(
        data_path("kemar", "standard_kemar", "subject_021", "hrir_final.mat")
    )
    left = data["hrir_l"]
    right = data["hrir_r"]

    # Compute time.
    t = np.arange(left.shape[2]) / 441000

    # Arrange into convenient data frames.
    angles = pd.MultiIndex.from_frame(
        pd.DataFrame({"azimuth": azimuths.flatten(), "elevation": elevations.flatten()})
    )
    left = pd.DataFrame(left.reshape(len(angles), -1).T, columns=angles, index=t)
    right = pd.DataFrame(right.reshape(len(angles), -1).T, columns=angles, index=t)

    return {"left": left, "right": right}


def _fetch():
    resource(
        target=data_path("kemar", "standard_kemar.tar"),
        url=(
            "https://ucdavis.app.box.com/index.php"
            "?rm=box_download_shared_file"
            "&shared_name=0418n9rcvh2krxg7zy4prbgkam90a56a"
            "&file_id=f_245793603750"
        ),
    )
    dependency(
        target=data_path("kemar", "standard_kemar"),
        source=data_path("kemar", "standard_kemar.tar"),
        commands=[
            "mkdir standard_kemar",
            "tar xzf standard_kemar.tar -C standard_kemar",
        ],
    )
