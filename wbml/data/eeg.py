import glob
import multiprocessing
import os
import pickle
import warnings

import numpy as np
import pandas as pd

import wbml.out
from .data import data_path, split_df, resource, dependency

__all__ = ["load", "load_full"]

cache_experiment = data_path("eeg", "experiment.pickle")
cache_experiment_extended = data_path("eeg", "experiment_extended.pickle")
cache_full = data_path("eeg", "full.pickle")


def _load(cache, parse):
    if not os.path.exists(cache):
        parse()
    with open(cache, "rb") as f:
        return pickle.load(f)


def load(extended=False):
    """Load the EEG data, which is a particular split of the first trial of subject
    337 in the large training data.

    Args:
        extended (bool, optional): Return the train test splits of the first
            train of all patients in the large training data. Defaults to `False`.

    Returns:
        tuple[:class:`pd.DataFrame`] or dict[int, :class:`pd.DataFrame`]:
            All data, training data, and test data of the first trail of one
            patient or all patients.
    """

    _fetch_large()
    if extended:
        return _load(cache_experiment_extended, _parse_experiment)
    else:
        return _load(cache_experiment, _parse_experiment)


def load_full():
    """Load the full EEG data.

    Returns:
        dict: Full data.
    """
    _fetch_full()
    return _load(cache_full, _parse_full)


def _fetch_large():
    """Fetch the large training and test data set."""
    # Large training data:
    resource(
        target=data_path("eeg", "SMNI_CMI_TRAIN.tar.gz"),
        url="https://kdd.ics.uci.edu/databases/eeg/SMNI_CMI_TRAIN.tar.gz",
    )
    dependency(
        target=data_path("eeg", "train"),
        source=data_path("eeg", "SMNI_CMI_TRAIN.tar.gz"),
        commands=[
            "tar xzf SMNI_CMI_TRAIN.tar.gz",
            "mv SMNI_CMI_TRAIN train",
            "find train | grep gz$ | xargs gunzip",
        ],
    )
    # Large test data:
    resource(
        target=data_path("eeg", "SMNI_CMI_TEST.tar.gz"),
        url="https://kdd.ics.uci.edu/databases/eeg/SMNI_CMI_TEST.tar.gz",
    )
    dependency(
        target=data_path("eeg", "test"),
        source=data_path("eeg", "SMNI_CMI_TEST.tar.gz"),
        commands=[
            "tar xzf SMNI_CMI_TEST.tar.gz",
            "mv SMNI_CMI_TEST test",
            "find test | grep gz$ | xargs gunzip",
        ],
    )


def _fetch_full():
    """Fetch the full data set."""
    resource(
        target=data_path("eeg", "eeg_full.tar"),
        url="https://kdd.ics.uci.edu/databases/eeg/eeg_full.tar",
    )
    dependency(
        target=data_path("eeg", "full"),
        source=data_path("eeg", "eeg_full.tar"),
        commands=[
            "mkdir full",
            "tar xf eeg_full.tar -C full",
            "ls full | grep gz$ | xargs -I {} tar xzf full/{} -C full",
            "ls full | grep gz$ | xargs -I {} rm full/{}",
            "find full | grep gz$ | xargs gunzip",
        ],
    )


def _parse_trial(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # `np.genfromtxt` produces a warning if the file is empty.
        sites = np.genfromtxt(path, usecols=1, dtype=str)
        data = np.genfromtxt(path, usecols=(2, 3))

    # The file can be empty. In that case, just return nothing.
    if len(sites) == 0:
        return None

    # File is not empty. Start parsing the data.
    x, y = data[:, 0], data[:, 1]

    # Assume that all sites have the same inputs. If that is not true, `np.stack` will
    # fail.
    inds = np.where(np.diff(x) < 0)[0] + 1
    sites = [sites[0]] + [sites[i] for i in inds]
    x = x[: inds[0]] / 256.0  # Sampled at 256 Hz.
    y = np.stack(np.split(y, inds), axis=1)

    return {"df": pd.DataFrame(y, index=pd.Index(x, name="time"), columns=sites)}


def _extract_trials(subject_path):
    paths = glob.glob(f"{subject_path}/*.rd.*")

    # Extract trial numbers.
    trial_numbers = map(lambda x: int(x.split(".rd.")[1]), paths)

    # Parse trials. Use half of all available CPUs.
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as pool:
        parsed_trials = pool.map(_parse_trial, paths)

    # Return dictionary mapping trial number to parsed result. Make sure to ignore empty
    # files.
    return {k: v for k, v in zip(trial_numbers, parsed_trials) if v is not None}


def _extract_subjects(path):
    paths = glob.glob(path + "/co*")

    # Safe determine the subject types.
    type_map = {"co2a": "2a", "co2c": "2c", "co3a": "3a", "co3c": "3c"}

    # Extract all subjects.
    subjects = {}
    for path in paths:
        subject = {
            "type": type_map[os.path.split(path)[-1][:4]],
            "trials": _extract_trials(path),
        }
        number = int(os.path.split(path)[-1][-7:])
        subjects[number] = subject

    return subjects


def _parse_experiment():
    wbml.out.out("Parsing EEG data. This should be fairly quick.")

    data = _extract_subjects(data_path("eeg", "train"))

    splits = {}

    for n in data.keys():
        # Just use the first trial of all patients.
        trial_numbers = data[n]["trials"].keys()
        trial = data[n]["trials"][min(trial_numbers)]

        # Select a number of fixed labels.
        labels = ["F3", "F4", "F5", "F6"] + ["FZ", "F1", "F2"]
        trial = trial["df"][labels]

        # Split according to paper.
        test, train = split_df(
            trial,
            index_range=(trial.shape[0] - 100, trial.shape[0]),
            columns=["FZ", "F1", "F2"],
            iloc=True,
        )

        splits[n] = (trial, train, test)

    # Save experiment data.
    with open(cache_experiment, "wb") as f:
        pickle.dump(splits[337], f)

    # Save extended experiment data.
    with open(cache_experiment_extended, "wb") as f:
        pickle.dump(splits, f)


def _parse_full():
    wbml.out.out("Parsing full EEG data. This may take a while.")

    data = _extract_subjects(data_path("eeg", "full"))

    with open(cache_full, "wb") as f:
        pickle.dump(data, f)
