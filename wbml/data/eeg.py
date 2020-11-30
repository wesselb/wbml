import glob
import multiprocessing
import os
import pickle
from itertools import product

import numpy as np
import pandas as pd
import wbml.out

from .data import data_path, split_df, resource, dependency

__all__ = ["load"]

cache_data = data_path("eeg", "data.pickle")
cache_experiment = data_path("eeg", "experiment.pickle")
cache_experiment_extended = data_path("eeg", "experiment_extended.pickle")


def load(extended=False):
    """Load the EEG data.

    Args:
        extended (bool, optional): Return the train test splits of the first
            train of all patients in the training data.

    Returns:
        tuple[:class:`pd.DataFrame`] or dict[int, :class:`pd.DataFrame`]:
            All data, training data, and test data of the first trail of one
            patient or all patients.
    """
    _fetch()

    # Generate cache if it does not exist.
    if not os.path.exists(cache_experiment) or not os.path.exists(
        cache_experiment_extended
    ):
        _parse()

    # Determine which cache to load.
    path = cache_experiment_extended if extended else cache_experiment

    # Return cached data.
    with open(path, "rb") as f:
        return pickle.load(f)


def _fetch():
    # Training data:
    resource(
        target=data_path("eeg", "SMNI_CMI_TRAIN.tar.gz"),
        url="https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/eeg-mld/"
        "SMNI_CMI_TRAIN.tar.gz",
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

    # Test data:
    resource(
        target=data_path("eeg", "SMNI_CMI_TEST.tar.gz"),
        url="https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/eeg-mld/"
        "SMNI_CMI_TEST.tar.gz",
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


def _parse_trial(fp):
    rows = np.genfromtxt(fp, delimiter=" ", dtype=str)
    sites = {}

    # Extract site names.
    site_names = tuple(sorted(set([x[1] for x in rows])))

    # Extract sites.
    for row in rows:
        try:
            sites[row[1]].append((row[2], row[3]))
        except KeyError:
            sites[row[1]] = [(row[2], row[3])]

    # Convert to data series, assuming that all inputs are the same for all
    # sites.
    x, ys = None, []
    for site in sorted(sites.keys()):
        rows = np.array(sites[site], dtype=float)
        x = rows[:, 0] / 256.0  # Sampled at 256 Hz.
        ys.append(rows[:, 1])

    return {
        "df": pd.DataFrame(
            np.stack(ys, axis=0).T, index=pd.Index(x, name="time"), columns=site_names
        )
    }


def _extract_trials(fps):
    # Extract trial numbers.
    trial_numbers = map(lambda x: int(x.split(".rd.")[1]), fps)

    # Parse trials.
    with multiprocessing.Pool(processes=8) as pool:
        parsed_trials = pool.map(_parse_trial, fps)

    # Check that all lists of site names are the same.
    if len(set([tuple(d["df"].columns) for d in parsed_trials])) != 1:
        raise AssertionError("Site names are inconsistent between trials.")

    # Return dictionary mapping trial number to parsed result.
    return {k: v for k, v in zip(trial_numbers, parsed_trials)}


def _parse():
    wbml.out.out("Parsing EEG data. This may take a while.")

    numbers = [("c", n) for n in [337, 338, 339, 340, 341, 342, 344, 345, 346, 347]] + [
        ("a", n) for n in [364, 365, 368, 369, 370, 371, 372, 375, 377, 378]
    ]
    partitions = ["train", "test"]
    subject_dir_format = data_path("eeg", "{partition}", "co2{type}{subject_n:07d}")

    # Create containers for the different partitions.
    data = {partition: {} for partition in partitions}

    # Iterate over all subjects.
    for partition, (subject_type, subject_n) in product(partitions, numbers):

        # Create an entry for the current subject.
        if subject_n not in data[partition]:
            data[partition][subject_n] = {"type": subject_type}

        # Determine directory of subject.
        subject_dir = subject_dir_format.format(
            partition=partition, type=subject_type, subject_n=subject_n
        )

        # Get all trials files and extract data.
        trial_files = glob.glob(subject_dir + "/*.rd.*")
        data[partition][subject_n]["trials"] = _extract_trials(trial_files)

        # Tag data series with subject number and subject type.
        for d in data[partition][subject_n]["trials"].values():
            d["label"] = (subject_n, subject_type)

    # Checks that all lists of site names are consistent.
    if (
        len(
            set(
                [
                    tuple(d["df"].columns)
                    for _, n in numbers
                    for p in partitions
                    for d in data[p][n]["trials"].values()
                ]
            )
        )
        != 1
    ):
        raise AssertionError("Site names are inconsistent between subjects.")

    # Dump extracted data to file.
    with open(cache_data, "wb") as f:
        pickle.dump(data, f)

    splits = {}

    for _, n in numbers:
        # Just use the first trial of all patients.
        trial_numbers = data["train"][n]["trials"].keys()
        trial = data["train"][n]["trials"][min(trial_numbers)]

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
