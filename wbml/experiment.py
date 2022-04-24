import datetime
import os
import pickle
import shutil
import subprocess
import sys
import time

import lab as B
import slugify

from .out import out, kv, streams

__all__ = ["generate_root", "WorkingDirectory"]


def generate_root(name):
    """Generate a root path.

    Args:
        name (str): Name of the experiment.

    Returns:
        str: Root path.
    """
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join("_experiments", f"{now}_{slugify.slugify(name)}")


class Logger:
    """A logger to a file. Open the file every time a message is written.

    Args:
        path (str): Path to file.
    """

    def __init__(self, path):
        self.path = path

        # Empty file.
        with open(self.path, "w"):
            pass

    def write(self, message):
        """Write to the stream.

        Args:
            message (str): Message to write.
        """
        with open(self.path, "a") as f:
            f.write(message)

    def flush(self):
        """Flush the stream."""
        # Nothing to do.


class MultiStream:
    """Combine multiple streams into one.

    Args:
        *streams (stream): Streams.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        """Write to the stream.

        Args:
            message (str): Message to write.
        """
        for stream in self.streams:
            stream.write(message)

    def flush(self):
        """Flush the stream."""
        for stream in self.streams:
            stream.flush()


_default_log = object()
_default_diff = object()


class WorkingDirectory:
    """Working directory.

    Args:
        *root (str): Root of working directory. Use different arguments for
            directories.
        observe (bool, optional): Observe the results. For example, do not overwrite
            the existing log and copied script. Defaults to `False.`
        override (bool, optional): Delete working directory if it already exists.
            Defaults to `False`.
        log (str or None, optional): Log. Set to `None` to disable. Defaults to
            `log.txt`.
        diff (str or None, optional): Record the output of `git diff`. Set to `None`
            to disable. Defaults to `diff.txt`.
        seed (int, optional): Value of random seed. Defaults to `0`.
    """

    def __init__(
        self,
        *root,
        observe=False,
        override=False,
        log=_default_log,
        diff=_default_diff,
        seed=0,
    ):
        self.root = os.path.join(*root)

        # Delete if the root already exists.
        if os.path.exists(self.root) and override:
            out("Experiment directory already exists. Overwriting.")
            shutil.rmtree(self.root)

        # Create root directory.
        os.makedirs(self.root, exist_ok=True)

        # Set default log.
        if observe:
            if log is _default_log:
                log = None
            if diff is _default_diff:
                diff = None
        else:
            if log is _default_log:
                log = "log.txt"
            if diff is _default_diff:
                diff = "diff.txt"

        # Initialise loggers.
        if log is not None:
            # The standard log tracks standard out and standard error.
            logger = Logger(self.file(log))
            sys.stdout = MultiStream(sys.stdout, logger)
            sys.stderr = MultiStream(sys.stderr, logger)

            # Also setup a log which only tracks the output of :mod:`out`.
            base, ext = os.path.splitext(log)
            log_out = base + "_out" + ext
            streams.append(Logger(self.file(log_out)))

        kv("Root", self.root)
        kv("Call", " ".join(sys.argv))
        kv("Now", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        kv("Python", sys.version)

        try:
            # This command will fail if we are not in a git repo.
            subprocess.check_output(
                ["git", "status"],
                # Hide the output if it fails.
                stderr=subprocess.DEVNULL,
            )

            sha = subprocess.check_output(["git", "rev-parse", "HEAD"])
            sha = sha.decode("ascii").strip()

            # Print details about git.
            kv("Commit", sha)

            # Write the output of `git diff`.
            if diff:
                git_diff = subprocess.check_output(["git", "diff"])
                with open(self.file(diff), "w") as f:
                    f.write(git_diff.decode("utf8"))

        except subprocess.CalledProcessError:
            pass

        # Copy calling script.
        if not observe:
            path = os.path.abspath(sys.argv[0])
            if os.path.exists(path):
                shutil.copy(path, self.file("script.py"))
            else:
                out("Could not save calling script.")

        # Sed and print random seed.
        B.set_random_seed(seed)
        kv("Seed", seed)

    def file(self, *name, exists=False):
        """Get the path of a file.

        Args:
            *name (str): Path to file, relative to the root directory. Use
                different arguments for directories.
            exists (bool): Assert that the file already exists. Defaults to
                `False`.

        Returns:
            str: Path to file.
        """
        path = os.path.join(self.root, *name)

        # Ensure that path exists.
        if exists and not os.path.exists(path):
            raise AssertionError(f'File "{path}" does not exist.')
        elif not exists:
            path_dir = os.path.join(self.root, *name[:-1])
            os.makedirs(path_dir, exist_ok=True)

        return path

    def save(self, obj, *args, **kw_args):
        """Save an object to a file.

        Further takes in arguments and keyword arguments from
        :meth:`.experiment.WorkingDirectory.file`.

        Args:
            obj (object): Object to save.
        """
        with open(self.file(*args, **kw_args), "wb") as f:
            pickle.dump(obj, f)

    def load(self, *args, **kw_args):
        """Load an object from a file.

        Further takes in arguments and keyword arguments from
        :meth:`.experiment.WorkingDirectory.file`.
        """
        with open(self.file(*args, **kw_args), "rb") as f:
            return pickle.load(f)
