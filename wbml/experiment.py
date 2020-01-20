import os
import shutil
import time

import slugify

from .out import out, kv, streams

__all__ = ['generate_root', 'WorkingDirectory']


def generate_root(name):
    """Generate a root path.

    Args:
        name (str): Name of the experiment.

    Returns:
        str: Root path.
    """
    now = time.strftime('%Y-%m-%d_%H-%M-%S')
    return os.path.join('_experiments', f'{now}_{slugify.slugify(name)}')


class Logger:
    """A logger to a file that automatically closes the stream if the
    object is deleted.

    Args:
        path (str): Path to file.
    """

    def __init__(self, path):
        self.stream = open(path, 'w')

    def write(self, message):
        """Write to the file.

        Args:
            message (str): Message to write.
        """
        self.stream.write(message)

    def flush(self):
        """Flush the stream."""
        self.stream.flush()

    def __del__(self):
        self.stream.close()


class WorkingDirectory:
    """Working directory.

    Args:
        *root (str): Root of working directory. Use different arguments for
            directories.
        override (bool, optional): Delete working directory if it already
            exists. Defaults to `False`.
    """

    def __init__(self, *root, override=False):
        self.root = os.path.join(*root)

        # Delete if the root already exists.
        if os.path.exists(self.root) and override:
            out('Experiment directory already exists. Overwriting.')
            shutil.rmtree(self.root)

        kv('Root', self.root)

        # Create root directory.
        os.makedirs(self.root, exist_ok=True)

        # Initialise logger.
        streams.append(Logger(self.file('log.txt')))

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
