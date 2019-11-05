import logging
import sys
import time
from collections import defaultdict

import lab as B
import numpy as np
from plum import Dispatcher, Referentiable, Self

__all__ = ['Section', 'out', 'kv', 'format', 'Counter', 'Progress']

log = logging.getLogger(__name__)

_dispatch = Dispatcher()

indent_level = 0  #: Indentation level.
indent_width = 4  #: Number of indentation characters per level.
indent_char = ' '  #: Indentation character.
key_width = 10  #: Minimum width for printing keys, for alignment.
streams = [sys.stdout]  #: Output streams.
digits = 3  #: Number of digits to show.


def _print(msg, line_end='\n', indent=True, flush=False):
    """Print a line with proper indentation.

    Args:
        msg (str): Message to print.
        line_end (str, optional): String to print at the end of the line.
            Defaults to "\n".
        indent (bool, optional): Insert indentation. Defaults to `True`.
        flush (bool, optional): Flush after writing. Defaults to `False`.
    """
    global indent_level
    global indent_width
    global indent_char

    if indent:
        # Construct total indentation.
        indent = indent_char * indent_width * indent_level

        # Indent the message and all newlines therein.
        msg = indent + msg.replace('\n', '\n' + indent)

    # Write the message plus line end.
    for stream in streams:
        stream.write(msg + line_end)

        if flush:
            stream.flush()


class Section:
    """Create a section in which output is indented.

    Args:
        name (str): Name of the section.
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        # Print the section name, if one is given.
        if self.name is not None:
            _print('{}:'.format(self.name))

        # Increase indentation.
        global indent_level
        indent_level += 1

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Decrease indentation.
        global indent_level
        indent_level -= 1


def out(msg):
    """Print a message.

    Args:
        msg (str): Message to print.
    """
    _print(msg)


@_dispatch(object, object)
def kv(key, value):
    """Print a key-value pair.

    Args:
        key (str): Key.
        value (object): Value.
    """
    global key_width

    # Construct the format.
    f = '{{key:{width}}} {{value}}'.format(width=key_width + 1)

    # Format the key and value.
    formatted_key = format(key)
    formatted_value = format(value)

    # If the value contains a newline, print it within a section.
    if '\n' in formatted_value:
        with Section(name=formatted_key):
            out(formatted_value)
    else:
        _print(f.format(key=formatted_key + ':', value=formatted_value))


@_dispatch(dict)
def kv(dict_):
    for k, v in dict_.items():
        kv(k, v)


@_dispatch(object, dict)
def kv(key, dict_):
    with Section(key):
        for k, v in dict_.items():
            kv(k, v)


@_dispatch(object)
def format(x):
    """Format an object.

    Args:
        x (object): Object to format.

    Returns:
        str: `x` as a string.
    """
    return str(x)


@_dispatch(B.Number)
def format(x):
    global digits

    # Format number in scientific notation.
    out = '{{:.{}e}}'.format(digits).format(x)

    # If exponent is in [-2, 2], format as float instead.
    if -2 <= int(out.split('e')[1]) <= 2:
        return str(float(out))

    return out


@_dispatch(B.Int)
def format(x):
    return str(x)


@_dispatch(list)
def format(xs):
    return '[{}]'.format(', '.join([format(x) for x in xs]))


@_dispatch(tuple)
def format(xs):
    return '({})'.format(', '.join([format(x) for x in xs]))


@_dispatch(set)
def format(xs):
    return '{{{}}}'.format(', '.join([format(x) for x in xs]))


@_dispatch(B.NPNumeric)
def format(x):
    # A NumPy array can be a scalar.
    if x.shape == ():
        return format.invoke(B.Number)(x)

    # Represent as an array.
    x_str = np.array_str(x, precision=digits)

    # If the array is displayed on multiple lines, also show its shape and
    # data type.
    if '\n' in x_str:
        x_str = '({} array of data type {})\n' \
                ''.format('x'.join([str(d) for d in x.shape]), x.dtype) + x_str

    return x_str


@_dispatch(B.TorchNumeric)
def format(x):
    return format(x.detach().numpy())


@_dispatch(B.TFNumeric)
def format(x):
    return format(x.numpy())


class Counter:
    """A counter.

    Args:
        name (str, optional): Name of the counter. Defaults to "Counting".
        total (int, optional): Total number of counts. Defaults to no total.
    """

    def __init__(self, name='Counting', total=None):
        self.name = name
        self.total = total

    def __enter__(self):
        # Reset the counter.
        self.iteration = 0

        # Print the name.
        title = self.name

        # Print the total, if one is given.
        if self.total:
            title += ' (total: {})'.format(self.total)

        # Perform printing and flush, because there is no newline.
        _print(title + ':', line_end='', flush=True)

        return self

    def count(self):
        """Count once."""
        self.iteration += 1
        _print(' {}'.format(self.iteration),
               line_end='', indent=False, flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _print('\n', line_end='', indent=False, flush=True)

    @staticmethod
    def map(f, xs, name='Mapping'):
        """Perform a mapping operation that is counted.

        Args:
            f (function): Function to apply.
            xs (iterable): Arguments to apply.
            name (str, optional): Name of the mapping operation. Defaults to
                "Mapping".

        Returns:
            list: Result.
        """
        # Perform mapping operation.
        results = []
        with Counter(name=name, total=len(xs)) as counter:
            for x in xs:
                counter.count()
                results.append(f(x))
        return results


def _compute_alpha(cutoff_lag):
    """Compute the coefficient `a` of the one-pole filter
    `y[n] = a x[n] + (1 - a) x[n - 1]`.

    Args:
        cutoff_lag (int): Cut-off frequency, in number of lags.

    Returns:
        float: Coefficient `a`.
    """
    if cutoff_lag is None:
        return 1
    else:
        a = np.cos(2 * np.pi / cutoff_lag)
        return a - 1 + np.sqrt(a ** 2 - 4 * a + 3)


class Progress(metaclass=Referentiable):
    """Display progress.

    Args:
        name (str): Name of operation. Defaults to "Progress".
        total (int): Total number of iterations. Defaults to no total.
        filter (dict): A dictionary mapping names of recorded values to the
            cut-off frequency of a one-pole filter, in number of lags.
        interval (int or float): Interval at which to print a report in
            either number of seconds (float) or number of iterations (int).
            Defaults to one second.
        filter_global (int): Cut-off frequency of a one-pole filter of the
            default smoother, in number of lags.

    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self,
                 name='Progress',
                 total=None,
                 filter=None,
                 interval=1.0,
                 filter_global=5):
        self.name = name
        self.total = total
        self.interval = interval
        self.section = Section(name)

        global_alpha = _compute_alpha(filter_global)
        self.alpha = defaultdict(lambda: global_alpha)
        if filter:
            for name, cutoff in filter.items():
                self.alpha[name] = _compute_alpha(cutoff)

        self.cur_time = None
        self.iteration = None
        self.last_report = None
        self.report_was_just_shown = False
        self.values = dict()

    def __enter__(self):
        self.start_time = time.time()
        self.last_time = time.time()
        self.last_report = -np.inf
        self.report_was_just_shown = False
        self.iteration = 0
        self.values.clear()
        self.section.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.report_was_just_shown:
            self.report()
        out('Done!')
        self.section.__exit__(exc_type, exc_val, exc_tb)

    @_dispatch()
    def __call__(self, **updates):
        return self(updates)

    @_dispatch(dict)
    def __call__(self, updates):
        now = time.time()

        # Increase iteration counter.
        self.iteration += 1

        # Record time per iteration.
        updates['_time_per_it'] = now - self.last_time
        self.last_time = now

        # Update tracked values.
        for name, value in updates.items():

            # Perform smoothing if `value` is numeric.
            if name in self.values and isinstance(value, B.Numeric):
                alpha = self.alpha[name]
                value = alpha * value + (1 - alpha) * self.values[name]

            self.values[name] = value

        # See if a progress report should be displayed.
        if (
                # If `self.interval` is a float, it specifies the number of
                # seconds between reports.
                (isinstance(self.interval, float) and
                 now - self.last_report > self.interval) or

                # If `self.interval` is an int, it specifies the number of
                # iterations between reports.
                (isinstance(self.interval, int) and
                 self.iteration % self.interval == 0) or

                # Always show a report on the first iteration.
                self.iteration == 1
        ):
            self.report()
        else:
            self.report_was_just_shown = False

    def report(self):
        """Show a report."""
        self.report_was_just_shown = True

        # Construct title.
        title = 'Iteration {}'.format(self.iteration)
        if self.total:
            title += '/' + str(self.total)

        with Section(title):
            # Show how much time has elapsed.
            kv('Time elapsed',
               '{:.1f} s'.format(self.last_time - self.start_time))

            if self.total:
                # Estimate time left.
                time_left = max(self.total + 1 - self.iteration, 0) * \
                            self.values['_time_per_it']
                kv('Time left', '{:.1f} s'.format(time_left))

            # Print all updates.
            for name in sorted(self.values.keys()):
                if name == '_time_per_it':
                    continue
                kv(name, self.values[name])

        # Record time of report.
        self.last_report = time.time()

    @staticmethod
    def map(f, xs, name='Mapping'):
        """Perform a mapping operation whose progress is shown.

        Args:
            f (function): Function to apply.
            xs (iterable): Arguments to apply.
            name (str, optional): Name of the mapping operation. Defaults to
                "Mapping".

        Returns:
            list: Result.
        """
        # Perform mapping operation.
        results = []
        with Progress(name=name, total=len(xs)) as progress:
            for x in xs:
                progress()
                results.append(f(x))
        return results
