# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import reduce
from operator import mul
from itertools import product

import numpy as np
from lab import B
from plum import Dispatcher, Self, Referentiable

__all__ = ['Packer', 'Vars', 'vars32', 'vars64', 'VarsFrom', 'inv_perm',
           'identity', 'map_cols', 'Initialiser', 'construct_display_resolver']

_dispatch = Dispatcher()


class Packer(object):
    """Pack objects into a vector.

    Args:
        *objs (tensor): Objects to pack.
    """

    def __init__(self, *objs):
        self._shapes = [B.shape(obj) for obj in objs]
        self._lengths = [B.length(obj) for obj in objs]

    def pack(self, *objs):
        """Pack objects.

        Args:
            *objs (tensor): Objects to pack.

        Returns:
            tensor: Vector representation of the objects.
        """
        return B.concat([B.reshape(obj, [-1]) for obj in objs], axis=0)

    def unpack(self, package):
        """Unpack vector.

        Args:
            package (tensor): Vector to unpack.

        Returns:
            tuple[tensor]: Original objects.
        """
        i, outs = 0, []
        for shape, length in zip(self._shapes, self._lengths):
            outs.append(B.reshape(package[i:i + length], shape))
            i += length
        return outs


class Initialiser(object):
    """Variable initialiser."""

    def __init__(self):
        self._assignments = {}

    def assign(self, name, values):
        """Assign values to a particular variable.

        Args:
            name (str): Name of variables.
            values (list[tensor]): List of values to assign.
        """
        self._assignments[name] = values

    def generate(self, vs):
        """Generate initialisers.

        Args:
            vs (:class:`.util.Vars`): Variable storage.

        Returns:
            list[list[tensor]]: List of initialisers. Run a initialiser with a
                TensorFlow session to set the initialisation.
        """
        names, values = zip(*self._assignments.items())
        return [[vs.assign(name, val) for name, val in zip(names, value)]
                for value in product(*values)]


class Vars(Referentiable):
    """Variable storage manager.

    Args:
        dtype (data type, optional): Data type of the variables. Defaults to
            `np.float32`.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, dtype=np.float32):
        self.vars = []
        self.dtype = dtype
        self.generators = {}
        self.assigners = {}
        self.lookup = {}

    def init(self, session):
        """Initialise the variables.

        Args:
            session (:class:`B.Session`): TensorFlow session.
        """
        session.run(B.variables_initializer(self.vars))

    def get_vars(self, *names):
        """Get variables by name.

        Args:
            *names (hashable): Names of the variables.

        Returns:
            list[tensor]: Variables specified by `names`. If no names are
                specified, all variables are returned.
        """
        if len(names) == 0:
            return self.vars
        else:
            return [self.lookup[name] for name in names]

    def get(self, init=None, shape=(), dtype=None, name=None):
        """Get an unbounded variable.

        Args:
            init (tensor, optional): Initialisation of the variable.
            shape (tuple[int], optional): Shape of the variable. Defaults to
                scalar.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (hashable, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """

        def generate_init(shape, dtype):
            return B.randn(shape, dtype=dtype)

        return self._get_var(transform=lambda x: x,
                             inverse_transform=lambda x: x,
                             init=init,
                             generate_init=generate_init,
                             shape=shape,
                             dtype=dtype,
                             name=name)

    def positive(self, init=None, shape=(), dtype=None, name=None):
        """Get a positive variable.

        Args:
            init (tensor, optional): Initialisation of the variable.
            shape (tuple[int], optional): Shape of the variable. Defaults to
                scalar.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (hashable, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """

        def generate_init(shape, dtype):
            return B.rand(shape, dtype=dtype)

        return self._get_var(transform=lambda x: B.log(B.exp(x) + 1) + 1e-3,
                             inverse_transform=lambda y: B.log(B.exp(y - 1e-3) - 1),
                             init=init,
                             generate_init=generate_init,
                             shape=shape,
                             dtype=dtype,
                             name=name)

    def pos(self, *args, **kw_args):
        """Alias for :meth:`.util.Vars.positive`."""
        return self.positive(*args, **kw_args)

    def bounded(self,
                init=None,
                lower=1e-4,
                upper=1e4,
                shape=(),
                dtype=None,
                name=None):
        """Get a bounded variable.

        Args:
            init (tensor, optional): Initialisation of the variable.
            lower (tensor, optional): Lower bound. Defaults to `1e-4`.
            upper (tensor, optional): Upper bound. Defaults to `1e4`.
            shape (tuple[int], optional): Shape of the variable. Defaults to
                scalar.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (hashable, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """

        def transform(x):
            return lower + (upper - lower) / (1 + B.exp(x))

        def inverse_transform(x):
            return B.log(upper - x) - B.log(x - lower)

        def generate_init(shape, dtype):
            return lower + B.rand(shape, dtype=dtype) * (upper - lower)

        return self._get_var(transform=transform,
                             inverse_transform=inverse_transform,
                             init=init,
                             generate_init=generate_init,
                             shape=shape,
                             dtype=dtype,
                             name=name)

    def bnd(self, *args, **kw_args):
        """Alias for :meth:`.util.Vars.bounded`."""
        return self.bounded(*args, **kw_args)

    def _get_var(self,
                 transform,
                 inverse_transform,
                 init,
                 generate_init,
                 shape,
                 dtype,
                 name):
        # If the name already exists, return that variable.
        try:
            return self[name]
        except KeyError:
            pass

        # Resolve data type.
        dtype = self.dtype if dtype is None else dtype

        # Resolve initialisation and inverse transform.
        if init is None:
            init = generate_init(shape=shape, dtype=dtype)
        else:
            init = B.array(init, dtype=dtype)

        # Construct latent variable and store if a name is given.
        latent = B.Variable(inverse_transform(init))
        self.vars.append(latent)
        if name:
            self.lookup[name] = latent

        # Construct generator and assigner.
        def generate():
            return transform(latent)

        def assign(value):
            return B.assign(latent, inverse_transform(value))

        # Store if a name is given.
        if name:
            self.generators[name] = generate
            self.assigners[name] = assign

        # Generate the variable and return.
        return generate()

    def assign(self, name, value):
        """Assign a value to a variable.

        Args:
            name (hashable): Name of variable to assign value to.
            value (tensor): Value to assign.

        Returns:
            tensor: TensorFlow tensor that can be run to perform the assignment.
        """
        return self.assigners[name](value)

    def __getitem__(self, name):
        """Get a variable by name.

        Args:
            name (hashable): Name of variable.

        Returns:
            tensor: Variable.
        """
        return self.generators[name]()


class VarsFrom(object):
    def __init__(self, source):
        self._source = source
        self._i = 0

    def get(self, shape):
        length = reduce(mul, shape, 1)
        out = B.reshape(self._source[self._i: self._i + length], shape)
        self._i += length
        return out


vars32 = Vars(np.float32)
vars64 = Vars(np.float64)


def inv_perm(perm):
    """Invert a permutation.

    Args:
        perm (list): Permutation to invert.

    Returns:
        list: Inverse permutation.
    """
    out = [0 for _ in range(len(perm))]
    for i, p in enumerate(perm):
        out[p] = i
    return out


def identity(x):
    """Identity function.

    Args:
        x (object): Input.

    Returns:
        object: `x`.
    """
    return x


def map_cols(f, xs):
    """Map the columns of a matrix on a function.

    Args:
        f (function): Function to map.
        xs (tensor): Matrix onto which to map the function.

    Returns:
        tensor: Result.
    """
    return B.map_fn(lambda x: f(x[:, None]), B.transpose(xs))


def construct_display_resolver(sess):
    """Construct a resolver displaying variables.

    Args:
        sess: TensorFlow session

    Returns:
        function: Resolver.
    """
    dispatch = Dispatcher()

    @dispatch(B.TF)
    def resolve(x):
        return resolve(sess.run(x))

    @dispatch(object)
    def resolve(x):
        x = np.squeeze(x)
        if B.rank(x) == 0:
            return '{:.2e}'.format(x)
        elif B.rank(x) == 1:
            return '[{}]'.format(', '.join(['{:.2e}'.format(y) for y in x]))
        else:
            raise RuntimeError('Cannot nicely display a rank-{} tensor.'
                               ''.format(B.rank(x)))

    return resolve
