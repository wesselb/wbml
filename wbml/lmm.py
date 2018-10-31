# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from lab import B
from plum import Dispatcher, Referentiable, Self
from stheno import GP, Delta, Graph, Normal, Obs, dense

__all__ = ['LMMPP', 'OLMM']

log = logging.getLogger(__name__)


class LMMPP(object):
    """PP implementation of the linear mixing model.

    Args:
        kernels (list[:class:`stheno.Kernel`]) Kernels.
        noise_obs (tensor): Observation noise. One.
        noises_latent (tensor): Latent noises. One per latent processes.
        H (tensor): Mixing matrix.
    """

    def __init__(self, kernels, noise_obs, noises_latent, H):
        self.graph = Graph()
        self.p, self.m = B.shape_int(H)

        # Create latent processes.
        xs = [GP(k, graph=self.graph) for k in kernels]

        # Create latent noise.
        es = [GP(noise * Delta(), graph=self.graph)
              for noise in noises_latent]

        # Create noisy latent processes.
        xs_noisy = [x + e for x, e in zip(xs, es)]

        # Multiply with mixing matrix.
        self.fs = [0 for _ in range(self.p)]
        for i in range(self.p):
            for j in range(self.m):
                self.fs[i] += xs_noisy[j] * H[i, j]
        # Create two observed process.
        self.ys = [f + GP(noise_obs * Delta(), graph=self.graph)
                   for f in self.fs]
        self.y = self.graph.cross(*self.ys)

    def observe(self, x, y):
        """Observe data.

        Args:
            x (tensor): Inputs.
            y (tensor): Outputs.
        """
        obs = Obs(*((self.ys[i](x), y[:, i]) for i in range(self.p)))
        self.fs = [p | obs for p in self.fs]
        self.ys = [p | obs for p in self.ys]
        self.y = self.y | obs

    def sample(self, x):
        """Sample data.

        Args:
            x (tensor): Inputs to sample at.
        """
        samples = self.graph.sample(*(y(x) for y in self.ys))
        return B.concat(samples, axis=1)

    def lml(self, x, y):
        """Compute the LML.

        Args:
            x (tensor): Inputs of data.
            y (tensor): Output of data.

        Returns:
            tensor: LML of data.
        """
        ys = list(self.ys)

        # Compute the LML using the product rule.
        lml = 0
        for i in range(self.p):
            # Compute `log p(y_i | y_{1:i - 1})`.
            lml += ys[i](x).logpdf(y[:, i])

            # Condition the remainder on the observation for `y_i`.
            obs = Obs(ys[i](x), y[:, i])
            ys[i:] = [p | obs for p in ys[i:]]

        return lml

    def marginals(self, x):
        """Compute marginals.

        Args:
            x (tensor): Inputs to construct marginals at.

        Returns:
            tuple[tensor]: Marginals per output, spatial means per time, and
                spatial variance per time.
        """
        preds = [y(x).marginals() for y in self.ys]
        means, vars = [], []
        for i in range(B.shape_int(x)[0]):
            d = self.y(x[i])
            means.append(d.mean)
            vars.append(dense(d.var))
        return preds, means, vars


class OLMM(Referentiable):
    """Orthogonal linear mixing model.

    Args:
        kernels (list[:class:`stheno.Kernel`]) Kernels.
        noise_obs (tensor): Observation noise. One.
        noises_latent (tensor): Latent noises. One per latent processes.
        H (tensor): Mixing matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch({list, tuple}, B.Numeric, {B.Numeric, list}, B.Numeric)
    def __init__(self, kernels, noise_obs, noises_latent, H):
        U, S_sqrt, _ = B.svd(H)
        OLMM.__init__(self, kernels, noise_obs, noises_latent, U, S_sqrt)

    @_dispatch({list, tuple},
               B.Numeric,
               {B.Numeric, list},
               B.Numeric,
               B.Numeric)
    def __init__(self, kernels, noise_obs, noises_latent, U, S_sqrt):
        # Determine number of outputs and latent processes.
        self.p, self.m = B.shape(U)

        # Save components of mixing matrix.
        self.U = U
        self.S_sqrt = S_sqrt

        # Save noises.
        self.noise_obs = noise_obs
        self.noises_latent = noises_latent

        # Compute projected noises.
        noises_projected = [noise_obs / self.S_sqrt[i] ** 2 + noises_latent[i]
                            for i in range(self.m)]

        # Construct single-output models.
        self.graphs = [Graph() for _ in range(self.m)]
        self.xs = [GP(k, graph=g) for k, g in zip(kernels, self.graphs)]
        self.xs_noises = [GP(Delta() * n, graph=g)
                          for n, g in zip(noises_projected, self.graphs)]
        self.xs_noisy = [p + n for p, n in zip(self.xs, self.xs_noises)]

        # Construct mixing matrix and projection.
        self._construct_H_and_P()

    def _construct_H_and_P(self):
        # Compute mixing matrix.
        self.H = self.U * self.S_sqrt[None, :]

        # Construct data projection.
        self.P = B.transpose(self.H) / self.S_sqrt[:, None] ** 2

    def optimal_U(self, x, y):
        """Approximate the optimal `U`.

        Note:
            This assumes no noise on the latent processes.

        Args:
            x (tensor): Inputs of data.
            y (tensor): Output of data.
        """
        # Construct first `A`.
        raise NotImplementedError()
        A = None

        # Greedy approximation of optimal U.
        U, _, _ = B.svd(As[0])
        us, V = [U[:, :1]], U[:, 1:]
        for i in range(1, self.m):
            # Construct `A` for iteration `i`.
            raise NotImplementedError()
            A = None

            U, _, _ = B.svd(B.matmul(B.matmul(V, A, tr_a=True), V))
            us.append(B.matmul(V, U[:, :1]))
            V = B.matmul(V, U[:, 1:])
        self.U = B.concat(us, axis=1)

        # Reconstruct mixing matrix and projection.
        self._construct_H_and_P()

    def observe(self, x, y):
        """Observe data.

        Args:
            x (tensor): Inputs.
            y (tensor): Outputs.
        """
        # Perform projection.
        ys = self.project(y)

        # Condition latent processes and noises.
        obses = [Obs(p(x), y) for p, y in zip(self.xs_noisy, ys)]
        self.xs = [p | obs for p, obs in zip(self.xs, obses)]
        self.xs_noisy = [p | obs for p, obs in zip(self.xs_noisy, obses)]
        self.xs_noises = [p | obs for p, obs in zip(self.xs_noises, obses)]

    def lml(self, x, y):
        """Compute the LML.

        Args:
            x (tensor): Inputs of data.
            y (tensor): Output of data.

        Returns:
            tensor: LML of data.
        """
        # Perform projection.
        ys_proj = self.project(y)

        # Add contributions of latent processes.
        lml = 0
        for p, n, yi in zip(self.xs_noisy, self.xs_noises, ys_proj):
            lml += p(x).logpdf(yi) - n(x).logpdf(yi)

        # Add regularisation contribution.
        noise_latent = B.matmul(self.H * B.array(self.noises_latent)[None, :],
                                self.H, tr_b=True)
        noise_obs = self.noise_obs * \
                    B.eye(self.p, dtype=B.dtype(self.noise_obs))
        d = Normal(noise_obs + noise_latent)
        lml += B.sum(d.logpdf(B.transpose(y)))

        return lml

    def project(self, y):
        """Project observations.

        Args:
            y (tensor): Data to project.

        Returns:
            tensor: Projected observations. Rows correspond to outputs and
                columns to time points.
        """
        return B.unstack(B.matmul(self.P, y, tr_b=True), axis=0)

    def marginals(self, x):
        """Compute marginals.

        Args:
            x (tensor): Inputs to construct marginals at.

        Returns:
            tuple[tensor]: Marginals per output, spatial means per time, and
                spatial variance per time.
        """

        # Extract means and variances of the latent processes.
        def extract(p):
            d = p(x)
            return d.mean, dense(d.var)

        x_means, x_vars = zip(*[extract(p) for p in self.xs])

        # Pull means through mixing matrix and unstack.
        y_means = B.matmul(self.H, B.concat(x_means, axis=1), tr_b=True)
        y_means_per_output = B.unstack(y_means, axis=0)
        y_means_per_time = [x[:, None] for x in B.unstack(y_means, axis=1)]

        # Get the diagonals: ignore temporal correlations.
        x_vars_per_time = B.unstack(B.stack(
            [B.diag(var) for var in x_vars], axis=1), axis=0)

        # Compute noise matrices.
        noise_latent = B.matmul(self.H * B.array(self.noises_latent)[None, :],
                                self.H, tr_b=True)
        noise_obs = self.noise_obs * \
                    B.eye(self.p, dtype=B.dtype(self.noise_obs))

        # Determine spatial variances.
        y_vars_per_time = []
        for x_vars in x_vars_per_time:
            # Pull through mixing matrix.
            f_var = B.matmul(self.H * x_vars[None, :], self.H, tr_b=True)
            y_vars_per_time.append(f_var + noise_obs + noise_latent)

        # Compute variances per output.
        y_vars_per_output = B.unstack(B.stack(
            [B.diag(var) for var in y_vars_per_time], axis=1), axis=0)

        # Return marginal predictions, means per time, and variances per time.
        return [(mean, mean - 2 * var ** .5, mean + 2 * var ** .5)
                for mean, var in zip(y_means_per_output, y_vars_per_output)], \
               y_means_per_time, y_vars_per_time
