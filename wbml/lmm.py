# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from lab import B
from plum import Dispatcher, Referentiable, Self
from stheno import GP, Delta, Graph, Normal, Obs, dense

__all__ = ['LMMPP', 'OLMM']

log = logging.getLogger(__name__)


def _to_tuples(x, y):
    """Extract tuples with the input locations, output index,
    and observations from a matrix of observations.

    Args:
        x (tensor): Input locations.
        y (tensor): Outputs.

    Returns:
        list[tuple]: List of tuples with the input locations, output index,
            and observations.
    """
    xys = []
    for i in range(B.shape(y)[1]):
        mask = ~B.isnan(y[:, i])
        if B.any(mask):
            xys.append((x[mask], i, y[mask, i]))

    # Ensure that any data was extracted.
    if len(xys) == 0:
        raise ValueError('No data was extracted.')

    return xys


class LMMPP(Referentiable):
    """PP implementation of the linear mixing model.

    Args:
        kernels (list[:class:`stheno.Kernel`]) Kernels.
        noise_obs (tensor): Observation noise. One.
        noises_latent (tensor): Latent noises. One per latent processes.
        H (tensor): Mixing matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, kernels, noise_obs, noises_latent, H):
        self.graph = Graph()
        self.p, self.m = B.shape(H)

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

    @_dispatch(tuple, [tuple])
    def observe(self, *xys):
        """Observe data.

        Args:
            x (tensor): Input locations.
            y (tensor): Observed values.
        """
        # Condition all processes on all evidence.
        obs = Obs(*[(self.ys[i](x), y) for x, i, y in xys])
        self.fs = [p | obs for p in self.fs]
        self.ys = [p | obs for p in self.ys]
        self.y = self.y | obs

    @_dispatch(object, B.Numeric)
    def observe(self, x, y):
        self.observe(*_to_tuples(x, y))

    @_dispatch(tuple, [tuple])
    def logpdf(self, *xys):
        """Compute the logpdf of data.

        Args:
            x (tensor): Input locations.
            y (tensor): Observed values.

        Returns:
            tensor: Logpdf of data.
        """
        ys = list(self.ys)  # Make a copy so that we can modify it.

        # Compute the LML using the product rule.
        logpdf = 0
        for x, i, y in xys:
            # Compute `log p(y_i | y_{1:i - 1})`.
            logpdf += ys[i](x).logpdf(y)

            # Condition the remainder on the observation for `y_i`.
            obs = Obs(ys[i](x), y)
            ys = [p | obs for p in ys]

        return logpdf

    @_dispatch(object, B.Numeric)
    def logpdf(self, x, y):
        return self.logpdf(*_to_tuples(x, y))

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
        for i in range(B.shape(x)[0]):
            d = self.y(x[i])
            means.append(d.mean)
            vars.append(dense(d.var))
        return preds, means, vars

    def sample(self, x, latent=True):
        """Sample data.

        Args:
            x (tensor): Inputs to sample at.
            latent (bool, optional): Sample latent processes instead of the
                observed values. Defaults to `True`.
        """
        ps = self.fs if latent else self.ys
        samples = self.graph.sample(*[p(x) for p in ps])
        return B.concat(*samples, axis=1)


class OLMM(Referentiable):
    """Orthogonal linear mixing model.

    Args:
        kernels (list[:class:`stheno.Kernel`]) Kernels.
        noise_obs (tensor): Observation noise. One.
        noises_latent (tensor): Latent noises. One per latent processes.
        H (tensor): Mixing matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch({list, tuple},
               B.Numeric,
               B.Numeric,
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

    @_dispatch({list, tuple}, B.Numeric, {B.Numeric, list}, B.Numeric)
    def __init__(self, kernels, noise_obs, noises_latent, H):
        U, S_sqrt, _ = B.svd(H)
        OLMM.__init__(self, kernels, noise_obs, noises_latent, U, S_sqrt)

    def _construct_H_and_P(self):
        """Construct mixing matrix and projection."""
        self.H = self.U * self.S_sqrt[None, :]
        self.P = B.transpose(self.H) / self.S_sqrt[:, None] ** 2

    def optimal_U(self, x, y):  # pragma: no cover
        """Approximate the optimal `U`.

        Args:
            x (tensor): Input locations.
            y (tensor): Observed values.
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
        self.U = B.concat(*us, axis=1)

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

    def logpdf(self, x, y):
        """Compute the logpdf.

        Args:
            x (tensor): Inputs of data.
            y (tensor): Output of data.

        Returns:
            tensor: Logpdf of data.
        """
        # Perform projection.
        ys_proj = self.project(y)

        # Add contributions of latent processes.
        lml = 0
        for p, n, yi in zip(self.xs_noisy, self.xs_noises, ys_proj):
            lml += p(x).logpdf(yi) - n(x).logpdf(yi)

        # Add regularisation contribution.
        lml += self.likelihood(y)

        return lml

    @property
    def likelihood_covariance(self):
        """Covariance of the likelihood."""
        noise_latent = \
            B.matmul(self.H * self.noises_latent[None, :], self.H, tr_b=True)
        noise_obs = self.noise_obs * B.eye(B.dtype(self.noise_obs), self.p)
        return noise_latent + noise_obs

    def likelihood(self, y, mean=0):
        """Compute the likelihood of data for a given mean.

        Args:
            y (tensor): Data to evaluate likelihood at.
            mean (tensor): Mean of likelihood. Defaults to zero.

        Returns:
            tensor: Likelihood of `y` given `mean`.
        """
        d = Normal(self.likelihood_covariance)
        return B.sum(d.logpdf(B.transpose(y - mean)))

    def project(self, y):
        """Project observations.

        Args:
            y (tensor): Data to project.

        Returns:
            list[tensor]: Projected observations per output.
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
        x_means, x_vars = \
            zip(*[(p.mean(x), p.kernel.elwise(x)[:, 0]) for p in self.xs])

        # Pull means through mixing matrix and unstack.
        y_means = B.matmul(self.H, B.concat(*x_means, axis=1), tr_b=True)
        y_means_per_output = B.unstack(y_means, axis=0)
        y_means_per_time = [x[:, None] for x in B.unstack(y_means, axis=1)]

        # Get the diagonals: ignore temporal correlations.
        x_vars_per_time = B.unstack(B.stack(*x_vars, axis=1), axis=0)

        # Determine spatial variances.
        y_vars_per_time = []
        lik_cov = self.likelihood_covariance
        for x_vars in x_vars_per_time:
            # Pull through mixing matrix.
            f_var = B.matmul(self.H * x_vars[None, :], self.H, tr_b=True)
            y_vars_per_time.append(f_var + lik_cov)

        # Compute variances per output.
        diags = [B.diag(var) for var in y_vars_per_time]
        y_vars_per_output = B.unstack(B.stack(*diags, axis=1), axis=0)

        # Return marginal predictions, means per time, and variances per time.
        return [(mean, mean - 2 * var ** .5, mean + 2 * var ** .5)
                for mean, var in zip(y_means_per_output, y_vars_per_output)], \
               y_means_per_time, y_vars_per_time

    def sample(self, x, latent=True):
        """Sample model.

        Args:
            x (tensor): Points to sample at.
            latent (bool): Sample latent processes instead of observed values.
                Defaults to `True`.

        Returns:
            tensor: Sample.
        """
        ps = self.xs if latent else self.xs_noisy
        x_samples = B.concat(*[p(x).sample() for p in ps], axis=1)
        samples = B.matmul(x_samples, self.H, tr_b=True)
        return samples
