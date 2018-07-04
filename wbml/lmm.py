# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from lab import B
from stheno import GP, Delta, Graph, Normal, At, SPD, Diagonal

__all__ = ['LMMPP', 'OLMM']


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
        self.xs = [GP(k, graph=self.graph) for k in kernels]

        # Create latent noise.
        self.es_x = [GP(noise * Delta(), graph=self.graph)
                     for noise in noises_latent]

        # Create noisy latent processes.
        self.xs_noisy = [x + e for x, e in zip(self.xs, self.es_x)]

        # Multiply with mixing matrix.
        self.fs = [0 for _ in range(self.p)]
        for i in range(self.p):
            for j in range(self.m):
                self.fs[i] += H[i, j] * self.xs_noisy[j]

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
        for i in range(self.p):
            self.ys[i].condition(x, y[:, i])

    def sample(self, x):
        """Sample data.

        Args:
            x (tensor): Inputs to sample at.
        """
        samples = self.graph.sample(*(At(y)(x) for y in self.ys))
        return B.concat(samples, axis=1)

    def lml(self, x, y):
        """Compute the LML.

        Note:
            Revert to prior afterwards.

        Args:
            x (tensor): Inputs of data.
            y (tensor): Output of data.

        Returns:
            tensor: LML of data.
        """
        lml = 0
        for i in range(self.p):
            lml += self.ys[i](x).log_pdf(y[:, i])[0]
            self.ys[i].condition(x, y[:, i])
        self.graph.revert_prior()
        return lml

    def predict(self, x):
        """Predict.

        Args:
            x (tensor): Inputs to predict at.

        Returns:
            tuple[tensor}: Marginals predictions, spatial means per time, and
                spatial variance per time.
        """
        preds = [y.predict(x) for y in self.ys]
        means, vars = [], []
        for i in range(B.shape_int(x)[0]):
            d = self.y(x[i])
            means.append(d.mean)
            vars.append(d.var)
        return preds, means, vars


class OLMM(object):
    """Orthogonal linear mixing model.

    Args:
        kernels (list[:class:`stheno.Kernel`]) Kernels.
        noise_obs (tensor): Observation noise. One.
        noises_latent (tensor): Latent noises. One per latent processes.
        H (tensor): Mixing matrix.
    """

    def __init__(self, kernels, noise_obs, noises_latent, H):
        # Determine number of outputs and latent processes.
        self.p, self.m = B.shape(H)

        # Save noises.
        self.noise_obs = noise_obs
        self.noises_latent = noises_latent

        # Deconstruct mixing matrix.
        self.U, self.S_sqrt, _ = B.svd(H)

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
        # Compute quadratic forms.
        n = B.shape(x)[0]
        Ks = [SPD(s ** 2 * p.kernel(x) +
                  self.noise_obs * B.eye(n, dtype=B.dtype(self.noise_obs)))
              for s, p in zip(self.S_sqrt, self.xs)]
        As = [B.matmul(y, y, tr_a=True) / self.noise_obs - K.quadratic_form(y)
              for K in Ks]

        # Greedy approximation of optimal U.
        U, _, _ = B.svd(As[0])
        us, V = [U[:, :1]], U[:, 1:]
        for i in range(1, self.m):
            U, _, _ = B.svd(B.matmul(B.matmul(V, As[i], tr_a=True), V))
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

        # Condition latent processes.
        for p, y in zip(self.xs_noisy, ys):
            p.condition(x, y)

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
            lml += p(x).log_pdf(yi)[0] - n(x).log_pdf(yi)[0]

        # Add regularisation contribution.
        noise_latent = B.matmul(self.H * B.array(self.noises_latent)[None, :],
                                self.H, tr_b=True)
        noise_obs = self.noise_obs * B.eye(self.p,
                                           dtype=B.dtype(self.noise_obs))
        d = Normal(noise_obs + noise_latent)
        lml += B.sum(d.log_pdf(B.transpose(y)))

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

    def predict(self, x):
        """Predict.

        Args:
            x (tensor): Inputs to predict at.

        Returns:
            tuple[tensor}: Marginals predictions, spatial means per time, and
                spatial variance per time.
        """

        # Extract means and variances of the latent processes.
        def extract(p):
            d = p(x)
            return d.mean, d.var

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
