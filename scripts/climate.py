import argparse
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from lab.torch import B
from sklearn.covariance import ledoit_wolf
from stheno import EQ, Normal

from wbml import Vars, OLMM
from wbml.data.climate import load

B.epsilon = 1e-8


def to_np(x):
    """Convert a PyTorch tensor to NumPy.

    Args:
        x (tensor): PyTorch tensor.

    Returns:
        tensor: `x` in NumPy.
    """
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.detach().numpy()


parser = argparse.ArgumentParser()
parser.add_argument('-m', type=int, default=100,
                    help='Number of latent processes.')
parser.add_argument('-i', '--its', type=int, default=20,
                    help='Number of optimiser iterations.')
parser.add_argument('-p', '--plot', action='store_true', help='Plot.')
parser.add_argument('-e', '--explore', action='store_true', help='Explore.')
parser.add_argument('-n', '--model', type=int, default=0, help='Simulator to model.')
args = parser.parse_args()


# # Set up results file.
# f = open("results/smoothing.csv", 'w')
# f.write("model_number, naive_rmse, smoothed_rmse, internal_rmse\n")
# f.close()

# Iterate over all of the data.
Nte = 1_000
obs, sims = load()
for n_sim in range(args.model, args.model+1):

    # Model parameters:
    m = args.m
    noise_latent = 0.05

    # Experiment parameters:
    noise_obs = 0.1
    optimiser_iterations = args.its

    # Load data:
    y_train = sims[n_sim].reshape(sims[n_sim].shape[0], -1)[:500, :]
    x_train = np.arange(y_train.shape[0])[:, None]

    y_test = sims[n_sim].reshape(sims[n_sim].shape[0], -1)[:Nte, :]
    x_test = np.arange(y_test.shape[0])[:Nte, None]
    y_obs = obs.reshape(obs.shape[0], -1)[:Nte, :]

    # If number of latent processes is set to `-1`, make it equal to the number
    # of outputs
    if m == -1:
        m = y_train.shape[1]

    # Compute observation noise.
    noise_obs *= np.std(y_train) ** 2

    # Extract sizes.
    p = y_train.shape[1]
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    # Construct mixing matrix.
    U, S, _ = np.linalg.svd(ledoit_wolf(y_train)[0])
    H = U.dot(np.diag(S ** .5)[:, :m])

    # Convert to PyTorch tensors.
    x_train = torch.tensor(x_train, dtype=torch.double)
    y_train = torch.tensor(y_train, dtype=torch.double)
    x_test = torch.tensor(x_test, dtype=torch.double)
    y_test = torch.tensor(y_test, dtype=torch.double)
    y_obs = torch.tensor(y_obs, dtype=torch.double)

    # Create variable manager.
    vs = Vars(torch.double)

    def new_lmm():
        """Construct a new LMM."""
        return OLMM(
            # Kernels:
            [EQ().stretch(vs.pos(30., name=('eq_ls', i))).select([-1]) *
             vs.pos(1., name=('eq_s2', i))
             for i in range(m)],

            # Observation noise:
            vs.pos(noise_obs, name='noise_obs'),

            # Noises on the latent processes:
            B.stack([vs.pos(noise_latent, name=('noise_lat', i))
                     for i in range(m)], axis=0),

            # Mixing matrix:
            vs.get(H, name='H')
        )

    # Plot first three and last three latent processes.
    if args.explore:
        lmm = new_lmm()
        x_proj_train = lmm.project(y_train)
        x_proj_test = lmm.project(y_test)
        plt.figure(figsize=(20, 10))
        for i in range(3):
            plt.subplot(2, 3, 1 + i)
            plt.title('Latent Process {}'.format(i))
            plt.plot(to_np(x_train[:, -1]), to_np(x_proj_train[i]))
            plt.plot(to_np(x_test[:, -1]), to_np(x_proj_test[i]))
        for i in range(3):
            plt.subplot(2, 3, 6 - i)
            plt.title('Latent Process -{}'.format(i + 1))
            plt.plot(to_np(x_train[:, -1]), to_np(x_proj_train[-(i + 1)]))
            plt.plot(to_np(x_test[:, -1]), to_np(x_proj_test[-(i + 1)]))
        plt.show()

    # Instantiate optimiser.
    new_lmm()  # Instantiate variables.
    opt = torch.optim.LBFGS(vs.get_vars(), lr=1, max_iter=10)

    def objective():
        """NLML objective.

        Also does a backwards pass.
        """
        nlml = -new_lmm().lml(x_train, y_train) / n_train / p
        opt.zero_grad()
        nlml.backward()
        print('NLML: {:.3e}'.format(nlml.detach().numpy()))
        return nlml

    # Perform optimisation.
    for i in range(optimiser_iterations):
        print('Optimisation iteration: {:3d}/{:3d}'
              ''.format(i + 1, optimiser_iterations))
        start = time()
        opt.step(objective)

    # Condition and predict.
    lmm = new_lmm()
    lmm.observe(x_test, y_test)
    preds, means, vars = lmm.predict(x_test)

    # Compute scores for OLMM.
    lmls, mses, mses_obs = [], [], []
    for i in range(n_test):
        mean, var, y = means[i], vars[i], B.transpose(y_test[i:i + 1, :])
        y_obs_ = B.transpose(y_obs[i:i + 1, :])
        mses.append(B.mean((y - mean) ** 2).detach().numpy())
        lmls.append(-Normal(var, mean).logpdf(y).detach().numpy())
        mses_obs.append(B.mean((y_obs_ - mean) ** 2).detach().numpy())

    # Print scores for OLMM.
    lml_mean = np.mean(lmls)
    lml_std = np.std(lmls)
    # lml_full = -lmm.lml(x_test, y_test).detach().numpy() / 24
    print('Scores:')
    print('  NLML: {:-6.0f} [{:-6.0f}, {:-6.0f}]'
          ''.format(lml_mean, lml_mean - 2 * lml_std, lml_mean + 2 * lml_std))
    print('  RMSE: {:-6.2f}'.format(np.mean(mses) ** .5))

    print(' RMSE naive: {:-6.2f}'.format(torch.sqrt(torch.mean((y_test - y_obs) ** 2))))
    print(' RMSE smoothed: {:-6.2f}'.format(np.mean(mses_obs) ** .5))

    def plot_prediction(x, pred, c, f=None, x_obs=None, y_obs=None, label=None):
        if f is not None:
            plt.plot(to_np(x), to_np(f), label='True', c='tab:blue')
        if x_obs is not None:
            plt.scatter(to_np(x_obs), to_np(y_obs), label='Observations',
                        c='tab:red')
        mean, lower, upper = pred
        if label is None:
            postfix = ''
        else:
            postfix = ' ({})'.format(label)
        plt.plot(to_np(x), to_np(mean), label='Prediction' + postfix,
                 c='tab:{}'.format(c))
        plt.plot(to_np(x), to_np(lower), ls='--', c='tab:{}'.format(c))
        plt.plot(to_np(x), to_np(upper), ls='--', c='tab:{}'.format(c))
        plt.legend()

    # Plot predictions for 16 randomly chosen outputs.
    if args.plot:
        plt.figure(figsize=(20, 10))
        for num, i in enumerate(sorted(np.random.permutation(p)[:16])):
            plt.subplot(4, 4, num + 1)
            plt.title('Output {}'.format(i + 1))
            plot_prediction(x_test[:, -1], preds[i], 'green', f=y_test[:, i])
        plt.show()

    rmse_naive = torch.sqrt(torch.mean((y_test - y_obs) ** 2))
    rmse_smoothed = np.mean(mses_obs) ** .5
    rmse_internal = np.mean(mses) ** .5
    f = open("results/smoothing.csv", 'a')
    f.write(f'{n_sim},{rmse_naive},{rmse_smoothed},{rmse_internal}\n')
    f.close()

# Iterate over all of the data.
obs, sims = load()
mean_model = torch.tensor(B.zeros(Nte, obs.shape[1] * obs.shape[2]), dtype=torch.double)
for n_sim in range(28):
    y_test = sims[n_sim].reshape(sims[n_sim].shape[0], -1)[:Nte, :]
    mean_model = mean_model + torch.tensor(y_test, dtype=torch.double)

mean_model /= 28

obs_ = obs.reshape(obs.shape[0], -1)[:Nte, :]
rmse_mean = B.sqrt(B.mean((mean_model - torch.tensor(obs_, dtype=torch.double)) ** 2))
