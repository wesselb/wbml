import argparse
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from lab.torch import B
from sklearn.covariance import ledoit_wolf
from stheno import EQ, Normal, Exp

from wbml import Vars, OLMM


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
parser.add_argument('split', type=int, help='Test split.')
parser.add_argument('-m', type=int, default=200,
                    help='Number of latent processes.')
parser.add_argument('-i', '--its', type=int, default=0,
                    help='Number of optimiser iterations.')
parser.add_argument('-w', '--weeks', type=int, default=3,
                    help='Number of weeks.')
parser.add_argument('-p', '--plot', action='store_true', help='Plot.')
parser.add_argument('-e', '--explore', action='store_true', help='Explore.')
parser.add_argument('-u', action='store_true', help='Optimal U.')
args = parser.parse_args()

# Model parameters:
m = args.m
noise_latent = 0.05

# Experiment parameters:
weeks = args.weeks
noise_obs_factor = 1.5
split = args.split
optimiser_iterations = args.its

# Load data.
x_train = np.genfromtxt('data/{}weeks/split_{}_xtrain.csv'
                        ''.format(weeks, split), delimiter=',')
y_train = np.genfromtxt('data/{}weeks/split_{}_ytrain.csv'
                        ''.format(weeks, split), delimiter=',')
x_test = np.genfromtxt('data/{}weeks/split_{}_xtest.csv'
                       ''.format(weeks, split), delimiter=',')
y_test = np.genfromtxt('data/{}weeks/split_{}_ytest.csv'
                       ''.format(weeks, split), delimiter=',')

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

# Create variable manager.
vs = Vars(torch.double)


def new_lmm(init=False):
    """Construct a new LMM."""
    lmm = OLMM(
        # Kernels:
        kernels=[
            # Exp:
            Exp().stretch(vs.pos(24., name=('exp_ls', i))).select([-1]) *
            vs.pos(.3, name=('exp_s2', i)) +

            # Exp:
            EQ().stretch(vs.pos(24., name=('eq_ls', i))).select([-1]) *
            vs.pos(.3, name=('eq_s2', i)) +

            # Constant:
            vs.pos(.2, name=('const', i)) +

            # Daily periodic, changing and modulated by load:
            EQ().stretch(vs.pos(5.0, name=('daily_ls_load', i))).select([4]) *
            EQ().stretch(vs.pos(0.5, name=('daily_ls_time', i)))
                .periodic(24.).select([-1]) *
            EQ().stretch(vs.pos(24. * 4., name=('daily_ls_time_window', i))) \
                .select([-1]) *
            vs.pos(.2, name=('daily_s2', i))

            for i in range(m)],

        # Observation noise:
        noise_obs=vs.pos(noise_obs_factor * np.sum(S[m:]) / p, name='noise'),

        # Noises on the latent processes:
        noises_latent=B.stack([vs.pos(noise_latent, name=('noise', i))
                               for i in range(m)], axis=0),

        # Mixing matrix:
        H=vs.get(H, name='H')
    )

    # Construct optimal U.
    if args.u and not init:
        lmm.optimal_U(x_train, y_train)

    return lmm


# Plot first and last latent processes.
if args.explore:
    lmm = new_lmm()
    x_proj_train = lmm.project(y_train)
    x_proj_test = lmm.project(y_test)
    plt.figure(figsize=(20, 10))
    for i in range(6):
        plt.subplot(4, 3, 1 + i)
        plt.title('Latent Process {}'.format(i))
        plt.plot(to_np(x_train[:, -1]), to_np(x_proj_train[i]))
        plt.plot(to_np(x_test[:, -1]), to_np(x_proj_test[i]))
    for i in range(6):
        plt.subplot(4, 3, 12 - i)
        plt.title('Latent Process -{}'.format(i + 1))
        plt.plot(to_np(x_train[:, -1]), to_np(x_proj_train[-(i + 1)]))
        plt.plot(to_np(x_test[:, -1]), to_np(x_proj_test[-(i + 1)]))
    plt.show()

# Instantiate optimiser.
new_lmm(init=True)  # Instantiate variables.
opt = torch.optim.Adam(vs.get_vars(), lr=5e-2)


def objective():
    """NLML objective.

    Also does a backwards pass.
    """
    nlml = -new_lmm().lml(x_train, y_train) / n_train / p
    opt.zero_grad()
    nlml.backward()
    print('LML:', nlml)
    return nlml


# Perform optimisation.
for i in range(optimiser_iterations):
    print('\nPerforming gradient step {}/{}...'
          ''.format(i + 1, optimiser_iterations))
    start = time()
    opt.step(objective)
    print('Took {:.2f} seconds.'.format(time() - start))
    print('Noise:', vs['noise'])

# Condition and predict.
lmm = new_lmm()
lmm.observe(x_train, y_train)
preds, means, vars = lmm.predict(x_test)

# Compute scores for OLMM.
lmls, mses = [], []
for i in range(n_test):
    mean, var, y = means[i], vars[i], B.transpose(y_test[i:i + 1, :])
    mses.append(B.mean((y - mean) ** 2).detach().numpy())
    lmls.append(-Normal(var, mean).log_pdf(y).detach().numpy())

# Print scores for OLMM.
lml_mean = np.mean(lmls)
lml_std = np.std(lmls)
lml_full = -lmm.lml(x_test, y_test).detach().numpy() / 24
print('Scores for split {} ({} weeks):'.format(split, weeks))
print('  OLMM:')
print('    NLML: {:-6.0f} [{:-6.0f}, {:-6.0f}] (Full: {:-6.0f})'
      ''.format(lml_mean, lml_mean - 2 * lml_std, lml_mean + 2 * lml_std,
                lml_full))
print('    RMSE: {:-6.2f}'.format(np.mean(mses) ** .5))

# Print scores for LW.
mse = np.genfromtxt('data/{}weeks/mse.csv'.format(weeks))[split - 1]
lml_mean = np.genfromtxt('data/{}weeks/meanLW.csv'
                         ''.format(weeks))[split - 1]
if weeks == 3:
    lml_std = np.genfromtxt('data/{}weeks/varLW.csv'
                            ''.format(weeks))[split - 1] ** .5
else:
    lml_std = 0
print('  LW:')
print('    NLML: {:-6.0f} [{:-6.0f}, {:-6.0f}]'
      ''.format(lml_mean, lml_mean - 2 * lml_std, lml_mean + 2 * lml_std))
print('    RMSE: {:-6.2f}'.format(mse ** .5))

# Print scores for EB.
mse = np.genfromtxt('data/{}weeks/mse.csv'.format(weeks))[split - 1]
lml_mean = np.genfromtxt('data/{}weeks/meanEB.csv'
                         ''.format(weeks))[split - 1]
if weeks == 3:
    lml_std = np.genfromtxt('data/{}weeks/varEB.csv'
                            ''.format(weeks))[split - 1] ** .5
else:
    lml_std = 0
print('  EB:')
print('    NLML: {:-6.0f} [{:-6.0f}, {:-6.0f}]'
      ''.format(lml_mean, lml_mean - 2 * lml_std, lml_mean + 2 * lml_std))
print('    RMSE: {:-6.2f}'.format(mse ** .5))


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


# Plot predictions of 16 randomly chosen outputs.
if args.plot:
    plt.figure(figsize=(20, 10))
    for num, i in enumerate(np.random.permutation(p)[:16]):
        plt.subplot(4, 4, num + 1)
        plt.title('Output {}'.format(i + 1))
        plot_prediction(x_test[:, -1], preds[i], 'green', f=y_test[:, i])
    plt.show()
