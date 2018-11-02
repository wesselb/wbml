import argparse
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch, stheno
from lab.torch import B
from sklearn.covariance import ledoit_wolf
from stheno import EQ, Normal, GP, Delta

from wbml import Vars, OLMM
from wbml.data.climate import load

B.epsilon = 1e-6


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
parser.add_argument('-i', '--its', type=int, default=500,
                    help='Number of optimiser iterations.')
parser.add_argument('-p', '--plot', action='store_true', help='Plot.')
parser.add_argument('-e', '--explore', action='store_true', help='Explore.')
parser.add_argument('-n', '--model', type=int, default=0, help='Simulator to model.')
args = parser.parse_args()


# # Set up results file.
# f = open("results/smoothing_independent.csv", 'w')
# f.write("model_number, naive_rmse, smoothed_rmse\n")
# f.close()

# Iterate over all of the data.
Nte = 1_000
obs, sims = load()
for n_sim in range(args.model, args.model+1):

    # Cache results for each independent GP:
    lml_cache = np.zeros(obs.shape[1:])
    mse_cache = np.zeros(obs.shape[1:])
    mse_obs_cache = np.zeros(obs.shape[1:])

    for p in B.arange(obs.shape[1]):
        for q in B.arange(obs.shape[2]):

            # Experiment parameters:
            optimiser_iterations = args.its

            # Construct train data:
            y_train_np = sims[n_sim][:, p, q][:500]
            x_train_np = np.arange(y_train_np.shape[0])

            # Construct test data:
            y_test_np = sims[n_sim][:, p, q][:Nte]
            x_test_np = np.arange(y_test_np.shape[0])

            # Compute appropriate observational data.
            y_obs_np = obs[:, p, q][:Nte]

            # Extract sizes:
            n_train = y_train_np.shape[0]
            n_test = y_test_np.shape[0]

            # Convert to PyTorch tensors:
            x_train = torch.tensor(x_train_np, dtype=torch.double)
            y_train = torch.tensor(y_train_np, dtype=torch.double)
            x_test = torch.tensor(x_test_np, dtype=torch.double)
            y_test = torch.tensor(y_test_np, dtype=torch.double)
            y_obs = torch.tensor(y_obs_np, dtype=torch.double)

            std_train = np.std(y_train_np)

            def new_gp():
                g = stheno.Graph()
                mu = vs.get(B.mean(y_train).detach(), name='mu')
                s_eq = vs.pos(0.75 * std_train + np.random.rand(), name='eq_s2')
                l_eq = vs.pos(50. + np.random.rand(), name='eq_ls')
                s_noise = vs.pos(0.25 * std_train + np.random.rand(), name='s2_noise')
                f_smooth = GP(EQ().stretch(l_eq + 10), graph=g) * (s_eq + 1e-2)
                f_noise = GP(Delta(), graph=g) * (s_noise + 1e-2)
                f_noise_2 = GP(Delta(), graph=g) * (s_noise + 1e-2)
                return f_smooth + f_noise + mu, f_smooth + f_noise_2 + mu

            success = False
            while not success:

                print(f'Optimising {p} {q}')

                # Create variable manager:
                vs = Vars(torch.double)

                # Instantiate variables and optimiser:
                new_gp()
                # opt = torch.optim.LBFGS(vs.get_vars(), lr=1, max_iter=10)
                opt = torch.optim.Adam(vs.get_vars(), amsgrad=True)

                def objective():
                    """NLML objective.

                    Also does a backwards pass.
                    """
                    # print(vs.get_vars())
                    nlml = -new_gp()[0](x_train).logpdf(y_train)
                    nlml = nlml / n_train
                    opt.zero_grad()
                    nlml.backward()
                    print('NLML: {:.3e}'.format(nlml.detach().numpy()))
                    return nlml

                # Try to optimise:
                try:
                    for i in range(optimiser_iterations):
                        print('Optimisation iteration: {:3d}/{:3d}'
                              ''.format(i + 1, optimiser_iterations))
                        start = time()
                        opt.step(objective)
                    success = True
                except:
                    print("Failure. Re-trying.")

            # Condition and predict:
            f, f2 = new_gp()
            f_post = f2 | (f(x_test), y_test)

            # Compute results for the (pq)^{th} independent GP:
            lml_cache[p, q] = f(x_test).logpdf(y_test).detach().numpy()
            mse_cache[p, q] = B.mean((y_test - f_post(x_test).mean.flatten()) ** 2).detach().numpy()
            mse_obs_cache[p, q] = B.mean((y_obs - f_post(x_test).mean.flatten()) ** 2).detach().numpy()

    print(lml_cache)
    print(mse_cache ** .5)
    print(mse_obs_cache ** .5)
    rmse_smoothed = np.mean(mse_obs_cache) ** .5
    rmse_internal = np.mean(mse_cache) ** .5
    f = open("results/smoothing_independent.csv", 'a')
    f.write(f'{n_sim},{rmse_smoothed},{rmse_internal}\n')
    f.close()
