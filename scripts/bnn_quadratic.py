import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lab.tf import B
from stheno import Normal, Diagonal, UniformDiagonal

from wbml import feedforward, vars32, VarsFrom, elbo

x = np.linspace(-10, 10, 100, dtype=np.float32)[None, :]
y = x ** 2 + 5 * np.random.randn(*x.shape)

# Normalise inputs and outputs.
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

nn_config = {
    'widths': (1, 20, 20, 20, 20, 1),
    'normalise': True,
    'nonlinearity': tf.nn.relu
}
its = 1000

# Get number of weights of NN.
# TODO: Refactor this.
n_w = B.shape_int(feedforward(**nn_config).weights())[0]

# Construct prior, q-distribution for weights, and likelihood.
p_w = Normal(UniformDiagonal(B.cast(1., dtype=np.float32), n_w))
q_w = Normal(Diagonal(1e-3 * vars32.pos(shape=[n_w])),
             vars32.get(shape=[n_w, 1]))
lik_noise = vars32.pos(1e-1)


def lik(w):
    f = feedforward(**dict(nn_config, vars=VarsFrom(w)))(x)
    return B.sum(Normal(UniformDiagonal(lik_noise, 1), f).log_pdf(y))


# Construct objective
obj = -elbo(lik, p_w, q_w)

# Peform optimisation.
s = tf.Session()
opt = tf.train.AdamOptimizer(1e-3).minimize(obj)
s.run(tf.global_variables_initializer())
for i in range(its):
    _, val = s.run([opt, obj])
    if i % 100 == 0:
        print(i, val)

# Make predictions.
nn = feedforward(**dict(nn_config, vars=VarsFrom(q_w.sample())))
pred = nn(x)[0, :]

# Estimate predictive.
preds = [s.run(pred) for _ in range(50)]
mean = np.mean(preds, axis=0)
lower = np.percentile(preds, 2.5, axis=0)
upper = np.percentile(preds, 100 - 2.5, axis=0)

# Plot results.
x, y = x.flatten(), y.flatten()
plt.figure()
plt.scatter(x, y, label='Observed', marker='o')
plt.plot(x, lower, c='tab:orange', ls='--')
plt.plot(x, upper, c='tab:orange', ls='--')
plt.plot(x, mean, c='tab:orange', ls='-', label='Prediction')
plt.legend()
plt.show()
