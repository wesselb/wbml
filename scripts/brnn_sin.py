import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lab.tf import B
from stheno.tf import Normal, UniformDiagonal, Diagonal

from wbml import normalise_01, normalise_norm, rnn, elbo, VarsFrom, vars32

x = np.linspace(0, 20, 40, dtype=np.float32)[:, None, None]
y = np.sin(x) + .1 * np.random.randn(*x.shape).astype(np.float32)

# Normalise inputs and outputs.
x = normalise_01(x)
y = normalise_norm(y)

f = rnn(1, 1, ((10, 10),), gru=False)
its = 5000

# Construct prior, q-distribution for weights, and likelihood.
#   Prevent the KL from pulling too hard initially.
n_w = f.num_weights()
p_w = Normal(UniformDiagonal(vars32.pos(), n_w))
q_w = Normal(Diagonal(1e-2 * vars32.pos(shape=[n_w])),
             vars32.get(shape=[n_w, 1]))
lik_noise = vars32.pos(1e-4)


def lik(w):
    f.initialise(VarsFrom(w))
    dist = Normal(UniformDiagonal(lik_noise, 1), B.reshape(f(x), [1, -1]))
    return B.sum(dist.log_pdf(B.reshape(y, [1, -1])))


# Construct objective
obj = -elbo(lik, p_w, q_w)

# Perform optimisation.
s = tf.Session()
opt = tf.train.AdamOptimizer(1e-2).minimize(obj)
s.run(tf.global_variables_initializer())
for i in range(its):
    _, val = s.run([opt, obj])
    if i % 100 == 0:
        print(i, val)

# Make predictions.
f.initialise(VarsFrom(q_w.sample()))
pred = f(x)

# Estimate predictive.
preds = [s.run(pred).flatten() for _ in range(50)]
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
