import numpy as np
import tensorflow as tf
from lab.tf import B
from stheno.tf import Normal, Diagonal, UniformDiagonal

from wbml import elbo, vars32

s = tf.Session()

d = 2

# Define a prior and likelihood and generate data.
prior = Normal(Diagonal(np.linspace(1, 2, d, dtype=np.float32)))
lik_noise = Normal(UniformDiagonal(np.array(.1, dtype=np.float32), d))
z_true = s.run(prior.sample())
y = s.run((lik_noise + z_true).sample(100))

# Define likelihood and q distribution.
lik = lik_noise + y
q = Normal(Diagonal(vars32.positive(shape=[d])), vars32.get(shape=[d, 1]))

# Construct objective.
obj = -elbo(lik, prior, q, num_samples=1)

# Optimise.
opt = tf.train.AdamOptimizer(1e-2).minimize(obj)
s.run(tf.global_variables_initializer())
for i in range(5000):
    _, val = s.run([opt, obj])
    if i % 100 == 0:
        print(i, val)

# Print result.
print('True', z_true.flatten())
sample = q.sample(1000)
z_est = B.mean(sample, axis=1)
z_est_std = B.mean((sample - z_est[:, None]) ** 2, axis=1) ** .5
print('Mean estimate', s.run(z_est))
print('Std estimate', s.run(z_est_std))
