import tensorflow as tf
from lab.tf import B
from stheno.tf import Normal

from wbml.vi import QNormalDiag, elbo

s = tf.Session()

# Define a prior and likelihood and generate data.
prior = Normal(B.eye(2))
lik_noise = Normal(.1 * B.eye(2))
z_true = s.run(prior.sample())
y = s.run((lik_noise + z_true).sample(100))

# Define likelihood and q distribution.
lik = lik_noise + y
pars = QNormalDiag.random_init_pars(2)
q = QNormalDiag(*pars)

# Construct objective.
obj = -elbo(lik, prior, q)

# Optimise.
opt = tf.train.AdamOptimizer(5e-2).minimize(obj)
s.run(tf.global_variables_initializer())
for i in range(5000):
    _, val = s.run([opt, obj])
    print(i, val)

# Print result.
print('True', z_true.flatten())
sample = q.sample(1000)
z_est = B.mean(sample, axis=1)
z_est_std = B.mean((sample - z_est[:, None]) ** 2, axis=1) ** .5
print('Mean estimate', s.run(z_est))
print('Std estimate', s.run(z_est_std))
