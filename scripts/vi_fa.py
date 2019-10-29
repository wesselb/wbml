import lab.tensorflow as B
import tensorflow as tf
import wbml.out as out
from stheno.tensorflow import Normal, Diagonal
from varz.tensorflow import Vars, minimise_adam
from wbml.vi import elbo

d = 2
vs = Vars(tf.float32)

# Define a prior and likelihood and generate data.
prior = Normal(Diagonal(B.linspace(1, 2, d)))
lik_noise = Normal(Diagonal(0.1 * B.ones(d)))
z_true = prior.sample()
y = (lik_noise + z_true).sample(100)


# Define likelihood.
def lik(x):
    return B.sum((lik_noise + y).logpdf(x))


# Construct q distribution.
def construct_q(vs):
    q = Normal(Diagonal(vs.positive(shape=[d], name='q/var')),
               vs.get(shape=[d, 1], name='q/mean'))
    return q


# Construct objective.
def objective(vs):
    q = construct_q(vs)
    return -elbo(lik, prior, q, num_samples=1)


# Optimise.
objective_compiled = tf.function(objective, autograph=False)
minimise_adam(objective_compiled, vs, trace=True, iters=2000, rate=5e-2)

# Print result.
out.kv('True', z_true)
out.kv('Mean q', construct_q(vs).mean)
