import lab as B
from plum import Dispatcher
from stheno import Normal

__all__ = ["elbo"]

_dispatch = Dispatcher()


@_dispatch
def elbo(lik, p, q, num_samples=1):
    """Construct the ELBO.

    Args:
        lik (function): Likelihood that taken in one or more samples from the
            approximate posterior.
        p (distribution): Prior.
        q (distribution): Approximate posterior.
        num_samples (int, optional): Number of samples. Defaults to `1`.

    Returns:
        tensor: ELBO.
    """
    samples = q.sample(num_samples)
    log_lik = B.mean(lik(samples))
    log_p = B.mean(p.logpdf(samples))
    log_q = B.mean(q.logpdf(samples))
    return log_lik + log_p - log_q


@_dispatch
def elbo(lik, p: Normal, q: Normal, num_samples=1):
    return B.mean(lik(q.sample(num_samples))) - q.kl(p)
