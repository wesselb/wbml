import lab as B
from stheno import Normal

from wbml.vi import elbo
from .util import approx


def rand_normal(n=3):
    cov = B.randn(n, n)
    cov = B.mm(cov, cov, tr_b=True)
    return Normal(B.randn(n, 1), cov)


def test_elbo():
    lik = rand_normal()
    p = rand_normal()
    q = rand_normal()

    # Check that the two implementations are consistent.
    estimate1 = elbo.invoke(object, Normal, Normal)(lik.logpdf, p, q, num_samples=50000)
    estimate2 = elbo.invoke(object, object, object)(lik.logpdf, p, q, num_samples=50000)
    approx(estimate1, estimate2, rtol=1e-2)
