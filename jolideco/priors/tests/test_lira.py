import torch
from jolideco.priors import LIRAPrior


def test_lira_prior():
    prior = LIRAPrior(alphas=torch.ones(3))
    torch.testing.assert_allclose(prior.alphas, [1, 1, 1])
