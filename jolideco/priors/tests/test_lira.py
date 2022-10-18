import torch
from jolideco.priors import LIRAPrior


def test_lira_prior():
    prior = LIRAPrior(alphas=torch.ones(3))
    torch.testing.assert_close(prior.alphas, torch.ones(3))
