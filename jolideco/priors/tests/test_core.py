import torch
from jolideco.priors import UniformPrior


def test_uniform_prior():
    prior = UniformPrior()
    x = torch.ones((2, 2))
    value = prior(flux=x)
    torch.testing.assert_close(value, torch.tensor(0))
