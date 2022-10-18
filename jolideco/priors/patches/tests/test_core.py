import pytest
import torch
from jolideco.priors import GMMPatchPrior


@pytest.mark.xfail
def test_uniform_prior():
    # TODO: make gmm model available for download
    prior = GMMPatchPrior()
    x = torch.ones((2, 2))
    value = prior(flux=x)
    torch.testing.assert_allclose(value, 0)
