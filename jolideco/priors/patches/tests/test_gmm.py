import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from jolideco.priors.patches import GMM_REGISTRY, GaussianMixtureModel


def test_gmm_torch_basic():
    means = np.linspace(-1, 1, 9).reshape((1, 9))
    covariances = np.array([np.eye(9) for _ in means])
    weights = np.arange(9)
    weights = weights / weights.sum()

    gmm_torch = GaussianMixtureModel(
        means=means, covariances=covariances, weights=weights, stride=None
    )

    gmm = GaussianMixture()
    gmm.weights_ = weights
    gmm.covariances_ = covariances
    gmm.means_ = means
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, "full")

    x = np.ones((2, 9), dtype=np.float32)

    assert gmm_torch.patch_shape == (3, 3)
    result_ref = gmm._estimate_log_prob(X=x)
    result = gmm_torch.estimate_log_prob(x=torch.from_numpy(x))
    result = result.detach().numpy()
    assert_allclose(result_ref, result)


@pytest.mark.parametrize("name", GMM_REGISTRY)
def test_gmm_registry(name):
    gmm = GaussianMixtureModel.from_registry(name=name)

    x = torch.ones((2, 64))

    values = gmm.estimate_log_prob(x=x)

    assert values.shape == (2, gmm.n_components)
    assert name in str(gmm)
