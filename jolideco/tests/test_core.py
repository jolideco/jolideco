from webbrowser import MacOSX
import numpy as np
import pytest
from numpy.testing import assert_allclose
from requests import session

from jolideco.core import MAPDeconvolver, MAPDeconvolverResult
from jolideco.data import gauss_and_point_sources_gauss_psf
from jolideco.priors import Priors, UniformPrior

RANDOM_STATE = np.random.RandomState(642020)


@pytest.fixture(scope="session")
def datasets():
    datasets = []

    for idx in range(3):
        dataset = gauss_and_point_sources_gauss_psf(
            random_state=RANDOM_STATE,
        )
        datasets.append(dataset)

    return datasets


@pytest.fixture(scope="session")
def deconvolver_result(datasets):
    priors = Priors()
    priors["flux"] = UniformPrior()

    deco = MAPDeconvolver(
        n_epochs=100,
        learning_rate=0.1,
        loss_function_prior=priors,
        upsampling_factor=1,
        use_log_flux=True,
    )

    fluxes_init = {"flux": RANDOM_STATE.gamma(20, size=(32, 32))}

    result = deco.run(datasets=datasets, fluxes_init=fluxes_init)
    return result


def test_map_deconvolver_str():
    deco = MAPDeconvolver(n_epochs=1_000)
    assert "n_epochs" in str(deco)


def test_map_deconvolver_result(deconvolver_result):
    assert_allclose(deconvolver_result.flux_total[12, 12], 1.551957, rtol=1e-3)
    assert_allclose(deconvolver_result.flux_total[0, 0], 0.204177, rtol=1e-3)


def test_map_deconvolver_result_io(deconvolver_result, tmpdir):
    filename = tmpdir / "result.fits"
    deconvolver_result.write(filename)

    result = MAPDeconvolverResult.read(filename=filename)

    assert result.config["n_epochs"] == 100
    assert_allclose(result.flux_total[12, 12], 1.551957, rtol=1e-3)
    assert_allclose(result.flux_total[0, 0], 0.204177, rtol=1e-3)
