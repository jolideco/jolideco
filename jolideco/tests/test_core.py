import numpy as np
import pytest
from numpy.testing import assert_allclose

from jolideco.core import MAPDeconvolver, MAPDeconvolverResult
from jolideco.data import disk_source_gauss_psf, gauss_and_point_sources_gauss_psf
from jolideco.models import FluxComponent, FluxComponents
from jolideco.priors import Priors, UniformPrior

RANDOM_STATE = np.random.RandomState(642020)


@pytest.fixture(scope="session")
def datasets_gauss():
    datasets = []

    for idx in range(3):
        dataset = gauss_and_point_sources_gauss_psf(
            random_state=RANDOM_STATE,
        )
        datasets.append(dataset)

    return datasets


@pytest.fixture(scope="session")
def datasets_disk():
    datasets = []

    for idx in range(3):
        dataset = disk_source_gauss_psf(
            random_state=RANDOM_STATE,
        )
        datasets.append(dataset)

    return datasets


@pytest.fixture(scope="session")
def deconvolver_result(datasets_gauss):
    priors = Priors()
    priors["flux-1"] = UniformPrior()

    deco = MAPDeconvolver(
        n_epochs=100,
        learning_rate=0.1,
        loss_function_prior=priors,
        upsampling_factor=1,
    )

    fluxes_init = RANDOM_STATE.gamma(20, size=(32, 32))

    components = FluxComponents()
    components["flux-1"] = FluxComponent.from_flux_init_numpy(flux_init=fluxes_init)

    result = deco.run(datasets=datasets_gauss, components=components)
    return result


def test_map_deconvolver_str():
    deco = MAPDeconvolver(n_epochs=1_000)
    assert "n_epochs" in str(deco)


def test_map_deconvolver_result(deconvolver_result):
    assert_allclose(deconvolver_result.flux_total[12, 12], 1.551957, rtol=1e-3)
    assert_allclose(deconvolver_result.flux_total[0, 0], 0.204177, rtol=1e-3)

    trace_loss = deconvolver_result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 5816.533375, rtol=1e-3)
    assert_allclose(trace_loss["dataset-0"], 1949.894236, rtol=1e-3)
    assert_allclose(trace_loss["dataset-1"], 1943.252298, rtol=1e-3)
    assert_allclose(trace_loss["dataset-2"], 1923.386841, rtol=1e-3)


def test_map_deconvolver_result_io(deconvolver_result, tmpdir):
    filename = tmpdir / "result.fits"
    deconvolver_result.write(filename)

    result = MAPDeconvolverResult.read(filename=filename)

    assert result.config["n_epochs"] == 100
    assert_allclose(result.flux_total[12, 12], 1.551957, rtol=1e-3)
    assert_allclose(result.flux_total[0, 0], 0.204177, rtol=1e-3)


def test_map_deconvolver_result_plot(deconvolver_result):
    deconvolver_result.plot_fluxes()
    deconvolver_result.plot_trace_loss()


def test_map_deconvolver_usampling(datasets_disk):
    priors = Priors()
    priors["flux-1"] = UniformPrior()

    deco = MAPDeconvolver(
        n_epochs=100,
        learning_rate=0.1,
        loss_function_prior=priors,
        upsampling_factor=2,
    )

    flux_init = RANDOM_STATE.gamma(20, size=(32, 32))

    components = FluxComponents()
    components["flux-1"] = FluxComponent.from_flux_init_numpy(
        flux_init=flux_init, upsampling_factor=2
    )

    result = deco.run(datasets=datasets_disk, components=components)

    assert result.flux_upsampled_total.shape == (64, 64)
    assert_allclose(result.flux_total[12, 12], 2.960147, rtol=1e-3)
    assert_allclose(result.flux_total[0, 0], 0.894996, rtol=1e-3)

    trace_loss = result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 5895.711304, rtol=1e-3)
    assert_allclose(trace_loss["dataset-0"], 1970.38928, rtol=1e-3)
    assert_allclose(trace_loss["dataset-1"], 1938.160889, rtol=1e-3)
    assert_allclose(trace_loss["dataset-2"], 1987.161133, rtol=1e-3)
