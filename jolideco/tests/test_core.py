import numpy as np
import pytest
from numpy.testing import assert_allclose

from jolideco.core import MAPDeconvolver, MAPDeconvolverResult
from jolideco.data import disk_source_gauss_psf, gauss_and_point_sources_gauss_psf
from jolideco.models import FluxComponent, FluxComponents
from jolideco.priors import InverseGammaPrior, UniformPrior
from jolideco.priors.core import ExponentialPrior


@pytest.fixture(scope="session")
def datasets_gauss():
    datasets = []

    random_state = np.random.RandomState(642020)

    for idx in range(3):
        dataset = gauss_and_point_sources_gauss_psf(
            random_state=random_state,
        )
        datasets.append(dataset)

    return datasets


@pytest.fixture(scope="session")
def datasets_disk():
    datasets = []

    random_state = np.random.RandomState(642020)

    for idx in range(3):
        dataset = disk_source_gauss_psf(
            random_state=random_state,
        )
        datasets.append(dataset)

    return datasets


@pytest.fixture(scope="session")
def deconvolver_result(datasets_gauss):
    deco = MAPDeconvolver(
        n_epochs=100,
        learning_rate=0.1,
    )

    random_state = np.random.RandomState(642020)
    flux_init = random_state.gamma(20, size=(32, 32))

    components = FluxComponents()
    components["flux-1"] = FluxComponent.from_flux_init_numpy(
        flux_init=flux_init, prior=UniformPrior()
    )

    result = deco.run(datasets=datasets_gauss, components=components)
    return result


def test_map_deconvolver_str():
    deco = MAPDeconvolver(n_epochs=1_000)
    assert "n_epochs" in str(deco)


def test_map_deconvolver_result(deconvolver_result):
    assert_allclose(deconvolver_result.flux_total[12, 12], 1.858458, rtol=1e-3)
    assert_allclose(deconvolver_result.flux_total[0, 0], 0.272629, rtol=1e-3)

    trace_loss = deconvolver_result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 5.680209, rtol=1e-3)
    assert_allclose(trace_loss["dataset-0"], 1.904194, rtol=1e-3)
    assert_allclose(trace_loss["dataset-1"], 1.897707, rtol=1e-3)
    assert_allclose(trace_loss["dataset-2"], 1.878308, rtol=1e-3)


def test_map_deconvolver_result_io(deconvolver_result, tmpdir):
    filename = tmpdir / "result.fits"
    deconvolver_result.write(filename)

    result = MAPDeconvolverResult.read(filename=filename)

    assert result.config["n_epochs"] == 100
    assert_allclose(result.flux_total[12, 12], 1.858458, rtol=1e-3)
    assert_allclose(result.flux_total[0, 0], 0.272629, rtol=1e-3)


def test_map_deconvolver_result_plot(deconvolver_result):
    deconvolver_result.components.plot_fluxes()
    deconvolver_result.plot_trace_loss()


def test_map_deconvolver_usampling(datasets_disk):
    deco = MAPDeconvolver(
        n_epochs=100,
        learning_rate=0.1,
    )

    random_state = np.random.RandomState(642020)
    flux_init = random_state.gamma(20, size=(32, 32))

    components = FluxComponents()
    components["flux-1"] = FluxComponent.from_flux_init_numpy(
        flux_init=flux_init, upsampling_factor=2, prior=UniformPrior()
    )

    result = deco.run(datasets=datasets_disk, components=components)

    assert result.flux_upsampled_total.shape == (64, 64)
    assert result.components["flux-1"].upsampling_factor == 2
    assert_allclose(result.flux_total[12, 12], 3.620424, rtol=1e-3)
    assert_allclose(result.flux_total[0, 0], 0.849501, rtol=1e-3)

    trace_loss = result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 5.784892, rtol=1e-3)
    assert_allclose(trace_loss["dataset-0"], 1.919231, rtol=1e-3)
    assert_allclose(trace_loss["dataset-1"], 1.937949, rtol=1e-3)
    assert_allclose(trace_loss["dataset-2"], 1.927711, rtol=1e-3)


def test_map_deconvolver_inverse_gamma_prior(datasets_disk):
    deco = MAPDeconvolver(
        n_epochs=100,
        learning_rate=0.1,
    )

    random_state = np.random.RandomState(642020)
    flux_init = random_state.gamma(20, size=(32, 32))

    components = FluxComponents()
    components["flux-1"] = FluxComponent.from_flux_init_numpy(
        flux_init=flux_init, upsampling_factor=1, prior=InverseGammaPrior(alpha=10)
    )

    result = deco.run(datasets=datasets_disk, components=components)

    assert result.flux_upsampled_total.shape == (32, 32)
    assert_allclose(result.flux_total[12, 12], 0.136744, rtol=1e-3)
    assert_allclose(result.flux_total[0, 0], 0.135454, rtol=1e-3)

    trace_loss = result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 4.593177, rtol=1e-3)
    assert_allclose(trace_loss["dataset-0"], 1.743475, rtol=1e-3)
    assert_allclose(trace_loss["dataset-1"], 1.76312, rtol=1e-3)
    assert_allclose(trace_loss["dataset-2"], 1.737586, rtol=1e-3)

    assert_allclose(trace_loss["prior-flux-1"], 0.651003, rtol=1e-3)


def test_map_deconvolver_validation_datasets(datasets_disk):
    deco = MAPDeconvolver(
        n_epochs=100,
        learning_rate=0.1,
        stop_early_n_average=10,
    )

    random_state = np.random.RandomState(642020)
    flux_init = random_state.gamma(20, size=(32, 32))

    components = FluxComponents()
    components["flux-1"] = FluxComponent.from_flux_init_numpy(
        flux_init=flux_init, upsampling_factor=1, prior=ExponentialPrior(alpha=1)
    )

    result = deco.run(
        datasets=datasets_disk[:2],
        components=components,
        datasets_validation=datasets_disk[2:],
    )

    assert result.flux_upsampled_total.shape == (32, 32)
    assert_allclose(result.flux_total[12, 12], 1.3698, rtol=1e-3)
    assert_allclose(result.flux_total[0, 0], 0.320872, rtol=1e-3)

    trace_loss = result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 4.141166, rtol=1e-3)
    assert_allclose(trace_loss["dataset-0"], 1.846206, rtol=1e-3)
    assert_allclose(trace_loss["prior-flux-1"], -0.4010013, rtol=1e-3)
    assert_allclose(trace_loss["datasets-validation-total"], 1.881669, rtol=1e-3)
