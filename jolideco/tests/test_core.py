import numpy as np
import pytest
from numpy.testing import assert_allclose

from jolideco.core import MAPDeconvolver, MAPDeconvolverResult
from jolideco.data import disk_source_gauss_psf, gauss_and_point_sources_gauss_psf
from jolideco.models import FluxComponents, SpatialFluxComponent
from jolideco.priors import GMMPatchPrior, InverseGammaPrior, UniformPrior
from jolideco.priors.core import ExponentialPrior
from jolideco.utils.norms import ASinhImageNorm
from jolideco.utils.testing import requires_device


@pytest.fixture(scope="session")
def datasets_gauss():
    datasets = {}

    random_state = np.random.RandomState(642020)

    for idx in range(3):
        dataset = gauss_and_point_sources_gauss_psf(
            random_state=random_state,
        )
        datasets[f"{idx}"] = dataset

    return datasets


@pytest.fixture(scope="session")
def datasets_disk():
    datasets = {}

    random_state = np.random.RandomState(642020)

    for idx in range(3):
        dataset = disk_source_gauss_psf(
            random_state=random_state,
        )
        datasets[f"{idx}"] = dataset

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
    components["flux-1"] = SpatialFluxComponent.from_numpy(
        flux=flux_init, prior=UniformPrior()
    )

    result = deco.run(datasets=datasets_gauss, components=components)
    return result


def test_map_deconvolver_str():
    deco = MAPDeconvolver(n_epochs=1_000)
    assert "n_epochs" in str(deco)


def test_map_deconvolver_result(deconvolver_result):
    assert_allclose(deconvolver_result.flux_total[12, 12], 1.542659, rtol=1e-3)
    assert_allclose(deconvolver_result.flux_total[0, 0], 3.927929, rtol=1e-3)

    trace_loss = deconvolver_result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 5.842237, rtol=1e-3)
    assert_allclose(trace_loss["dataset-0"], 1.956523, rtol=1e-3)
    assert_allclose(trace_loss["dataset-1"], 1.945902, rtol=1e-3)
    assert_allclose(trace_loss["dataset-2"], 1.939812, rtol=1e-3)


@pytest.mark.parametrize("format", ["fits", "asdf"])
def test_map_deconvolver_result_io(format, deconvolver_result, tmpdir):
    filename = tmpdir / "result.fits"
    deconvolver_result.write(filename, format=format)

    result = MAPDeconvolverResult.read(filename=filename, format=format)

    assert result.config["n_epochs"] == 100
    assert_allclose(result.flux_total[12, 12], 1.542659, rtol=1e-3)
    assert_allclose(result.flux_total[0, 0], 3.927929, rtol=1e-3)


def test_map_deconvolver_result_plot(deconvolver_result):
    deconvolver_result.components.plot()
    deconvolver_result.plot_trace_loss()


def test_map_deconvolver_usampling(datasets_disk):
    deco = MAPDeconvolver(
        n_epochs=100,
        learning_rate=0.1,
    )

    random_state = np.random.RandomState(642020)
    flux_init = random_state.gamma(20, size=(32, 32))

    components = FluxComponents()
    components["flux-1"] = SpatialFluxComponent.from_numpy(
        flux=flux_init, upsampling_factor=2, prior=UniformPrior()
    )

    result = deco.run(datasets=datasets_disk, components=components)

    assert result.flux_upsampled_total.shape == (64, 64)
    assert result.components["flux-1"].upsampling_factor == 2
    assert_allclose(result.flux_total[12, 12], 3.565998, rtol=1e-3)
    assert_allclose(result.flux_total[0, 0], 1.605782, rtol=1e-3)

    trace_loss = result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 5.844786, rtol=1e-3)
    assert_allclose(trace_loss["dataset-0"], 1.946759, rtol=1e-3)
    assert_allclose(trace_loss["dataset-1"], 1.958015, rtol=1e-3)
    assert_allclose(trace_loss["dataset-2"], 1.940012, rtol=1e-3)


def test_map_deconvolver_inverse_gamma_prior(datasets_disk):
    deco = MAPDeconvolver(
        n_epochs=100,
        learning_rate=0.1,
    )

    random_state = np.random.RandomState(642020)
    flux_init = random_state.gamma(20, size=(32, 32))

    components = FluxComponents()
    components["flux-1"] = SpatialFluxComponent.from_numpy(
        flux=flux_init, upsampling_factor=1, prior=InverseGammaPrior(alpha=10)
    )

    for name, dataset in datasets_disk.items():
        dataset["psf"] = {"flux-1": dataset["psf"]}

    result = deco.run(datasets=datasets_disk, components=components)

    assert result.flux_upsampled_total.shape == (32, 32)
    assert_allclose(result.flux_total[12, 12], 0.136798, rtol=1e-3)
    assert_allclose(result.flux_total[0, 0], 0.136563, rtol=1e-3)

    trace_loss = result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 3.478109, rtol=1e-3)
    assert_allclose(trace_loss["dataset-0"], 1.817045, rtol=1e-3)
    assert_allclose(trace_loss["dataset-1"], 1.825257, rtol=1e-3)
    assert_allclose(trace_loss["dataset-2"], 1.786648, rtol=1e-3)

    assert_allclose(trace_loss["prior-flux-1"], -1.950841, rtol=1e-3)


def test_map_deconvolver_validation_datasets(datasets_disk):
    deco = MAPDeconvolver(
        n_epochs=100,
        learning_rate=0.1,
        stop_early_n_average=10,
    )

    random_state = np.random.RandomState(642020)
    flux_init = random_state.gamma(20, size=(32, 32))

    components = FluxComponents()
    components["flux-1"] = SpatialFluxComponent.from_numpy(
        flux=flux_init, upsampling_factor=1, prior=ExponentialPrior(alpha=1)
    )

    datasets = {name: datasets_disk[name] for name in ["0", "1"]}
    datasets_validation = {name: datasets_disk[name] for name in ["2"]}

    result = deco.run(
        datasets=datasets,
        components=components,
        datasets_validation=datasets_validation,
    )

    assert result.flux_upsampled_total.shape == (32, 32)
    assert_allclose(result.flux_total[12, 12], 1.382768, rtol=1e-3)
    assert_allclose(result.flux_total[0, 0], 0.407479, rtol=1e-3)

    trace_loss = result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 4.66624, rtol=1e-3)
    assert_allclose(trace_loss["dataset-0"], 1.917588, rtol=1e-3)
    assert_allclose(trace_loss["prior-flux-1"], 0.825783, rtol=1e-3)
    assert_allclose(trace_loss["datasets-validation-total"], 1.888031, rtol=1e-3)


def test_map_deconvolver_gmm(datasets_disk):
    deco = MAPDeconvolver(n_epochs=10, learning_rate=0.1)

    random_state = np.random.RandomState(642020)
    flux_init = random_state.gamma(20, size=(32, 32))

    components = FluxComponents()
    prior = GMMPatchPrior(norm=ASinhImageNorm())
    components["flux-1"] = SpatialFluxComponent.from_numpy(
        flux=flux_init, upsampling_factor=2, prior=prior
    )

    result = deco.run(
        datasets=datasets_disk,
        components=components,
    )

    assert result.flux_upsampled_total.shape == (64, 64)
    assert_allclose(result.flux_total[12, 12], 10.796226, rtol=1e-2)
    assert_allclose(result.flux_total[0, 0], 10.553964, rtol=1e-2)

    trace_loss = result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 16.326, rtol=2e-3)
    assert_allclose(trace_loss["dataset-0"], 6.76425, rtol=1e-3)
    assert_allclose(trace_loss["prior-flux-1"], -4.093091, rtol=1e-2)


@pytest.mark.xfail
def test_map_deconvolver_gmm_odd_stride_jitter():
    random_state = np.random.RandomState(642020)

    dataset = gauss_and_point_sources_gauss_psf(
        random_state=random_state, shape=(37, 37)
    )

    deco = MAPDeconvolver(n_epochs=10, learning_rate=0.1)

    flux_init = random_state.gamma(20, size=(37, 37))

    components = FluxComponents()
    components["flux-1"] = SpatialFluxComponent.from_numpy(
        flux=flux_init, upsampling_factor=1, prior=GMMPatchPrior(stride=3, jitter=True)
    )

    result = deco.run(
        datasets={"dataset-1": dataset},
        components=components,
    )

    trace_loss = result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 5.180159, rtol=1e-3)


@requires_device("cuda")
def test_map_deconvolver_gpu():
    deco = MAPDeconvolver(n_epochs=100, learning_rate=0.1, device="cuda")

    random_state = np.random.RandomState(642020)
    flux_init = random_state.gamma(20, size=(32, 32))

    components = FluxComponents()
    components["flux-1"] = SpatialFluxComponent.from_numpy(
        flux=flux_init, upsampling_factor=1, prior=GMMPatchPrior()
    )

    datasets = {name: datasets_disk[name] for name in ["0", "1"]}
    datasets_validation = {name: datasets_disk[name] for name in ["2"]}

    result = deco.run(
        datasets=datasets,
        components=components,
        datasets_validation=datasets_validation,
    )

    assert result.flux_upsampled_total.shape == (32, 32)
    assert_allclose(result.flux_total[12, 12], 1.3698, rtol=1e-3)
    assert_allclose(result.flux_total[0, 0], 0.320872, rtol=1e-3)

    trace_loss = result.trace_loss[-1]
    assert_allclose(trace_loss["total"], 4.141166, rtol=1e-3)
    assert_allclose(trace_loss["dataset-0"], 1.846206, rtol=1e-3)
    assert_allclose(trace_loss["prior-flux-1"], -0.4010013, rtol=1e-3)
    assert_allclose(trace_loss["datasets-validation-total"], 1.881669, rtol=1e-3)
