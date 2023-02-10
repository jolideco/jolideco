import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.convolution import Gaussian2DKernel
import torch
from jolideco.models import (
    FluxComponents,
    NPredModel,
    SparseSpatialFluxComponent,
    SpatialFluxComponent,
)
from jolideco.priors import PRIOR_REGISTRY, UniformPrior


@pytest.fixture
def dataset():
    shape = (1, 1, 25, 25)
    exposure = torch.ones(shape)
    psf = Gaussian2DKernel(3).array
    return {
        "psf": torch.from_numpy(psf[None, None]),
        "exposure": exposure,
    }


@pytest.fixture
def dataset_zero_background():
    shape = (1, 1, 25, 25)
    exposure = torch.ones(shape)
    psf = Gaussian2DKernel(3).array
    return {
        "psf": torch.from_numpy(psf[None, None]),
        "exposure": exposure,
    }


@pytest.fixture
def dataset_3d():
    shape = (1, 3, 25, 25)
    exposure = torch.ones(shape)
    psf = np.stack([Gaussian2DKernel(_, x_size=25) for _ in [1, 2, 3]])
    return {
        "psf": torch.from_numpy(psf[None]),
        "exposure": exposure,
    }


@pytest.fixture
def dataset_3d_rmf():
    shape = (1, 3, 25, 25)
    exposure = torch.ones(shape)
    psf = np.stack([Gaussian2DKernel(_, x_size=25) for _ in [1, 2, 3]]).astype(
        np.float32
    )
    rmf = torch.ones((3, 1)) / 3.0
    return {
        "psf": torch.from_numpy(psf[None]),
        "exposure": exposure,
        "rmf": rmf,
    }


def test_simple_npred_model(dataset):
    flux_init = torch.zeros(dataset["exposure"].shape)
    flux_init[0, 0, 10, 10] = 1

    component = SpatialFluxComponent(flux_upsampled=flux_init)

    npred_model = NPredModel(**dataset)

    npred = npred_model(flux=component.flux)

    npred = npred.detach().numpy()[0, 0]
    assert_allclose(npred[10, 10], 0.017684, atol=1e-5)
    assert_allclose(npred.sum(), 1.0, atol=1e-3)


def test_simple_npred_model_sparse(dataset_zero_background):
    flux = torch.tensor([3.7, 2.1, 4.2])
    x_pos = torch.tensor([7.2, 12.1, 19.2])
    y_pos = torch.tensor([7.7, 3.1, 14.2])

    component = SparseSpatialFluxComponent(
        flux=flux, x_pos=x_pos, y_pos=y_pos, shape=(25, 25)
    )
    npred_model = NPredModel(**dataset_zero_background)

    npred = npred_model(flux=component.flux)

    npred = npred.detach().numpy()[0, 0]
    assert_allclose(npred[10, 10], 0.033733, atol=1e-5)
    assert_allclose(npred.sum(), 9.55952, rtol=1e-4)


def test_simple_npred_model_3d(dataset_3d):
    flux_init = torch.zeros(dataset_3d["exposure"].shape)
    flux_init[0, :, 12, 12] = 1

    component = SpatialFluxComponent(flux_upsampled=flux_init)
    npred_model = NPredModel(**dataset_3d)

    npred = npred_model(flux=component.flux)

    npred = npred.detach().numpy()[0]
    assert npred.shape == (3, 25, 25)
    assert_allclose(npred[0, 10, 10], 0.002915, rtol=1e-5)
    assert_allclose(npred.sum(), 3, rtol=1e-3)


def test_simple_npred_model_3d_rmf(dataset_3d_rmf):
    flux_init = torch.zeros(dataset_3d_rmf["exposure"].shape)
    flux_init[0, :, 12, 12] = 1

    component = SpatialFluxComponent(flux_upsampled=flux_init)
    npred_model = NPredModel(**dataset_3d_rmf)

    npred = npred_model(flux=component.flux)

    npred = npred.detach().numpy()[0]
    assert npred.shape == (1, 25, 25)
    assert_allclose(npred[0, 10, 10], 0.00963, rtol=1e-4)
    assert_allclose(npred.sum(), 1, rtol=2e-5)


@pytest.mark.parametrize("prior_class", PRIOR_REGISTRY.values())
@pytest.mark.parametrize("format", ["fits", "yaml", "asdf"])
def test_flux_component_io(prior_class, format, tmpdir):
    flux_init = torch.ones((1, 1, 32, 32))

    prior = prior_class()
    component = SpatialFluxComponent(
        flux_upsampled=flux_init,
        upsampling_factor=2,
        use_log_flux=False,
        frozen=True,
        prior=prior,
    )

    filename = tmpdir / f"test.{format}"

    component.write(filename=filename, format=format)

    component_new = SpatialFluxComponent.read(filename=filename, format=format)

    assert component.shape == component_new.shape
    assert component.upsampling_factor == component_new.upsampling_factor
    assert component.use_log_flux == component_new.use_log_flux
    assert isinstance(component_new.prior, prior_class)


@pytest.mark.parametrize("prior_class", PRIOR_REGISTRY.values())
@pytest.mark.parametrize("format", ["fits", "asdf", "yaml"])
def test_flux_components_io(prior_class, format, tmpdir):
    components = FluxComponents()

    flux_init = torch.ones((1, 1, 32, 32))

    components["flux-uniform"] = SpatialFluxComponent(
        flux_upsampled=flux_init,
        upsampling_factor=2,
        use_log_flux=False,
        frozen=False,
        prior=UniformPrior(),
    )

    prior = prior_class()
    components["flux-point"] = SpatialFluxComponent(
        flux_upsampled=flux_init,
        upsampling_factor=2,
        use_log_flux=False,
        frozen=False,
        prior=prior,
    )

    filename = tmpdir / f"test.{format}"

    components.write(filename=filename, format=format)

    components_new = FluxComponents.read(filename=filename, format=format)

    assert list(components_new) == ["flux-uniform", "flux-point"]


@pytest.mark.parametrize("format", ["fits"])
def test_sparse_flux_components_io(format, tmpdir):
    components = FluxComponents()

    flux = torch.ones((3))
    x_pos = torch.arange(3)
    y_pos = torch.arange(3) + 0.1

    components["flux-sparse"] = SparseSpatialFluxComponent(
        x_pos=x_pos,
        y_pos=y_pos,
        flux=flux,
        shape=(11, 9),
        use_log_flux=False,
        frozen=False,
    )

    filename = tmpdir / f"test.{format}"

    components.write(filename=filename, format=format)

    components_new = FluxComponents.read(filename=filename, format=format)

    assert list(components_new) == ["flux-sparse"]

    component = components_new["flux-sparse"]
    assert_allclose(component.x_pos_numpy, [0, 1, 2])
    assert_allclose(component.y_pos_numpy, [0.1, 1.1, 2.1])

    component.flux_numpy.shape == (11, 9)

    assert component.shape == (1, 1, 11, 9)
    assert not component.frozen
