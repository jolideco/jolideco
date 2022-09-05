import numpy as np
import pytest
import torch
from astropy.convolution import Gaussian2DKernel
from numpy.testing import assert_allclose

from jolideco.models import FluxComponent, FluxComponents, NPredModel
from jolideco.priors import PRIOR_REGISTRY, UniformPrior


@pytest.fixture
def dataset():
    shape = (1, 1, 25, 25)
    exposure = torch.ones(shape)
    psf = Gaussian2DKernel(3).array
    background = torch.ones(shape)
    return {
        "psf": torch.from_numpy(psf[None, None]),
        "exposure": exposure,
        "background": background,
    }


@pytest.fixture
def dataset_3d():
    shape = (1, 3, 25, 25)
    exposure = torch.ones(shape)
    psf = np.stack([Gaussian2DKernel(_, x_size=25) for _ in [1, 2, 3]])
    background = torch.ones(shape)
    return {
        "psf": torch.from_numpy(psf[None]),
        "exposure": exposure,
        "background": background,
    }


@pytest.fixture
def dataset_3d_rmf():
    shape = (1, 3, 25, 25)
    exposure = torch.ones(shape)
    psf = np.stack([Gaussian2DKernel(_, x_size=25) for _ in [1, 2, 3]]).astype(
        np.float32
    )
    background = torch.zeros(shape)
    rmf = torch.ones((3, 1)) / 3.0
    return {
        "psf": torch.from_numpy(psf[None]),
        "exposure": exposure,
        "background": background,
        "rmf": rmf,
    }


def test_simple_npred_model(dataset):
    flux_init = torch.zeros(dataset["exposure"].shape)
    flux_init[0, 0, 10, 10] = 1

    component = FluxComponent(flux_upsampled=flux_init)
    npred_model = NPredModel(**dataset)

    npred = npred_model(flux=component.flux)

    npred = npred.detach().numpy()[0, 0]
    assert_allclose(npred[10, 10], 1.017218, rtol=1e-5)
    assert_allclose(npred.sum(), 513.039549, rtol=1e-5)


def test_simple_npred_model_3d(dataset_3d):
    flux_init = torch.zeros(dataset_3d["exposure"].shape)
    flux_init[0, :, 12, 12] = 1

    component = FluxComponent(flux_upsampled=flux_init)
    npred_model = NPredModel(**dataset_3d)

    npred = npred_model(flux=component.flux)

    npred = npred.detach().numpy()[0]
    assert npred.shape == (3, 25, 25)
    assert_allclose(npred[0, 10, 10], 1.002915, rtol=1e-5)
    assert_allclose(npred.sum(), 1653.52562, rtol=1e-5)


def test_simple_npred_model_3d_rmf(dataset_3d_rmf):
    flux_init = torch.zeros(dataset_3d_rmf["exposure"].shape)
    flux_init[0, :, 12, 12] = 1

    component = FluxComponent(flux_upsampled=flux_init)
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
    component = FluxComponent(
        flux_upsampled=flux_init,
        upsampling_factor=2,
        use_log_flux=False,
        frozen=True,
        prior=prior,
    )

    filename = tmpdir / f"test.{format}"

    component.write(filename=filename, format=format)

    component_new = FluxComponent.read(filename=filename, format=format)

    assert component.shape == component_new.shape
    assert component.upsampling_factor == component_new.upsampling_factor
    assert component.use_log_flux == component_new.use_log_flux
    assert isinstance(component_new.prior, prior_class)


@pytest.mark.parametrize("prior_class", PRIOR_REGISTRY.values())
@pytest.mark.parametrize("format", ["fits", "asdf", "yaml"])
def test_flux_components_io(prior_class, format, tmpdir):
    components = FluxComponents()

    flux_init = torch.ones((1, 1, 32, 32))

    components["flux-uniform"] = FluxComponent(
        flux_upsampled=flux_init,
        upsampling_factor=2,
        use_log_flux=False,
        frozen=False,
        prior=UniformPrior(),
    )

    prior = prior_class()
    components["flux-point"] = FluxComponent(
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
