import numpy as np
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel, convolve_fft

__all__ = [
    "point_source_gauss_psf",
    "disk_source_gauss_psf",
    "gauss_and_point_sources_gauss_psf",
]


BACKGROUND_LEVEL_DEFAULT = 2


def point_source_gauss_psf(
    shape=(32, 32),
    shape_psf=(17, 17),
    sigma_psf=3,
    source_level=1000,
    background_level=BACKGROUND_LEVEL_DEFAULT,
    random_state=None,
    dtype=np.float32,
):
    """Get point source with Gaussian PSF test data.

    The exposure is assumed to be constant.

    Parameters
    ----------
    shape : tuple
        Shape of the data array.
    shape_psf : tuple
        Shape of the psf array.
    sigma_psf : float
        Width of the psf in pixels.
    source_level : float
        Total integrated counts of the source
    background_level : float
        Background level in counts / pixel.
    random_state : `~numpy.random.RandomState`
        Random state
    dtype : `~numpy.dtype`
        Data type

    Returns
    -------
    data : dict of `~numpy.ndarray`
        Data dictionary
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    background = background_level * np.ones(shape)
    exposure = np.ones(shape)

    flux = np.zeros(shape)
    flux[shape[0] // 2, shape[1] // 2] = source_level

    psf = Gaussian2DKernel(sigma_psf, x_size=shape_psf[1], y_size=shape_psf[1])
    npred = background + convolve_fft(flux * exposure, psf)

    counts = random_state.poisson(npred)
    return {
        "counts": counts.astype(dtype),
        "psf": psf.array.astype(dtype),
        "exposure": exposure.astype(dtype),
        "background": background.astype(dtype),
        "flux": flux.astype(dtype),
    }


def disk_source_gauss_psf(
    shape=(32, 32),
    shape_psf=(17, 17),
    sigma_psf=3,
    source_level=1000,
    source_radius=3,
    background_level=BACKGROUND_LEVEL_DEFAULT,
    random_state=None,
    dtype=np.float32,
):
    """Get disk source with Gaussian PSF test data.

    The exposure has a gradient of 50% from left to right.

    Parameters
    ----------
    shape : tuple
        Shape of the data array.
    shape_psf : tuple
        Shape of the psf array.
    sigma_psf : float
        Width of the psf in pixels.
    source_level : float
        Total integrated counts of the source
    source_radius : float
        Radius of the disk source
    background_level : float
        Background level in counts / pixel.
    random_state : `~numpy.random.RandomState`
        Random state
    dtype : `~numpy.dtype`

    Returns
    -------
    data : dict of `~numpy.ndarray`
        Data dictionary
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    background = background_level * np.ones(shape)
    exposure = np.ones(shape) + 0.5 * np.linspace(-1, 1, shape[0])

    flux = (
        source_level
        * Tophat2DKernel(
            radius=source_radius, x_size=shape[1], y_size=shape[1], mode="oversample"
        ).array
    )

    psf = Gaussian2DKernel(sigma_psf, x_size=shape_psf[1], y_size=shape_psf[1])
    npred = background + convolve_fft(flux * exposure, psf)

    counts = random_state.poisson(npred)
    return {
        "counts": counts.astype(dtype),
        "psf": psf.array.astype(dtype),
        "exposure": exposure.astype(dtype),
        "background": background.astype(dtype),
        "flux": flux.astype(dtype),
    }


def gauss_and_point_sources_gauss_psf(
    shape=(32, 32),
    shape_psf=(17, 17),
    sigma_psf=2,
    source_level=1000,
    source_radius=2,
    background_level=BACKGROUND_LEVEL_DEFAULT,
    random_state=None,
    dtype=np.float32,
):
    """Get data with a Gaussian source in the center and point sources of
    varying brightness of 100%, 30%, 10% and 3% of the Gaussian source.

    The exposure has a gradient of 50% from top to bottom.

    Parameters
    ----------
    shape : tuple
        Shape of the data array.
    shape_psf : tuple
        Shape of the psf array.
    sigma_psf : float
        Width of the psf in pixels.
    source_level : float
        Total integrated counts of the source
    source_radius : float
        Radius of the disk source
    background_level : float
        Background level in counts / pixel.
    random_state : `~numpy.random.RandomState`
        Random state
    dtype : `~numpy.dtype`
        Data type

    Returns
    -------
    data : dict of `~numpy.ndarray`
        Data dictionary
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    background = background_level * np.ones(shape)
    exposure = np.ones(shape) + 0.5 * np.linspace(-1, 1, shape[0]).reshape((-1, 1))

    flux = (
        source_level
        * Gaussian2DKernel(
            source_radius, x_size=shape[1], y_size=shape[1], mode="oversample"
        ).array
    )

    for fraction, idx_x, idx_y in zip(
        [1, 0.3, 0.1, 0.03], [16, 16, 26, 6], [26, 6, 16, 16]
    ):
        flux[idx_y, idx_x] = fraction * source_level

    psf = Gaussian2DKernel(sigma_psf, x_size=shape_psf[1], y_size=shape_psf[1])
    npred = background + convolve_fft(flux * exposure, psf)

    counts = random_state.poisson(npred)
    return {
        "counts": counts.astype(dtype),
        "psf": psf.array.astype(dtype),
        "exposure": exposure.astype(dtype),
        "background": background.astype(dtype),
        "flux": flux.astype(dtype),
    }
