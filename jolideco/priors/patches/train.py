import numpy as np
from astropy.table import Table
from jolideco.utils.numpy import view_as_overlapping_patches


def sklearn_gmm_to_table(gmm):
    """Convert scikit-learn GaussianMixture to table

    Parameters
    ----------
    gmm : `~sklearn.mixture.GaussianMixture`
        GMM model

    Returns
    -------
    table : `~astropy.table.Table`
        Table with columns `"means"`, `"covariances"` and `"weights"`.
    """
    table = Table()

    table["means"] = gmm.means_
    table["covariances"] = gmm.covariances_
    table["weights"] = gmm.weights_
    return table


def extract_patches_from_image(image, patch_shape=(8, 8), n_patches_max=300_000):
    """Extract normalized patches from image

    Parameters
    ----------
    image : `~numpy.ndarray`
        Image data as Numpy array
    patch_shape : tuple of int
        Patch shape
    n_patches_max : int
        Maximum number of patches to be used for training.

    Returns
    -------
    patches : `~numpy.ndarray`
        Array with flattened patches of shape (n_patches, patch_size),
        where patch_size is given by nx * ny.
    """
    patches = []

    for idx in range(patch_shape[0] // 2):
        for jdx in range(patch_shape[1] // 2):
            shifted = np.roll(image, shift=(idx, jdx))
            p = view_as_overlapping_patches(shifted, shape=(8, 8))
            valid = np.all(p > 0, axis=1)
            valid = valid & ~np.any(np.isnan(p), axis=1)
            patches.append(p[valid])

            if len(patches) >= n_patches_max:
                break

    patches = np.vstack(patches)
    return patches - patches.mean(axis=1, keepdims=True)


def train_gmm_from_patches(patches, n_components=100, **kwargs):
    """Train Gaussian mixture model from image data

    Parameters
    ----------
    patches : `~numpy.ndarray`
        Image data as Numpy array
    n_components: int
        Number of Gaussian components
    **kwargs : dict
        Keyword arguments forwarded to `~sklearn.mixture.GaussianMixture`

    Returns
    -------
    gmm : `~sklearn.mixture.GaussianMixture`
        Gaussian mixture model
    """
    from sklearn.mixture import GaussianMixture

    n_features = patches.shape[1]

    gmm = GaussianMixture(
        n_components=n_components,
        means_init=np.zeros((n_components, n_features)),
        **kwargs,
    )

    gmm.fit(X=patches)
    return gmm
