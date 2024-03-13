from itertools import product

import numpy as np

__all__ = [
    "view_as_overlapping_patches",
    "split_datasets_validation",
    "reconstruct_from_overlapping_patches",
    "compute_precision_cholesky",
    "split_datasets_validation",
    "get_pixel_weights",
    "evaluate_trapez",
]


def compute_precision_cholesky(covariances):
    """Compute precision matrices"""
    from scipy import linalg

    shape = covariances.shape

    precisions_chol = np.empty(shape)

    for k, covariance in enumerate(covariances):
        try:
            cov_chol = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            raise ValueError(f"Cholesky decomposition failed for {covariance}")

        precisions_chol[k] = linalg.solve_triangular(
            cov_chol, np.eye(shape[1]), lower=True
        ).T

    return precisions_chol


def evaluate_trapez(x, width, slope):
    """One dimensional Trapezoid model function"""
    # Compute the four points where the trapezoid changes slope
    x2 = min(-width / 2.0, 0)
    x3 = max(width / 2.0, 0)
    x1 = x2 - 1.0 / slope
    x4 = x3 + 1.0 / slope

    # Compute model values in pieces between the change points
    range_a = np.logical_and(x >= x1, x < x2)
    range_b = np.logical_and(x >= x2, x < x3)
    range_c = np.logical_and(x >= x3, x < x4)
    val_a = slope * (x - x1)
    val_c = slope * (x4 - x)
    return np.select([range_a, range_b, range_c], [val_a, 1, val_c])


def get_pixel_weights(patch_shape, stride):
    """Compute pixel weights for overlapping patches

    Parameters
    ----------
    patch_shape : tuple of int
        Patch shape
    stride : int
        Stride of the patches.

    Returns
    -------
    weights : `~numpy.ndarray`
        Weights array
    """
    width = np.max(patch_shape)
    overlap = width - stride

    value = (width - 1.0) / 2

    x = np.linspace(-value, value, width)

    values = evaluate_trapez(x=x, width=(stride - overlap), slope=1.0 / overlap)
    weights = values * values[:, np.newaxis]
    weights = weights / weights.sum() * stride**2
    return weights


def view_as_overlapping_patches(image, shape, stride=None):
    """View as overlapping patches

    Parameters
    ----------
    image : `~numpy.ndarray`
        Image tensor
    shape : tuple of int
        Shape of the patches.

    Returns
    -------
    patches : `~numpy.ndarray`
        Array of overlapping patches of shape
        (n_patches, patch_shape_flat)

    """
    from skimage.util import view_as_windows

    if stride is None:
        stride = shape[0] // 2

    patches = view_as_windows(image, shape, step=stride)
    ncols = shape[0] * shape[1]
    return patches.reshape(-1, ncols)


def reconstruct_from_overlapping_patches(patches, image_shape, stride=None):
    """Reconstruct an image from overlapping patches.

    Parameters
    ----------
    patches : `~numpy.ndarray`
        Array of overlapping patches of shape
        (n_patches, patch_shape_y, patch_shape_x)
    image_shape : tuple of int
        Image shape
    stride : int
        Stride of the patches. By default it is half of the patch size.

    Returns
    -------
    image : `~numpy.ndarray`
        Image array

    """
    if stride is None:
        stride = patches.shape[-1] // 2

    image_height, image_width = image_shape
    patch_height, patch_width = patches.shape[1:]
    image = np.zeros(image_shape)

    # compute the dimensions of the patches array
    n_h = image_height - patch_height + 1
    n_w = image_width - patch_width + 1

    weights = get_pixel_weights(patch_shape=patches.shape[1:], stride=stride)

    for patch, (i, j) in zip(
        patches, product(range(0, n_h, stride), range(0, n_w, stride))
    ):
        image[i : i + patch_height, j : j + patch_width] += (  # noqa E203 and W503
            weights * patch
        )

    return image


def split_datasets_validation(datasets, n_validation, random_state=None):
    """Split datasets into training and validation datasets

    Parameters
    ----------
    datasets : dict of [str, dict]
        Dictionary containing a name of the dataset as key and a dictionary containing,
        the data like "counts", "psf", "background" and "exposure".
    n_validation : int
        Number of validation datasets
    random_state : `~numpy.random.RandomState`
        Random state

    Returns
    -------
    datasets_training : dict of [str, dict]
        Training datasets
    """
    if random_state is None:
        random_state = np.random.RandomState()

    names = list(datasets.keys())
    random_state.shuffle(names)

    names_training = names[n_validation:]
    names_validation = names[:n_validation]

    datasets_training = {name: datasets[name] for name in names_training}
    datasets_validation = {name: datasets[name] for name in names_validation}

    return {"datasets": datasets_training, "datasets_validation": datasets_validation}
