import abc
import torch
from .torch import interp1d_torch

__all__ = ["ImageNorm", "MaxImageNorm", "SigmoidImageNorm", "ATanImageNorm"]


class ImageNorm(abc.ABC):
    """Image normalisation"""

    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax

    @abc.abstractmethod
    def __call__(self, image):
        pass

    def plot(self, ax=None, xrange=None, **kwargs):
        """Plot image norm transfer function

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Plotting axes
        xrange : tuple of float
            Range of x pixel values
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.plot`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Plotting axes

        """
        import matplotlib.pyplot as plt

        if xrange is None:
            if isinstance(self, InverseCDFImageNorm):
                xrange = float(self.x[0]), float(self.x[-2])
            else:
                xrange = 0, 1

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("label", self.__class__.__name__)

        x = torch.linspace(xrange[0], xrange[1], 1000)
        y = self(image=x)
        ax.plot(x.detach().numpy(), y.detach().numpy(), **kwargs)

        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Scaled pixel value / A.U.")
        ax.set_ylim(0, 1)

        plt.legend()
        return ax


class MaxImageNorm(ImageNorm):
    """Max Image normalisation"""
    def __call__(self, image):
        return image / image.max()


class SigmoidImageNorm(ImageNorm):
    """Sigmoid image normalisation"""
    def __init__(self, alpha=1):
        self.alpha = torch.Tensor([alpha])

    def __call__(self, image):
        return 1 / (1 + torch.exp(-image / self.alpha))


class ATanImageNorm(ImageNorm):
    """ATan image normalisation"""
    def __init__(self, alpha=1):
        self.alpha = torch.Tensor([alpha])

    def __call__(self, image):
        return 2 * torch.atan(image / self.alpha) / torch.pi


class InverseCDFImageNorm(ImageNorm):
    """Inverse CDF image normalisation"""
    def __init__(self, x, cdf):
        if not x.shape == cdf.shape:
            raise ValueError(
                f"'x' and 'cdf' must have same shape, got {x.shape} and {cdf.shape}"
            )

        self.x = x
        self.cdf = cdf

    @classmethod
    def from_image(cls, image, bins=1000):
        """Create from an image"""
        image = torch.from_numpy(image)
        weights, x = torch.histogram(image, bins=bins)
        cdf = torch.cumsum(weights, 0)
        shifted = cdf - cdf.min()
        cdf = shifted / shifted.max()
        x_mean = (x[1:] + x[:-1]) / 2
        return cls(x=x_mean, cdf=cdf)

    def __call__(self, image):
        return interp1d_torch(image, self.x, self.cdf)


NORMS_REGISTRY = {
    "max": MaxImageNorm,
    "sigmoid": SigmoidImageNorm,
    "atan": ATanImageNorm,
    "inverse-cdf": InverseCDFImageNorm,
}
