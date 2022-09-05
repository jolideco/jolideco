import abc

import torch

from .torch import interp1d_torch

__all__ = [
    "ImageNorm",
    "MaxImageNorm",
    "SigmoidImageNorm",
    "ATanImageNorm",
    "FixedMaxImageNorm",
    "ASinhImageNorm",
]


class ImageNorm(abc.ABC):
    """Image normalisation"""

    def to_dict(self):
        """Convert deconvolver configuration to dict, with simple data types.

        Returns
        -------
        data : dict
            Parameter dict.
        """
        from jolideco.utils.norms import NORMS_REGISTRY

        data = {}

        for name, cls in NORMS_REGISTRY.items():
            if isinstance(self, cls):
                data["type"] = name
                break

        return data

    @classmethod
    def from_dict(cls, data):
        """Create from dict"""
        from jolideco.utils.norms import NORMS_REGISTRY

        kwargs = data.copy()

        if "type" in data:
            type_ = kwargs.pop("type")
            cls = NORMS_REGISTRY[type_]
            return cls.from_dict(kwargs)

        return cls(**kwargs)

    @abc.abstractmethod
    def __call__(self, image):
        pass

    def inverse(self, image):
        raise NotImplementedError

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


class ASinhImageNorm(ImageNorm):
    """Inverse hyperbolic sine image norm"""

    def __init__(self, alpha, beta):
        self.alpha = torch.Tensor([alpha])
        self.beta = torch.Tensor([beta])

    def __call__(self, image):
        top = torch.asinh(image / self.alpha)
        bottom = torch.asinh(self.beta / self.alpha)
        return top / bottom

    def inverse(self, image):
        value = image * torch.asinh(self.beta / self.alpha)
        return self.alpha * torch.sinh(value)

    def to_dict(self):
        """To dict"""
        data = super().to_dict()
        data["alpha"] = float(self.alpha)
        data["beta"] = float(self.beta)
        return data


class MaxImageNorm(ImageNorm):
    """Max image normalisation"""

    def __call__(self, image):
        return image / image.max()

    def inverse(self, image):
        return super().inverse(image)

    def to_dict(self):
        """To dict"""
        data = super().to_dict()
        return data


class FixedMaxImageNorm(ImageNorm):
    """Fixed max image normalisation"""

    def __init__(self, max_value):
        self.max_value = max_value

    def __call__(self, image):
        return torch.clip(image / self.max_value, min=0, max=1)

    def inverse(self, image):
        """Inverse image norm"""
        return image * self.max_value

    def to_dict(self):
        """To dict"""
        data = super().to_dict()
        data["max_value"] = float(self.max_value)
        return data


class SigmoidImageNorm(ImageNorm):
    """Sigmoid image normalisation"""

    def __init__(self, alpha=1):
        self.alpha = torch.Tensor([alpha])

    def __call__(self, image):
        return 1 / (1 + torch.exp(-image / self.alpha))

    def to_dict(self):
        """To dict"""
        data = super().to_dict()
        data["alpha"] = float(self.alpha)
        return data


class ATanImageNorm(ImageNorm):
    """ATan image normalisation"""

    def __init__(self, alpha=1):
        self.alpha = torch.Tensor([alpha])

    def __call__(self, image):
        return 2 * torch.atan(image / self.alpha) / torch.pi

    def inverse(self, image):
        """Inverse image norm"""
        return 0.5 * torch.pi * torch.tan(image)

    def to_dict(self):
        """To dict"""
        data = super().to_dict()
        data["alpha"] = float(self.alpha)
        return data


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

    def to_dict(self):
        """To dict"""
        raise NotImplementedError


NORMS_REGISTRY = {
    "max": MaxImageNorm,
    "fixed-max": FixedMaxImageNorm,
    "sigmoid": SigmoidImageNorm,
    "atan": ATanImageNorm,
    "inverse-cdf": InverseCDFImageNorm,
    "asinh": ASinhImageNorm,
}
