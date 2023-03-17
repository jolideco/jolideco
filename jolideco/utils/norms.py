import abc
import numpy as np
import torch
from .misc import format_class_str
from .torch import interp1d_torch

__all__ = [
    "ImageNorm",
    "MaxImageNorm",
    "SigmoidImageNorm",
    "ATanImageNorm",
    "FixedMaxImageNorm",
    "ASinhImageNorm",
    "LogImageNorm",
    "PowerImageNorm",
]


class PatchNorm(torch.nn.Module):
    """Patch normalisation"""

    @abc.abstractmethod
    def inverse(self, patches_normed):
        """Inverse normalisation

        Parameters
        ----------
        patches_normed : `~torch.Tensor`
            Normalised patches with shape (n_patches, patch_size * patch_size)

        Returns
        -------
        patches : `~torch.Tensor`
            Patches with shape (n_patches, patch_size * patch_size)
        """
        pass

    @abc.abstractmethod
    def __call__(self, patches):
        """Normalise patches

        Parameters
        ----------
        patches : `~torch.Tensor`
            Patches with shape (n_patches, patch_size * patch_size)

        Returns
        -------
        normed : `~torch.Tensor`
            Normalised patches with shape (n_patches, patch_size * patch_size)
        """
        pass

    def evaluate_numpy(self, patches):
        """Evaluate norm on numpy array"""
        patches = torch.from_numpy(patches.astype(np.float32))
        return self(patches).detach().numpy()

    def to_dict(self):
        """Convert deconvolver configuration to dict, with simple data types.

        Returns
        -------
        data : dict
            Parameter dict.
        """
        from jolideco.utils.norms import NORMS_PATCH_REGISTRY

        data = {}

        for name, cls in NORMS_PATCH_REGISTRY.items():
            if isinstance(self, cls):
                data["type"] = name
                break

        return data

    @classmethod
    def from_dict(cls, data):
        """Create from dict"""
        from jolideco.utils.norms import NORMS_PATCH_REGISTRY

        kwargs = data.copy()

        if "type" in data:
            type_ = kwargs.pop("type")
            cls = NORMS_PATCH_REGISTRY[type_]
            return cls.from_dict(kwargs)

        return cls(**kwargs)

    def __str__(self):
        """String representation"""
        return format_class_str(instance=self)


class SubtractMeanPatchNorm(PatchNorm):
    """Subtract mean patch normalisation from Zoran & Weiss"""

    def __call__(self, patches):
        patches_mean = torch.nanmean(patches, dim=1, keepdims=True)
        normed = patches - patches_mean
        return normed


class StandardizedSubtractMeanPatchNorm(PatchNorm):
    """Standardized subtract mean patch normalisation"""

    def __call__(self, patches):
        patches_mean = torch.nanmean(patches, dim=1, keepdims=True)
        normed = (patches - patches_mean) / patches_mean
        return normed


class ImageNorm(torch.nn.Module):
    """Image normalisation"""

    def __init__(self, frozen=False):
        super().__init__()
        self.frozen = frozen

    def parameters(self, recurse=True):
        """Parameter list"""
        if self.frozen:
            return []
        else:
            return super().parameters(recurse)

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

    def __str__(self):
        """String representation"""
        return format_class_str(instance=self)

    @abc.abstractmethod
    def __call__(self, image):
        pass

    def evaluate_numpy(self, image):
        """Evaluate norm on numpy array"""
        image = torch.from_numpy(image.astype(np.float32))
        return self(image).detach().numpy()

    def inverse_numpy(self, image):
        """Evaluate inverse norm on numpy array"""
        image = torch.from_numpy(image.astype(np.float32))
        return self.inverse(image).detach().numpy()

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

        x = np.linspace(xrange[0], xrange[1], 1000)
        y = self.evaluate_numpy(image=x)
        ax.plot(x, y, **kwargs)

        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Scaled pixel value / A.U.")
        ax.set_ylim(0, 1)

        plt.legend()
        return ax


class IdentityImageNorm(ImageNorm):
    """Identity image norm"""

    def __call__(self, image):
        return image

    def inverse(self, image):
        return image


class ASinhImageNorm(ImageNorm):
    """Inverse hyperbolic sine image norm"""

    def __init__(self, alpha=1.0, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = torch.nn.Parameter(torch.Tensor([alpha]))
        self.beta = torch.nn.Parameter(torch.Tensor([beta]))

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

    def __init__(self, max_value, **kwargs):
        super().__init__(**kwargs)
        self.max_value = torch.nn.Parameter(torch.Tensor([max_value]))

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

    def __init__(self, alpha=1, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = torch.nn.Parameter(torch.Tensor([alpha]))
        self.beta = torch.nn.Parameter(torch.Tensor([beta]))

    def __call__(self, image):
        return 1 / (1 + torch.exp(-(image - self.beta / 2.0) / self.alpha))

    def inverse(self, image):
        """Inverse image norm"""
        return self.alpha * torch.log(image / (1.0 - image)) + self.beta / 2.0

    def to_dict(self):
        """To dict"""
        data = super().to_dict()
        data["alpha"] = float(self.alpha)
        data["beta"] = float(self.beta)
        return data


class ATanImageNorm(ImageNorm):
    """ATan image normalisation"""

    def __init__(self, alpha=1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = torch.nn.Parameter(torch.Tensor([alpha]))

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
        super().__init__()
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


class LogImageNorm(ImageNorm):
    """Log image normalisation"""

    def __init__(self, alpha=1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = torch.nn.Parameter(torch.Tensor([alpha]))

    def __call__(self, image):
        return torch.log(image / self.alpha)

    def inverse(self, image):
        """Inverse image norm"""
        return self.alpha * torch.exp(image)

    def to_dict(self):
        """To dict"""
        data = super().to_dict()
        data["alpha"] = float(self.alpha)
        return data


class PowerImageNorm(ImageNorm):
    """Power image normalisation"""

    def __init__(self, alpha=1, beta=1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = torch.nn.Parameter(torch.Tensor([alpha]))
        self.register_buffer("beta", torch.Tensor([beta]))

    def __call__(self, image):
        return torch.pow(image / self.beta, self.alpha)

    def inverse(self, image):
        """Inverse image norm"""
        return self.beta * torch.pow(image, 1 / self.alpha)

    def to_dict(self):
        """To dict"""
        data = super().to_dict()
        data["alpha"] = float(self.alpha)
        data["beta"] = float(self.beta)
        return data


NORMS_REGISTRY = {
    "max": MaxImageNorm,
    "fixed-max": FixedMaxImageNorm,
    "sigmoid": SigmoidImageNorm,
    "atan": ATanImageNorm,
    "inverse-cdf": InverseCDFImageNorm,
    "asinh": ASinhImageNorm,
    "log": LogImageNorm,
    "power": PowerImageNorm,
    "identity": IdentityImageNorm,
}

NORMS_PATCH_REGISTRY = {
    "std-subtract-mean": StandardizedSubtractMeanPatchNorm,
    "subtract-mean": SubtractMeanPatchNorm,
}
