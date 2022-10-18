from astropy.convolution import Gaussian2DKernel
from astropy.utils import lazyproperty
import torch
import torch.nn as nn
from jolideco.utils.misc import format_class_str
from jolideco.utils.torch import (
    TORCH_DEFAULT_DEVICE,
    convolve_fft_torch,
    cycle_spin_subpixel,
)

__all__ = [
    "Prior",
    "Priors",
    "UniformPrior",
    "ImagePrior",
    "SmoothnessPrior",
    "InverseGammaPrior",
    "ExponentialPrior",
]


class Prior(nn.Module):
    """Prior base class"""

    # TODO: this is a workaround for https://github.com/pytorch/pytorch/issues/43672
    # maybe remove the generator state from flux components?
    def __getstate__(self):
        state = self.__dict__.copy()
        generator = state.pop("generator", None)

        if generator:
            state["generator"] = generator.get_state()
            state["generator-device"] = generator.device

        return state

    def __setstate__(self, state):
        generator_state = state.pop("generator", None)
        generator_device = state.pop("generator-device", TORCH_DEFAULT_DEVICE)

        if generator_state is not None:
            generator = torch.Generator(device=generator_device)
            generator.set_state(generator_state)
            state["generator"] = generator

        self.__dict__ = state

    def to_dict(self):
        """Convert deconvolver configuration to dict, with simple data types.

        Returns
        -------
        data : dict
            Parameter dict.
        """
        from jolideco.priors import PRIOR_REGISTRY

        data = {}

        for name, cls in PRIOR_REGISTRY.items():
            if isinstance(self, cls):
                data["type"] = name
                break

        return data

    @classmethod
    def from_dict(cls, data):
        """Create from dict"""
        from jolideco.priors import PRIOR_REGISTRY

        kwargs = data.copy()

        if "type" in data:
            type_ = kwargs.pop("type")
            cls = PRIOR_REGISTRY[type_]
            return cls.from_dict(data=kwargs)

        return cls(**kwargs)

    def __str__(self):
        return format_class_str(instance=self)


class Priors(nn.ModuleDict):
    """Dict of mutiple priors"""

    def __call__(self, fluxes):
        """Evaluate all priors

        Parameters
        ----------
        fluxes : tuple of `~torch.Tensor`
            Tuple of flux tensors

        Returns
        -------
        log_prior : `~torch.tensor`
            Log prior value
        """
        value = 0

        for idx, prior in enumerate(self.values()):
            value += prior(flux=fluxes[idx])

        return value


class UniformPrior(Prior):
    """Uniform prior"""

    def __init__(self):
        super().__init__()

    def __call__(self, flux):
        """Evaluate the prior

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed flux

        Returns
        -------
        log_prior ; `~torch.tensor`
            Log prior set to zero.
        """
        return torch.tensor(0)


class InverseGammaPrior(Prior):
    """Sparse prior for point sources

    Defined by a product of inverse Gamma distributions. See e.g. [ref]_

    .. [ref] https://doi.org/10.1051/0004-6361/201323006


    To reproduce:

    .. code::

        from sympy import Symbol, Indexed, gamma, exp, Product, log
        from jolideco.utils.sympy import concrete_expand_log

        alpha = Symbol("alpha")
        beta = Symbol("beta")

        N = Symbol("N", integer=True, positive=True)
        i = Symbol("i", integer=True, positive=True)

        x = Indexed('x', i)

        inverse_gamma = beta ** alpha / gamma(alpha) * x ** (-alpha - 1) * exp(-beta / x)
        inverse_gamma

        like = Product(inverse_gamma, (i, 1, N))

        log_like = log(like)

        concrete_expand_log(log_like)


    Attributes
    ----------
    alpha : float
        Alpha parameter
    beta : float
        Beta parameter
    cycle_spin_subpix : bool
        Subpixel cycle spin.
    generator : `~torch.Generator`
        Random number generator

    """

    def __init__(self, alpha=10, beta=3 / 2, cycle_spin_subpix=False, generator=None):
        super().__init__()
        self.register_buffer("alpha", torch.Tensor([alpha]))
        self.register_buffer("beta", torch.Tensor([beta]))

        self.cycle_spin_subpix = cycle_spin_subpix

        if generator is None:
            generator = torch.Generator(TORCH_DEFAULT_DEVICE)

        self.generator = generator

    @lazyproperty
    def mean(self):
        """Mean of the distribution"""
        return self.beta / (self.alpha - 1)

    @lazyproperty
    def mode(self):
        """Mode of the distribution"""
        return self.beta / (self.alpha + 1)

    @lazyproperty
    def log_constant_term(self):
        """Log constant term"""
        value = self.alpha * torch.log(self.beta)
        value -= torch.lgamma(self.alpha)
        return float(value)

    def __call__(self, flux):
        """Evaluate the prior

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed point source flux

        Returns
        -------
        log_prior ; `~torch.tensor`
            Log prior value.
        """
        if self.cycle_spin_subpix:
            flux = cycle_spin_subpixel(image=flux, generator=self.generator)

        value = -self.beta / flux
        value += (-self.alpha - 1) * torch.log(flux)
        value_sum = torch.sum(value) / flux.numel() + self.log_constant_term
        return value_sum

    def to_dict(self):
        """Convert deconvolver configuration to dict, with simple data types.

        Returns
        -------
        data : dict
            Parameter dict.
        """
        data = super().to_dict()
        data["alpha"] = float(self.alpha)
        data["beta"] = float(self.beta)
        data["cycle_spin_subpix"] = bool(self.cycle_spin_subpix)
        return data


class ExponentialPrior(Prior):
    """Sparse prior for point sources

    Defined by a product of exponential distributions.

    To reproduce:

    .. code::

        from sympy import Symbol, Indexed, exp, Product, log
        from jolideco.utils.sympy import concrete_expand_log

        alpha = Symbol("alpha")

        N = Symbol("N", integer=True, positive=True)
        i = Symbol("i", integer=True, positive=True)

        x = Indexed('x', i)

        exponential = alpha * exp(-x * alpha)

        like = Product(exponential, (i, 1, N))

        log_like = log(like)

        concrete_expand_log(log_like)


    Attributes
    ----------
    alpha : float
        Alpha parameter
    cycle_spin_subpix : bool
        Subpixel cycle spin.
    generator : `~torch.Generator`
        Random number generator

    """

    def __init__(self, alpha=10, cycle_spin_subpix=False, generator=None):
        super().__init__()
        self.register_buffer("alpha", torch.Tensor([alpha]))

        self.cycle_spin_subpix = cycle_spin_subpix

        if generator is None:
            generator = torch.Generator(TORCH_DEFAULT_DEVICE)

        self.generator = generator

    @lazyproperty
    def mean(self):
        """Mean of the distribution"""
        return 1 / self.alpha

    @lazyproperty
    def mode(self):
        """Mode of the distribution"""
        return 0

    @lazyproperty
    def log_constant_term(self):
        """Log constant term"""
        return torch.log(self.alpha)

    def __call__(self, flux):
        """Evaluate the prior

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed point source flux

        Returns
        -------
        log_prior ; `~torch.tensor`
            Log prior value.
        """
        if self.cycle_spin_subpix:
            flux = cycle_spin_subpixel(image=flux, generator=self.generator)

        value = -self.alpha * flux
        value_sum = torch.sum(value) / flux.numel() + self.log_constant_term
        return value_sum

    def to_dict(self):
        """Convert deconvolver configuration to dict, with simple data types.

        Returns
        -------
        data : dict
            Parameter dict.
        """
        data = super().to_dict()
        data["alpha"] = float(self.alpha)
        data["cycle_spin_subpix"] = bool(self.cycle_spin_subpix)
        return data


class ImagePrior(Prior):
    """Image prior

    Parameters
    ----------
    flux_prior : `~pytorch.Tensor`
        Prior image
    flux_prior_error : `~pytorch.Tensor`
        Prior error image
    """

    def __init__(self, flux_prior, flux_prior_error=None):
        super().__init__()
        self.flux_prior = flux_prior
        self.flux_prior_error = flux_prior_error

    def __call__(self, flux):
        """Evaluate the prior

        Parameters
        ----------
        flux : `~pytorch.Tensor`
            Reconstructed flux
        """
        return ((flux - self.flux_prior) / self.flux_prior_error) ** 2

    def to_dict(self):
        """To dict"""
        raise NotImplementedError


class SmoothnessPrior(Prior):
    """Gradient based smoothness prior"""

    def __init__(self, width=2):
        super().__init__()
        self.width = width
        kernel = Gaussian2DKernel(width)
        self.kernel = torch.from_numpy(kernel.array[None, None])

    def __call__(self, flux):
        smooth = convolve_fft_torch(flux, self.kernel)
        return -torch.sum(flux * smooth)

    def to_dict(self):
        """Convert deconvolver configuration to dict, with simple data types.

        Returns
        -------
        data : dict
            Parameter dict.
        """
        data = super().to_dict()
        data["width"] = float(self.width)
        return data
