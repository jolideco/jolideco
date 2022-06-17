import torch

__all__ = [
    "ImageNorm",
    "MaxImageNorm",
    "SigmoidImageNorm",
    "ATanImageNorm"
]


class ImageNorm:
    """Image normalisation"""
    def __init__(self):
        pass

    def __call__(self, image):
        pass


class MaxImageNorm(ImageNorm):
    """Max Image normalisation"""
    def __call__(self, image):
        return image / image.max()


class SigmoidImageNorm(ImageNorm):
    """Sigmoid image normalisation"""
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, image):
        return 1 / (1 / + torch.exp(-image / self.alpha))


class ATanImageNorm(ImageNorm):
    """Max Image normalisation"""
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, image):
        return 2 * torch.atan(flux / self.alpha) / torch.pi

