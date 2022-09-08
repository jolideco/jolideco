import torch


def requires_device(device):
    """Decorator to declare required gpu for tests."""
    import pytest

    if device == "cuda":
        skip_it = not torch.cuda.is_available()
    elif device == "mps":
        skip_it = not torch.backends.mps.is_available()
    else:
        raise ValueError(f"Not a valid device: '{device}'")

    reason = f"Missing support for backend {device}"
    return pytest.mark.skipif(skip_it, reason=reason)
