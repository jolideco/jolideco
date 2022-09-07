import torch


def requires_gpu():
    """Decorator to declare required gpu for tests."""
    import pytest

    skip_it = not torch.cuda.is_available()
    reason = "Missing GPU support"
    return pytest.mark.skipif(skip_it, reason=reason)
