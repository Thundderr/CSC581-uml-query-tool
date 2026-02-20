"""Cross-platform device selection for PyTorch."""

import torch


def get_device() -> str:
    """Return the best available PyTorch device: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
