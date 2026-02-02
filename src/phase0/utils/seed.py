"""
Reproducibility utilities.

Provides consistent seeding across all random number generators.
"""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Sets seed for:
    - Python random module
    - NumPy
    - PyTorch (if available)

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_rng(seed: int) -> np.random.Generator:
    """
    Get a numpy random generator with specified seed.

    Useful for thread-safe random number generation.

    Args:
        seed: Random seed

    Returns:
        numpy Generator instance
    """
    return np.random.default_rng(seed)
