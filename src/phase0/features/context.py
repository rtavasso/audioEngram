"""
Context window extraction for conditioning.

Given utterance latents x[T, D], extracts context windows for target frames.
The context window ends at t-L and has length W, where:
- t is the target frame index
- L is the lag (1, 2, or 4 frames)
- W is the window size (8 frames)
"""

import numpy as np


def get_context_mean(
    x: np.ndarray,
    t: int,
    window_size: int,
    lag: int,
) -> np.ndarray:
    """
    Get mean-pooled context for a target frame.

    The context window is x[t-lag-window_size : t-lag], mean-pooled to [D].

    Args:
        x: Latent sequence [T, D]
        t: Target frame index
        window_size: Number of frames in context window (W)
        lag: Frames between context end and target (L)

    Returns:
        Mean-pooled context [D]
    """
    end = t - lag
    start = end - window_size

    if start < 0:
        raise ValueError(
            f"Invalid context window: t={t}, W={window_size}, L={lag} "
            f"gives start={start} < 0"
        )

    context = x[start:end]  # [W, D]
    return context.mean(axis=0)  # [D]


def get_context_flat(
    x: np.ndarray,
    t: int,
    window_size: int,
    lag: int,
) -> np.ndarray:
    """
    Get flattened context for a target frame.

    The context window is x[t-lag-window_size : t-lag], flattened to [W*D].

    Args:
        x: Latent sequence [T, D]
        t: Target frame index
        window_size: Number of frames in context window (W)
        lag: Frames between context end and target (L)

    Returns:
        Flattened context [W*D]
    """
    end = t - lag
    start = end - window_size

    if start < 0:
        raise ValueError(
            f"Invalid context window: t={t}, W={window_size}, L={lag} "
            f"gives start={start} < 0"
        )

    context = x[start:end]  # [W, D]
    return context.flatten()  # [W*D]


def get_valid_frame_range(
    n_frames: int,
    window_size: int,
    max_lag: int,
) -> tuple[int, int]:
    """
    Get valid frame range for context extraction.

    Frames must satisfy:
    - t >= window_size + max_lag (for valid context window)
    - t < n_frames (within utterance)
    - t >= 1 (for delta computation x[t] - x[t-1])

    Args:
        n_frames: Total frames in utterance
        window_size: Context window size (W)
        max_lag: Maximum lag to use (e.g., 4)

    Returns:
        Tuple of (first_valid_t, last_valid_t_exclusive)
    """
    first_valid = max(window_size + max_lag, 1)
    last_valid = n_frames
    return first_valid, last_valid


def extract_context_features(
    x: np.ndarray,
    frame_indices: list[int],
    window_size: int,
    lag: int,
    mode: str = "mean",
) -> np.ndarray:
    """
    Extract context features for multiple frames.

    Args:
        x: Latent sequence [T, D]
        frame_indices: List of target frame indices
        window_size: Context window size (W)
        lag: Lag between context end and target (L)
        mode: "mean" for mean-pooled [N, D], "flat" for flattened [N, W*D]

    Returns:
        Context features array
    """
    if mode == "mean":
        features = np.array([
            get_context_mean(x, t, window_size, lag)
            for t in frame_indices
        ])
    elif mode == "flat":
        features = np.array([
            get_context_flat(x, t, window_size, lag)
            for t in frame_indices
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return features
