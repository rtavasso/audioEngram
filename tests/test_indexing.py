"""
Tests for context window indexing.

Verifies that context extraction returns correct slices using hand-constructed data.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase0.features.context import (
    get_context_mean,
    get_context_flat,
    get_valid_frame_range,
    extract_context_features,
)


class TestContextExtraction:
    """Tests for context window extraction."""

    def test_get_context_mean_basic(self):
        """Test mean-pooled context with hand-constructed data."""
        # Create x where each frame equals its index
        # x[i] = [i, i, i, ...] for all dimensions
        T, D = 20, 4
        x = np.arange(T).reshape(-1, 1).repeat(D, axis=1).astype(np.float32)

        # Window size W=4, lag L=1, target t=10
        # Context should be x[10-1-4 : 10-1] = x[5:9]
        # Mean = mean([5,6,7,8]) = 6.5
        W, L, t = 4, 1, 10

        result = get_context_mean(x, t, W, L)

        expected_mean = 6.5  # (5+6+7+8)/4
        assert result.shape == (D,)
        assert np.allclose(result, expected_mean), f"Expected {expected_mean}, got {result[0]}"

    def test_get_context_mean_different_lags(self):
        """Test context extraction with different lag values."""
        T, D = 30, 2
        x = np.arange(T).reshape(-1, 1).repeat(D, axis=1).astype(np.float32)

        W = 4
        t = 15

        # Lag L=1: context = x[10:14], mean = 11.5
        result_l1 = get_context_mean(x, t, W, lag=1)
        assert np.allclose(result_l1, 11.5), f"Lag 1: expected 11.5, got {result_l1[0]}"

        # Lag L=2: context = x[9:13], mean = 10.5
        result_l2 = get_context_mean(x, t, W, lag=2)
        assert np.allclose(result_l2, 10.5), f"Lag 2: expected 10.5, got {result_l2[0]}"

        # Lag L=4: context = x[7:11], mean = 8.5
        result_l4 = get_context_mean(x, t, W, lag=4)
        assert np.allclose(result_l4, 8.5), f"Lag 4: expected 8.5, got {result_l4[0]}"

    def test_get_context_flat_basic(self):
        """Test flattened context with hand-constructed data."""
        T, D = 20, 3
        x = np.arange(T).reshape(-1, 1).repeat(D, axis=1).astype(np.float32)

        W, L, t = 4, 1, 10
        # Context = x[5:9] = frames 5,6,7,8
        # Flattened: [5,5,5, 6,6,6, 7,7,7, 8,8,8]

        result = get_context_flat(x, t, W, L)

        assert result.shape == (W * D,), f"Expected shape {(W*D,)}, got {result.shape}"

        # Check values
        expected = np.array([5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8], dtype=np.float32)
        assert np.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_context_invalid_range_raises(self):
        """Test that invalid context window raises error."""
        T, D = 10, 2
        x = np.zeros((T, D))

        W, L = 4, 2
        t = 3  # Too small: start = 3 - 2 - 4 = -3 < 0

        with pytest.raises(ValueError, match="Invalid context window"):
            get_context_mean(x, t, W, L)

    def test_get_valid_frame_range(self):
        """Test valid frame range computation."""
        # n_frames=50, W=8, max_lag=4
        # Need t >= W + max_lag = 12 and t >= 1
        # So first_valid = max(12, 1) = 12
        # last_valid = 50

        first, last = get_valid_frame_range(n_frames=50, window_size=8, max_lag=4)

        assert first == 12, f"Expected first=12, got {first}"
        assert last == 50, f"Expected last=50, got {last}"

    def test_extract_context_features_batch(self):
        """Test batch extraction of context features."""
        T, D = 30, 4
        x = np.arange(T).reshape(-1, 1).repeat(D, axis=1).astype(np.float32)

        W, L = 4, 1
        frame_indices = [10, 15, 20]

        # Mean mode
        result_mean = extract_context_features(x, frame_indices, W, L, mode="mean")
        assert result_mean.shape == (3, D)
        assert np.allclose(result_mean[0], 6.5)   # t=10: mean([5,6,7,8])
        assert np.allclose(result_mean[1], 11.5)  # t=15: mean([10,11,12,13])
        assert np.allclose(result_mean[2], 16.5)  # t=20: mean([15,16,17,18])

        # Flat mode
        result_flat = extract_context_features(x, frame_indices, W, L, mode="flat")
        assert result_flat.shape == (3, W * D)


class TestContextIndexingEdgeCases:
    """Edge case tests for context indexing."""

    def test_minimum_valid_frame(self):
        """Test context at minimum valid frame index."""
        T, D = 20, 2
        x = np.arange(T).reshape(-1, 1).repeat(D, axis=1).astype(np.float32)

        W, L = 4, 1
        t = W + L  # = 5, minimum valid

        result = get_context_mean(x, t, W, L)
        # Context = x[0:4], mean = 1.5
        assert np.allclose(result, 1.5)

    def test_maximum_valid_frame(self):
        """Test context at maximum valid frame index."""
        T, D = 20, 2
        x = np.arange(T).reshape(-1, 1).repeat(D, axis=1).astype(np.float32)

        W, L = 4, 1
        t = T - 1  # = 19, maximum valid

        result = get_context_mean(x, t, W, L)
        # Context = x[14:18], mean = 15.5
        assert np.allclose(result, 15.5)

    def test_different_window_sizes(self):
        """Test context with different window sizes."""
        T, D = 30, 2
        x = np.arange(T).reshape(-1, 1).repeat(D, axis=1).astype(np.float32)

        L, t = 1, 20

        # W=2: context = x[17:19], mean = 17.5
        assert np.allclose(get_context_mean(x, t, window_size=2, lag=L), 17.5)

        # W=8: context = x[11:19], mean = 14.5
        assert np.allclose(get_context_mean(x, t, window_size=8, lag=L), 14.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
