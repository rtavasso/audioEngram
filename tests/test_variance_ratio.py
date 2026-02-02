"""
Tests for variance ratio metric computation.

Verifies computation with synthetic clusters of known variance.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase0.metrics.variance_ratio import (
    compute_total_sse,
    compute_within_cluster_sse,
    compute_variance_ratio,
    compute_variance_ratio_per_speaker,
    compute_variance_explained,
)


class TestVarianceRatioBasic:
    """Basic tests for variance ratio computation."""

    def test_total_sse_centered_data(self):
        """Test total SSE with data centered at origin."""
        # 100 samples, 2 dimensions
        # x ~ N(0, 1) in each dimension
        np.random.seed(42)
        N, D = 100, 2
        data = np.random.randn(N, D).astype(np.float32)

        sse = compute_total_sse(data)

        # For N(0,1), variance ≈ 1, so SSE ≈ N * D * 1 = 200
        # But we compute from sample mean, so actual SSE ≈ (N-1) * D
        assert sse > 0
        assert 150 < sse < 250  # Reasonable range for N=100, D=2

    def test_within_cluster_sse_single_cluster(self):
        """Test within-cluster SSE when all samples in one cluster."""
        np.random.seed(42)
        N, D = 100, 2
        data = np.random.randn(N, D).astype(np.float32)
        cluster_ids = np.zeros(N, dtype=np.int32)  # All in cluster 0

        total_sse = compute_total_sse(data)
        within_sse = compute_within_cluster_sse(data, cluster_ids)

        # With single cluster, within SSE = total SSE
        assert np.allclose(within_sse, total_sse)

    def test_within_cluster_sse_perfect_clustering(self):
        """Test within-cluster SSE with perfectly separated clusters."""
        # Create 2 clusters that are very tight
        np.random.seed(42)
        N_per_cluster = 50
        D = 2

        # Cluster 0: centered at (0, 0), very low variance
        cluster0 = np.random.randn(N_per_cluster, D) * 0.01

        # Cluster 1: centered at (10, 10), very low variance
        cluster1 = np.random.randn(N_per_cluster, D) * 0.01 + 10

        data = np.vstack([cluster0, cluster1]).astype(np.float32)
        cluster_ids = np.array([0] * N_per_cluster + [1] * N_per_cluster, dtype=np.int32)

        within_sse = compute_within_cluster_sse(data, cluster_ids)
        total_sse = compute_total_sse(data)

        # Within-cluster SSE should be much smaller than total
        ratio = within_sse / total_sse
        assert ratio < 0.01, f"Expected ratio < 0.01, got {ratio}"

    def test_variance_ratio_formula(self):
        """Test variance ratio = within_sse / total_sse."""
        np.random.seed(42)
        N, D = 100, 4
        data = np.random.randn(N, D).astype(np.float32)
        cluster_ids = np.random.randint(0, 5, N).astype(np.int32)

        result = compute_variance_ratio(data, cluster_ids)

        total_sse = compute_total_sse(data)
        within_sse = compute_within_cluster_sse(data, cluster_ids)
        expected_ratio = within_sse / total_sse

        assert np.allclose(result["variance_ratio"], expected_ratio)
        assert result["n_samples"] == N
        assert result["n_clusters"] == 5


class TestVarianceRatioWithFiltering:
    """Tests for variance ratio with cluster filtering."""

    def test_effective_clusters_filtering(self):
        """Test that only effective clusters are used."""
        np.random.seed(42)
        N, D = 200, 2
        data = np.random.randn(N, D).astype(np.float32)

        # Create cluster assignments with one small cluster
        cluster_ids = np.zeros(N, dtype=np.int32)
        cluster_ids[:150] = 0  # 150 samples
        cluster_ids[150:190] = 1  # 40 samples (below min_size=50)
        cluster_ids[190:] = 2  # 10 samples (below min_size=50)

        # Only cluster 0 is effective (>= 100 samples)
        effective_clusters = np.array([0])

        result = compute_variance_ratio(data, cluster_ids, effective_clusters)

        # Should only use the 150 samples in cluster 0
        assert result["n_samples"] == 150
        assert result["n_clusters"] == 1

    def test_consistent_sample_set(self):
        """Test that numerator and denominator use same samples."""
        np.random.seed(42)
        N, D = 200, 2
        data = np.random.randn(N, D).astype(np.float32)

        cluster_ids = np.zeros(N, dtype=np.int32)
        cluster_ids[:100] = 0
        cluster_ids[100:] = 1

        effective_clusters = np.array([0])

        result = compute_variance_ratio(data, cluster_ids, effective_clusters)

        # Manually compute on filtered data
        mask = cluster_ids == 0
        filtered_data = data[mask]

        total_sse = compute_total_sse(filtered_data)
        within_sse = compute_within_cluster_sse(filtered_data, cluster_ids[mask])

        assert np.allclose(result["sse_total"], total_sse)
        assert np.allclose(result["sse_within"], within_sse)


class TestVarianceRatioPerSpeaker:
    """Tests for per-speaker variance ratio."""

    def test_per_speaker_computation(self):
        """Test variance ratio computed separately per speaker."""
        np.random.seed(42)
        N_per_speaker = 100
        n_speakers = 3
        D = 2

        data_list = []
        speaker_ids_list = []
        cluster_ids_list = []

        for spk in range(n_speakers):
            # Each speaker has different variance
            spk_data = np.random.randn(N_per_speaker, D) * (spk + 1)
            data_list.append(spk_data)
            speaker_ids_list.append(np.full(N_per_speaker, spk))
            cluster_ids_list.append(np.random.randint(0, 3, N_per_speaker))

        data = np.vstack(data_list).astype(np.float32)
        speaker_ids = np.concatenate(speaker_ids_list).astype(np.int32)
        cluster_ids = np.concatenate(cluster_ids_list).astype(np.int32)

        result = compute_variance_ratio_per_speaker(data, cluster_ids, speaker_ids)

        assert result["n_speakers"] == n_speakers
        assert len(result["per_speaker"]) == n_speakers
        assert 0 < result["mean"] < 1


class TestVarianceExplained:
    """Tests for variance explained conversion."""

    def test_variance_explained_conversion(self):
        """Test variance_explained = 1 - variance_ratio."""
        assert compute_variance_explained(0.6) == pytest.approx(0.4)
        assert compute_variance_explained(0.0) == pytest.approx(1.0)
        assert compute_variance_explained(1.0) == pytest.approx(0.0)


class TestVarianceRatioSyntheticClusters:
    """Tests with synthetic clusters of known structure."""

    def test_known_variance_reduction(self):
        """Test with clusters that have known variance structure."""
        np.random.seed(42)
        N_per_cluster = 500
        K = 4
        D = 10

        # Create clusters with known properties
        # Each cluster is N(mu_k, sigma^2 I) with sigma = 0.5
        # Clusters are centered at different locations
        sigma_within = 0.5
        cluster_centers = np.random.randn(K, D) * 5  # Spread clusters

        data_list = []
        cluster_ids_list = []

        for k in range(K):
            cluster_data = np.random.randn(N_per_cluster, D) * sigma_within + cluster_centers[k]
            data_list.append(cluster_data)
            cluster_ids_list.append(np.full(N_per_cluster, k))

        data = np.vstack(data_list).astype(np.float32)
        cluster_ids = np.concatenate(cluster_ids_list).astype(np.int32)

        result = compute_variance_ratio(data, cluster_ids)

        # Within-cluster variance ≈ sigma_within^2 = 0.25
        # Total variance = within + between
        # Since clusters are spread with sigma=5, between variance >> within
        # So variance ratio should be small

        assert result["variance_ratio"] < 0.3, f"Expected ratio < 0.3, got {result['variance_ratio']}"

    def test_random_assignment_baseline(self):
        """Test that random cluster assignment gives ratio ≈ 1."""
        np.random.seed(42)
        N, D = 1000, 10
        K = 10

        data = np.random.randn(N, D).astype(np.float32)
        cluster_ids = np.random.randint(0, K, N).astype(np.int32)

        result = compute_variance_ratio(data, cluster_ids)

        # With random assignment, clusters don't reduce variance
        # Ratio should be close to 1.0
        assert 0.9 < result["variance_ratio"] < 1.1, \
            f"Expected ratio ≈ 1.0, got {result['variance_ratio']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
